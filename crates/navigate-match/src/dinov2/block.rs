//! One pre-norm transformer block with LayerScale.
use crate::{
    MatchError, Weights,
    loftr::layers::{LayerNorm, Linear, upload},
};
use burn::tensor::{Device, Tensor, activation};

#[derive(Clone)]
pub(super) struct Block {
    norm1: LayerNorm,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    scale1: Tensor<1>,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    scale2: Tensor<1>,
    heads: usize,
}

impl Block {
    pub(super) fn load(
        weights: &mut Weights,
        name: &str,
        width: usize,
        heads: usize,
        epsilon: f64,
        device: &Device,
    ) -> Result<Self, MatchError> {
        let attention = format!("{name}.attention.attention");
        let linear = |weights: &mut Weights, n: &str, shape: [usize; 2]| {
            Linear::load(weights, n, shape, true, device)
        };
        let scale = |weights: &mut Weights, n: &str| -> Result<Tensor<1>, MatchError> {
            Ok(upload(
                weights.take(&format!("{name}.{n}.lambda1"), &[width])?,
                [width],
                device,
            ))
        };
        Ok(Self {
            norm1: LayerNorm::load_with_epsilon(
                weights,
                &format!("{name}.norm1"),
                width,
                epsilon,
                device,
            )?,
            query: linear(weights, &format!("{attention}.query"), [width, width])?,
            key: linear(weights, &format!("{attention}.key"), [width, width])?,
            value: linear(weights, &format!("{attention}.value"), [width, width])?,
            output: linear(
                weights,
                &format!("{name}.attention.output.dense"),
                [width, width],
            )?,
            scale1: scale(weights, "layer_scale1")?,
            norm2: LayerNorm::load_with_epsilon(
                weights,
                &format!("{name}.norm2"),
                width,
                epsilon,
                device,
            )?,
            fc1: linear(weights, &format!("{name}.mlp.fc1"), [4 * width, width])?,
            fc2: linear(weights, &format!("{name}.mlp.fc2"), [width, 4 * width])?,
            scale2: scale(weights, "layer_scale2")?,
            heads,
        })
    }

    /// `x` is `[batch, tokens, width]`.
    pub(super) fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [batch, tokens, width] = x.dims();
        let head = width / self.heads;
        let split = |t: Tensor<3>| t.reshape([batch, tokens, self.heads, head]).swap_dims(1, 2);
        let normed = self.norm1.forward(x.clone());
        let q = split(self.query.forward(normed.clone()));
        let k = split(self.key.forward(normed.clone()));
        let v = split(self.value.forward(normed));
        let scores = q.matmul(k.swap_dims(2, 3)) / (head as f64).sqrt();
        let attended = activation::softmax(scores, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, tokens, width]);
        let x = x + self.output.forward(attended) * self.scale1.clone().unsqueeze::<3>();
        let hidden = activation::gelu(self.fc1.forward(self.norm2.forward(x.clone())));
        x + self.fc2.forward(hidden) * self.scale2.clone().unsqueeze::<3>()
    }
}
