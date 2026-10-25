//! LoFTR encoder layers with linear attention, in self and cross order.
use super::layers::{LayerNorm, Linear};
use crate::{MatchError, Weights};
use burn::tensor::{
    Device, Tensor,
    activation::{elu, relu},
};

/// Head count of both transformers.
const HEADS: usize = 8;
/// Linear-attention denominator offset.
const EPSILON: f64 = 1e-6;

struct EncoderLayer {
    query: Linear,
    key: Linear,
    value: Linear,
    merge: Linear,
    mlp_in: Linear,
    mlp_out: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
}

impl EncoderLayer {
    fn load(
        weights: &mut Weights,
        name: &str,
        width: usize,
        device: &Device,
    ) -> Result<Self, MatchError> {
        let linear = |weights: &mut Weights, part: &str, shape: [usize; 2]| {
            Linear::load(weights, &format!("{name}.{part}"), shape, false, device)
        };
        Ok(Self {
            query: linear(weights, "q_proj", [width, width])?,
            key: linear(weights, "k_proj", [width, width])?,
            value: linear(weights, "v_proj", [width, width])?,
            merge: linear(weights, "merge", [width, width])?,
            mlp_in: linear(weights, "mlp.0", [2 * width, 2 * width])?,
            mlp_out: linear(weights, "mlp.2", [width, 2 * width])?,
            norm1: LayerNorm::load(weights, &format!("{name}.norm1"), width, device)?,
            norm2: LayerNorm::load(weights, &format!("{name}.norm2"), width, device)?,
        })
    }

    /// Update `x` `[N, L, C]` with messages from `source` `[N, S, C]`.
    fn forward(&self, x: Tensor<3>, source: Tensor<3>) -> Tensor<3> {
        let query = heads(self.query.forward(x.clone()));
        let key = heads(self.key.forward(source.clone()));
        let value = heads(self.value.forward(source));
        let message = merge_heads(linear_attention(query, key, value));
        let message = self.norm1.forward(self.merge.forward(message));
        let message = self.mlp_out.forward(relu(
            self.mlp_in
                .forward(Tensor::cat(vec![x.clone(), message], 2)),
        ));
        x + self.norm2.forward(message)
    }
}

/// `[N, L, H·D]` to `[N, H, L, D]`.
fn heads(x: Tensor<3>) -> Tensor<4> {
    let [n, length, width] = x.dims();
    x.reshape([n, length, HEADS, width / HEADS]).swap_dims(1, 2)
}

/// `[N, H, L, D]` to `[N, L, H·D]`.
fn merge_heads(x: Tensor<4>) -> Tensor<3> {
    let [n, heads, length, depth] = x.dims();
    x.swap_dims(1, 2).reshape([n, length, heads * depth])
}

/// Linear attention with the feature map `elu(x) + 1`, on `[N, H, L, D]` tensors.
///
/// The values are divided by the source length before the products and multiplied
/// after them, as in the trained model, which did this to avoid an fp16 overflow.
pub(crate) fn linear_attention(query: Tensor<4>, key: Tensor<4>, value: Tensor<4>) -> Tensor<4> {
    let length = value.dims()[2] as f64;
    let query = elu(query, 1.0) + 1.0;
    let key = elu(key, 1.0) + 1.0;
    let value = value / length;
    let key_value = key.clone().swap_dims(2, 3).matmul(value);
    let key_sum = key.sum_dim(2).swap_dims(2, 3);
    let normalizer = (query.clone().matmul(key_sum) + EPSILON).recip();
    query.matmul(key_value) * normalizer * length
}

/// Self and cross encoder layers that update two feature sequences together.
pub(crate) struct Transformer {
    layers: Vec<(EncoderLayer, Kind)>,
}

#[derive(Clone, Copy)]
enum Kind {
    SelfAttention,
    CrossAttention,
}

impl Transformer {
    /// Load `pairs` self and cross layer pairs named `{name}.layers.{i}`.
    pub(crate) fn load(
        weights: &mut Weights,
        name: &str,
        width: usize,
        pairs: usize,
        device: &Device,
    ) -> Result<Self, MatchError> {
        let mut layers = Vec::with_capacity(2 * pairs);
        for index in 0..2 * pairs {
            let kind = if index % 2 == 0 {
                Kind::SelfAttention
            } else {
                Kind::CrossAttention
            };
            layers.push((
                EncoderLayer::load(weights, &format!("{name}.layers.{index}"), width, device)?,
                kind,
            ));
        }
        Ok(Self { layers })
    }

    pub(crate) fn forward(&self, mut a: Tensor<3>, mut b: Tensor<3>) -> (Tensor<3>, Tensor<3>) {
        for (layer, kind) in &self.layers {
            match kind {
                Kind::SelfAttention => {
                    a = layer.forward(a.clone(), a);
                    b = layer.forward(b.clone(), b);
                }
                Kind::CrossAttention => {
                    a = layer.forward(a, b.clone());
                    b = layer.forward(b, a.clone());
                }
            }
        }
        (a, b)
    }
}
