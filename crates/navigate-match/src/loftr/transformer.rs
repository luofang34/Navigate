//! LoFTR encoder layers with linear attention, in self and cross order.
use super::layers::{LayerNorm, Linear, upload};
use crate::{
    MatchError, Weights,
    kernels::{Layout, product},
};
use burn::tensor::{Device, Tensor, activation::elu};

/// Head count of both transformers.
const HEADS: usize = 8;
/// Linear-attention denominator offset.
const EPSILON: f64 = 1e-6;

#[derive(Clone)]
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
    fn forward(&self, x: Tensor<3>, source: Tensor<3>, heads: &Heads) -> Tensor<3> {
        let query = self.query.forward(x.clone());
        let key = self.key.forward(source.clone());
        let value = self.value.forward(source);
        let message = linear_attention(query, key, value, heads);
        let message = self.norm1.forward(self.merge.forward(message));
        let hidden = self
            .mlp_in
            .forward_relu(Tensor::cat(vec![x.clone(), message], 2));
        x + self.norm2.forward(self.mlp_out.forward(hidden))
    }
}

/// Head structure of linear attention as matrices, so that no head reshapes are needed.
///
/// With `mask` block-diagonal (one block of ones for each head), the full
/// `Kᵀ V · mask` holds every head's `Kᵀ V` on the diagonal. `Q · (Kᵀ V · mask)` then
/// gives each head's message in its own channels, and `(Q ⊙ ΣK) · mask` gives each
/// channel the normalizer of its head.
#[derive(Clone)]
pub(crate) struct Heads {
    mask: Tensor<3>,
    zeros: Tensor<1>,
}

impl Heads {
    pub(crate) fn new(width: usize, device: &Device) -> Self {
        let depth = width / HEADS;
        let mask = (0..width * width)
            .map(|i| {
                if (i / width) / depth == (i % width) / depth {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        Self {
            mask: upload(mask, [1, width, width], device),
            zeros: upload(vec![0.0; width], [width], device),
        }
    }
}

/// Linear attention with the feature map `elu(x) + 1` on `[N, L, C]` tensors.
///
/// The values are divided by the source length before the products and multiplied
/// after them, as in the trained model, which did this to avoid an fp16 overflow.
pub(crate) fn linear_attention(
    query: Tensor<3>,
    key: Tensor<3>,
    value: Tensor<3>,
    heads: &Heads,
) -> Tensor<3> {
    let length = value.dims()[1] as f64;
    let query = elu(query, 1.0) + 1.0;
    let key = elu(key, 1.0) + 1.0;
    let value = value / length;
    let transposed = Layout {
        trans_a: true,
        ..Layout::default()
    };
    let key_value =
        product(key.clone(), value, heads.zeros.clone(), transposed) * heads.mask.clone();
    let key_sum = key.sum_dim(1);
    let normalizer = product(
        query.clone() * key_sum,
        heads.mask.clone(),
        heads.zeros.clone(),
        Layout::default(),
    );
    let message = product(query, key_value, heads.zeros.clone(), Layout::default());
    message / (normalizer + EPSILON) * length
}

/// Self and cross encoder layers that update two feature sequences together.
#[derive(Clone)]
pub(crate) struct Transformer {
    layers: Vec<(EncoderLayer, Kind)>,
    heads: Heads,
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
        Ok(Self {
            layers,
            heads: Heads::new(width, device),
        })
    }

    pub(crate) fn forward(&self, mut a: Tensor<3>, mut b: Tensor<3>) -> (Tensor<3>, Tensor<3>) {
        for (layer, kind) in &self.layers {
            match kind {
                Kind::SelfAttention => {
                    a = layer.forward(a.clone(), a, &self.heads);
                    b = layer.forward(b.clone(), b, &self.heads);
                }
                Kind::CrossAttention => {
                    a = layer.forward(a, b.clone(), &self.heads);
                    b = layer.forward(b, a.clone(), &self.heads);
                }
            }
        }
        (a, b)
    }
}
