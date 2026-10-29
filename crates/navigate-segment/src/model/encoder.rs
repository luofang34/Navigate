//! Hierarchical Mix Vision Transformer (MiT) encoder stages for SegFormer.

use crate::config::SegformerConfig;
use burn::{
    module::Module,
    nn::{
        LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Device, Tensor, activation},
};

/// Overlapping patch embedding at the start of one MiT encoder stage.
#[derive(Module, Debug)]
pub struct OverlapPatchEmbed {
    /// Strided patch convolution.
    pub proj: Conv2d,
    /// Post-projection token layer normalization.
    pub layer_norm: LayerNorm,
}

impl OverlapPatchEmbed {
    /// Initialize patch embedding weights on `device`.
    pub fn new(
        in_channels: usize,
        hidden_size: usize,
        patch_size: usize,
        stride: usize,
        ln_eps: f64,
        device: &Device,
    ) -> Self {
        let pad = patch_size / 2;
        let proj = Conv2dConfig::new([in_channels, hidden_size], [patch_size, patch_size])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(pad, pad, pad, pad))
            .with_bias(true)
            .init(device);
        let layer_norm = LayerNormConfig::new(hidden_size)
            .with_epsilon(ln_eps)
            .init(device);
        Self { proj, layer_norm }
    }

    /// Project `[B, C_in, H_in, W_in]` to `([B, H * W, C_out], H, W)`.
    pub fn forward(&self, x: Tensor<4>) -> (Tensor<3>, usize, usize) {
        let projected = self.proj.forward(x);
        let [_, _, h, w] = projected.dims();
        let tokens = projected.flatten::<3>(2, 3).swap_dims(1, 2);
        (self.layer_norm.forward(tokens), h, w)
    }
}

/// Spatial sequence reduction (`sr_ratio > 1`) inside efficient self-attention.
#[derive(Module, Debug)]
pub struct SequenceReduction {
    /// Strided reduction convolution.
    pub sr: Conv2d,
    /// Post-reduction token layer normalization.
    pub layer_norm: LayerNorm,
}

impl SequenceReduction {
    /// Initialize spatial reduction weights on `device`.
    pub fn new(hidden_size: usize, sr_ratio: usize, ln_eps: f64, device: &Device) -> Self {
        let sr = Conv2dConfig::new([hidden_size, hidden_size], [sr_ratio, sr_ratio])
            .with_stride([sr_ratio, sr_ratio])
            .with_padding(PaddingConfig2d::Valid)
            .with_bias(true)
            .init(device);
        let layer_norm = LayerNormConfig::new(hidden_size)
            .with_epsilon(ln_eps)
            .init(device);
        Self { sr, layer_norm }
    }

    /// Reduce `[B, H * W, C]` key/value sequence to `[B, (H / sr) * (W / sr), C]`.
    pub fn forward(&self, x: Tensor<3>, height: usize, width: usize) -> Tensor<3> {
        let [b, _, c] = x.dims();
        let spatial = x.swap_dims(1, 2).reshape([b, c, height, width]);
        let reduced = self.sr.forward(spatial);
        let tokens = reduced.flatten::<3>(2, 3).swap_dims(1, 2);
        self.layer_norm.forward(tokens)
    }
}

/// Efficient multi-head self-attention block.
#[derive(Module, Debug)]
pub struct EfficientSelfAttention {
    /// Query linear projection.
    pub query: Linear,
    /// Key linear projection.
    pub key: Linear,
    /// Value linear projection.
    pub value: Linear,
    /// Output linear projection.
    pub dense: Linear,
    /// Optional spatial sequence reduction (`None` when `sr_ratio == 1`).
    pub reduction: Option<SequenceReduction>,
    /// Attention head count.
    pub num_heads: usize,
    /// Per-head dimension (`hidden_size / num_heads`).
    pub head_dim: usize,
}

impl EfficientSelfAttention {
    /// Initialize attention weights on `device`.
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        sr_ratio: usize,
        ln_eps: f64,
        device: &Device,
    ) -> Self {
        let head_dim = hidden_size / num_heads;
        let reduction =
            (sr_ratio > 1).then(|| SequenceReduction::new(hidden_size, sr_ratio, ln_eps, device));
        Self {
            query: LinearConfig::new(hidden_size, hidden_size).init(device),
            key: LinearConfig::new(hidden_size, hidden_size).init(device),
            value: LinearConfig::new(hidden_size, hidden_size).init(device),
            dense: LinearConfig::new(hidden_size, hidden_size).init(device),
            reduction,
            num_heads,
            head_dim,
        }
    }

    /// Apply efficient self-attention over `[B, H * W, C]`.
    pub fn forward(&self, x: Tensor<3>, height: usize, width: usize) -> Tensor<3> {
        let [b, n, c] = x.dims();
        let q = self
            .query
            .forward(x.clone())
            .reshape([b, n, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let kv = match &self.reduction {
            Some(red) => red.forward(x, height, width),
            None => x,
        };
        let [_, n_kv, _] = kv.dims();
        let k = self
            .key
            .forward(kv.clone())
            .reshape([b, n_kv, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let v = self
            .value
            .forward(kv)
            .reshape([b, n_kv, self.num_heads, self.head_dim])
            .swap_dims(1, 2);
        let scale = (self.head_dim as f32).powf(-0.5);
        let scores = q.matmul(k.swap_dims(2, 3)).mul_scalar(scale);
        let probs = activation::softmax(scores, 3);
        let attended = probs.matmul(v).swap_dims(1, 2).reshape([b, n, c]);
        self.dense.forward(attended)
    }
}

/// Mix-FFN block with implicit `3 × 3` depthwise positional convolution.
#[derive(Module, Debug)]
pub struct MixFfn {
    /// Expansion linear layer (`C -> mlp_ratio * C`).
    pub dense1: Linear,
    /// Depthwise `3 × 3` convolution (`groups = hidden_features`).
    pub dwconv: Conv2d,
    /// Contraction linear layer (`mlp_ratio * C -> C`).
    pub dense2: Linear,
}

impl MixFfn {
    /// Initialize Mix-FFN weights on `device`.
    pub fn new(hidden_size: usize, mlp_ratio: usize, device: &Device) -> Self {
        let expanded = hidden_size * mlp_ratio;
        let dwconv = Conv2dConfig::new([expanded, expanded], [3, 3])
            .with_stride([1, 1])
            .with_padding(PaddingConfig2d::Explicit(1, 1, 1, 1))
            .with_groups(expanded)
            .with_bias(true)
            .init(device);
        Self {
            dense1: LinearConfig::new(hidden_size, expanded).init(device),
            dwconv,
            dense2: LinearConfig::new(expanded, hidden_size).init(device),
        }
    }

    /// Apply `dense1 -> dwconv -> gelu -> dense2` on `[B, H * W, C]`.
    pub fn forward(&self, x: Tensor<3>, height: usize, width: usize) -> Tensor<3> {
        let hidden = self.dense1.forward(x);
        let [b, _, c] = hidden.dims();
        let spatial = hidden.swap_dims(1, 2).reshape([b, c, height, width]);
        let conv = self
            .dwconv
            .forward(spatial)
            .flatten::<3>(2, 3)
            .swap_dims(1, 2);
        self.dense2.forward(activation::gelu(conv))
    }
}

/// One MiT transformer layer inside an encoder stage.
#[derive(Module, Debug)]
pub struct SegformerLayer {
    /// Pre-attention layer norm.
    pub layer_norm_1: LayerNorm,
    /// Efficient self-attention block.
    pub attention: EfficientSelfAttention,
    /// Pre-MLP layer norm.
    pub layer_norm_2: LayerNorm,
    /// Mix-FFN block.
    pub mlp: MixFfn,
}

impl SegformerLayer {
    /// Initialize transformer layer weights on `device`.
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        sr_ratio: usize,
        mlp_ratio: usize,
        ln_eps: f64,
        device: &Device,
    ) -> Self {
        Self {
            layer_norm_1: LayerNormConfig::new(hidden_size)
                .with_epsilon(ln_eps)
                .init(device),
            attention: EfficientSelfAttention::new(
                hidden_size,
                num_heads,
                sr_ratio,
                ln_eps,
                device,
            ),
            layer_norm_2: LayerNormConfig::new(hidden_size)
                .with_epsilon(ln_eps)
                .init(device),
            mlp: MixFfn::new(hidden_size, mlp_ratio, device),
        }
    }

    /// Execute residual attention and Mix-FFN on `[B, H * W, C]`.
    pub fn forward(&self, x: Tensor<3>, height: usize, width: usize) -> Tensor<3> {
        let x = x.clone()
            + self
                .attention
                .forward(self.layer_norm_1.forward(x), height, width);
        x.clone()
            + self
                .mlp
                .forward(self.layer_norm_2.forward(x), height, width)
    }
}

/// One hierarchical MiT stage producing `[B, C_stage, H_stage, W_stage]`.
#[derive(Module, Debug)]
pub struct SegformerStage {
    /// Overlap patch embedding.
    pub patch_embed: OverlapPatchEmbed,
    /// Ordered transformer layers.
    pub blocks: Vec<SegformerLayer>,
    /// Stage output layer normalization.
    pub layer_norm: LayerNorm,
}

impl SegformerStage {
    /// Execute patch embedding, transformer blocks, and output reshape to `[B, C, H, W]`.
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let (mut tokens, h, w) = self.patch_embed.forward(x);
        for block in &self.blocks {
            tokens = block.forward(tokens, h, w);
        }
        let normed = self.layer_norm.forward(tokens);
        let [b, _, c] = normed.dims();
        normed.reshape([b, h, w, c]).permute([0, 3, 1, 2])
    }
}

/// Four-stage hierarchical MiT encoder.
#[derive(Module, Debug)]
pub struct SegformerEncoder {
    /// Four hierarchical feature stages (`H/4, H/8, H/16, H/32`).
    pub stages: Vec<SegformerStage>,
}

impl SegformerEncoder {
    /// Initialize encoder stages from `config` on `device`.
    pub fn new(config: &SegformerConfig, device: &Device) -> Self {
        let mut stages = Vec::with_capacity(4);
        for idx in 0..4 {
            let in_ch = if idx == 0 {
                config.num_channels
            } else {
                config.hidden_sizes[idx - 1]
            };
            let hidden = config.hidden_sizes[idx];
            let patch_embed = OverlapPatchEmbed::new(
                in_ch,
                hidden,
                config.patch_sizes[idx],
                config.strides[idx],
                config.layer_norm_eps,
                device,
            );
            let blocks = (0..config.depths[idx])
                .map(|_| {
                    SegformerLayer::new(
                        hidden,
                        config.num_attention_heads[idx],
                        config.sr_ratios[idx],
                        config.mlp_ratios[idx],
                        config.layer_norm_eps,
                        device,
                    )
                })
                .collect();
            let layer_norm = LayerNormConfig::new(hidden)
                .with_epsilon(config.layer_norm_eps)
                .init(device);
            stages.push(SegformerStage {
                patch_embed,
                blocks,
                layer_norm,
            });
        }
        Self { stages }
    }

    /// Run all four stages and return `[stage0, stage1, stage2, stage3]`.
    pub fn forward(&self, pixel_values: Tensor<4>) -> [Tensor<4>; 4] {
        let s0 = self.stages[0].forward(pixel_values);
        let s1 = self.stages[1].forward(s0.clone());
        let s2 = self.stages[2].forward(s1.clone());
        let s3 = self.stages[3].forward(s2.clone());
        [s0, s1, s2, s3]
    }
}
