//! All-MLP SegFormer decode head and differentiable half-pixel bilinear upsampling.

use crate::config::SegformerConfig;
use burn::{
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Device, Tensor, TensorData, activation},
};

/// Differentiable 2D bilinear upsampling matching PyTorch `align_corners = False`.
///
/// Expresses separable half-pixel interpolation as `W_h @ X @ W_w`, which executes through
/// backend GEMM kernels and preserves exact reverse-mode gradients on `Autodiff` devices.
pub fn bilinear_upsample_align_false(x: Tensor<4>, out_h: usize, out_w: usize) -> Tensor<4> {
    let [_, _, in_h, in_w] = x.dims();
    if in_h == out_h && in_w == out_w {
        return x;
    }
    let device = x.device();
    let y = if in_h == out_h {
        x
    } else {
        let weights_h = half_pixel_matrix(in_h, out_h, false);
        let wh = Tensor::<4>::from_data(TensorData::new(weights_h, [1, 1, out_h, in_h]), &device);
        wh.matmul(x)
    };
    if in_w == out_w {
        y
    } else {
        let weights_w = half_pixel_matrix(in_w, out_w, true);
        let ww = Tensor::<4>::from_data(TensorData::new(weights_w, [1, 1, in_w, out_w]), &device);
        y.matmul(ww)
    }
}

fn half_pixel_matrix(in_len: usize, out_len: usize, transpose: bool) -> Vec<f32> {
    let mut mat = vec![0.0_f32; in_len * out_len];
    for o in 0..out_len {
        let src = (o as f64 + 0.5) * (in_len as f64 / out_len as f64) - 0.5;
        let idx0 = src.floor().clamp(0.0, (in_len - 1) as f64) as usize;
        let idx1 = (src.floor() + 1.0).clamp(0.0, (in_len - 1) as f64) as usize;
        let t = (src - src.floor()).clamp(0.0, 1.0) as f32;
        let w0 = if src < 0.0 {
            1.0
        } else if src >= (in_len - 1) as f64 {
            0.0
        } else {
            1.0 - t
        };
        let w1 = 1.0 - w0;
        if transpose {
            mat[idx0 * out_len + o] += w0;
            mat[idx1 * out_len + o] += w1;
        } else {
            mat[o * in_len + idx0] += w0;
            mat[o * in_len + idx1] += w1;
        }
    }
    mat
}

/// All-MLP segmentation decode head (`SegformerDecodeHead`).
#[derive(Module, Debug)]
pub struct SegformerDecodeHead {
    /// Per-stage channel projections (`C_i -> decoder_hidden_size`).
    pub linear_c: Vec<Linear>,
    /// `1 × 1` fusion convolution (`4 * decoder_hidden_size -> decoder_hidden_size`, no bias).
    pub linear_fuse: Conv2d,
    /// Fusion batch normalization.
    pub batch_norm: BatchNorm,
    /// Classifier dropout (active only on `autodiff` devices).
    pub dropout: Dropout,
    /// `1 × 1` class projection (`decoder_hidden_size -> num_labels`).
    pub classifier: Conv2d,
}

impl SegformerDecodeHead {
    /// Initialize decode-head weights from `config` on `device`.
    pub fn new(config: &SegformerConfig, device: &Device) -> Self {
        let dec = config.decoder_hidden_size;
        let linear_c = config
            .hidden_sizes
            .iter()
            .map(|&in_dim| LinearConfig::new(in_dim, dec).init(device))
            .collect();
        let linear_fuse = Conv2dConfig::new([dec * 4, dec], [1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_bias(false)
            .init(device);
        let batch_norm = BatchNormConfig::new(dec).init(device);
        let dropout = DropoutConfig::new(config.classifier_dropout_prob).init();
        let classifier = Conv2dConfig::new([dec, config.num_labels()], [1, 1])
            .with_padding(PaddingConfig2d::Valid)
            .with_bias(true)
            .init(device);
        Self {
            linear_c,
            linear_fuse,
            batch_norm,
            dropout,
            classifier,
        }
    }

    /// Fuse `[stage0, stage1, stage2, stage3]` into `[B, num_labels, H / 4, W / 4]` logits.
    pub fn forward(&self, features: [Tensor<4>; 4]) -> Tensor<4> {
        let [b, _, h0, w0] = features[0].dims();
        let [_, dec] = self.linear_c[0].weight.dims();
        let mut projected = Vec::with_capacity(4);
        for (stage_feat, proj) in features.into_iter().zip(&self.linear_c) {
            let [_, _, hi, wi] = stage_feat.dims();
            let flat = stage_feat.flatten::<3>(2, 3).swap_dims(1, 2);
            let mapped = proj.forward(flat).swap_dims(1, 2).reshape([b, dec, hi, wi]);
            projected.push(bilinear_upsample_align_false(mapped, h0, w0));
        }
        // HuggingFace SegformerDecodeHead concatenates `all_hidden_states[::-1]` along channels.
        projected.reverse();
        let fused = self.linear_fuse.forward(Tensor::cat(projected, 1));
        let activated = activation::relu(self.batch_norm.forward(fused));
        let dropped = if activated.device().is_autodiff() {
            self.dropout.forward(activated)
        } else {
            activated
        };
        self.classifier.forward(dropped)
    }
}
