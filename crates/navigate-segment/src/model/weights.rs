//! HuggingFace SafeTensors weight loading and saving for [`super::Segformer`].

use super::{
    Segformer,
    decoder::SegformerDecodeHead,
    encoder::{
        EfficientSelfAttention, MixFfn, OverlapPatchEmbed, SegformerEncoder, SegformerLayer,
        SegformerStage, SequenceReduction,
    },
};
use crate::{
    SegmentError,
    config::{PreprocessorConfig, SegformerConfig},
};
use burn::{
    module::{Flag, Param, RunningState},
    nn::{BatchNorm, DropoutConfig, LayerNorm, Linear, PaddingConfig2d, conv::Conv2d},
    tensor::{Device, Tensor, TensorData},
};
use safetensors::{Dtype, SafeTensors};
use std::{fs, path::Path};

struct Archive<'a> {
    tensors: SafeTensors<'a>,
}

impl<'a> Archive<'a> {
    fn open(bytes: &'a [u8]) -> Result<Self, SegmentError> {
        let tensors = SafeTensors::deserialize(bytes)
            .map_err(|e| SegmentError::Safetensors(e.to_string()))?;
        Ok(Self { tensors })
    }

    fn f32_vec(&self, name: &str, expected: &[usize]) -> Result<Vec<f32>, SegmentError> {
        let view = self
            .tensors
            .tensor(name)
            .map_err(|_| SegmentError::Safetensors(format!("missing tensor '{name}'")))?;
        if view.shape() != expected {
            return Err(SegmentError::Shape {
                name: name.into(),
                expected: expected.to_vec(),
                actual: view.shape().to_vec(),
            });
        }
        match view.dtype() {
            Dtype::F32 => Ok(view
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect()),
            other => Err(SegmentError::Safetensors(format!(
                "unsupported dtype {other:?} for '{name}'"
            ))),
        }
    }

    fn linear(
        &self,
        prefix: &str,
        d_in: usize,
        d_out: usize,
        device: &Device,
    ) -> Result<Linear, SegmentError> {
        let raw_w = self.f32_vec(&format!("{prefix}.weight"), &[d_out, d_in])?;
        let mut row_w = vec![0.0_f32; d_in * d_out];
        for o in 0..d_out {
            for i in 0..d_in {
                row_w[i * d_out + o] = raw_w[o * d_in + i];
            }
        }
        let bias = self.f32_vec(&format!("{prefix}.bias"), &[d_out])?;
        Ok(Linear {
            weight: Param::from_data(TensorData::new(row_w, [d_in, d_out]), device),
            bias: Some(Param::from_data(TensorData::new(bias, [d_out]), device)),
        })
    }

    fn layer_norm(
        &self,
        prefix: &str,
        dim: usize,
        eps: f64,
        device: &Device,
    ) -> Result<LayerNorm, SegmentError> {
        let mut ln = burn::nn::LayerNormConfig::new(dim)
            .with_epsilon(eps)
            .init(device);
        let w = self.f32_vec(&format!("{prefix}.weight"), &[dim])?;
        let b = self.f32_vec(&format!("{prefix}.bias"), &[dim])?;
        ln.gamma = Param::from_data(TensorData::new(w, [dim]), device);
        ln.beta = Some(Param::from_data(TensorData::new(b, [dim]), device));
        Ok(ln)
    }

    #[allow(clippy::too_many_arguments)]
    fn conv2d(
        &self,
        prefix: &str,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        pad: usize,
        groups: usize,
        has_bias: bool,
        device: &Device,
    ) -> Result<Conv2d, SegmentError> {
        let shape = [out_ch, in_ch / groups, kernel, kernel];
        let w = self.f32_vec(&format!("{prefix}.weight"), &shape)?;
        let bias = if has_bias {
            let b = self.f32_vec(&format!("{prefix}.bias"), &[out_ch])?;
            Some(Param::from_data(TensorData::new(b, [out_ch]), device))
        } else {
            None
        };
        let padding = if pad == 0 {
            PaddingConfig2d::Valid
        } else {
            PaddingConfig2d::Explicit(pad, pad, pad, pad)
        };
        Ok(Conv2d {
            weight: Param::from_data(TensorData::new(w, shape), device),
            bias,
            stride: [stride, stride],
            kernel_size: [kernel, kernel],
            dilation: [1, 1],
            groups,
            padding,
        })
    }
}

/// Load a HuggingFace `SegformerForSemanticSegmentation` directory onto `device`.
///
/// When `target_classes` is `Some(classes)` and its length differs from the stored
/// `decode_head.classifier`, the encoder and fusion layers are loaded verbatim and the `1 × 1`
/// classifier head is deterministically initialized for `classes.len()`.
pub(super) fn load_dir(
    dir: &Path,
    target_classes: Option<&[&str]>,
    device: &Device,
) -> Result<(Segformer, SegformerConfig, PreprocessorConfig), SegmentError> {
    let mut config = SegformerConfig::load(&dir.join("config.json"))?;
    let preprocessor = PreprocessorConfig::load_or_default(&dir.join("preprocessor_config.json"))?;
    let stored_labels = config.num_labels();
    if let Some(classes) = target_classes {
        config.set_classes(classes);
    }
    let desired_labels = config.num_labels();

    let weight_path = dir.join("model.safetensors");
    let raw = fs::read(&weight_path).map_err(|source| SegmentError::Io {
        path: weight_path,
        source,
    })?;
    let archive = Archive::open(&raw)?;

    let mut stages = Vec::with_capacity(4);
    for s in 0..4 {
        let in_ch = if s == 0 {
            config.num_channels
        } else {
            config.hidden_sizes[s - 1]
        };
        let dim = config.hidden_sizes[s];
        let k = config.patch_sizes[s];
        let st = config.strides[s];
        let pe_prefix = format!("segformer.encoder.patch_embeddings.{s}");
        let patch_embed = OverlapPatchEmbed {
            proj: archive.conv2d(
                &format!("{pe_prefix}.proj"),
                in_ch,
                dim,
                k,
                st,
                k / 2,
                1,
                true,
                device,
            )?,
            layer_norm: archive.layer_norm(
                &format!("{pe_prefix}.layer_norm"),
                dim,
                config.layer_norm_eps,
                device,
            )?,
        };
        let mut blocks = Vec::with_capacity(config.depths[s]);
        for b in 0..config.depths[s] {
            let bp = format!("segformer.encoder.block.{s}.{b}");
            let sr = config.sr_ratios[s];
            let reduction = if sr > 1 {
                Some(SequenceReduction {
                    sr: archive.conv2d(
                        &format!("{bp}.attention.self.sr"),
                        dim,
                        dim,
                        sr,
                        sr,
                        0,
                        1,
                        true,
                        device,
                    )?,
                    layer_norm: archive.layer_norm(
                        &format!("{bp}.attention.self.layer_norm"),
                        dim,
                        config.layer_norm_eps,
                        device,
                    )?,
                })
            } else {
                None
            };
            let heads = config.num_attention_heads[s];
            let attention = EfficientSelfAttention {
                query: archive.linear(&format!("{bp}.attention.self.query"), dim, dim, device)?,
                key: archive.linear(&format!("{bp}.attention.self.key"), dim, dim, device)?,
                value: archive.linear(&format!("{bp}.attention.self.value"), dim, dim, device)?,
                dense: archive.linear(&format!("{bp}.attention.output.dense"), dim, dim, device)?,
                reduction,
                num_heads: heads,
                head_dim: dim / heads,
            };
            let exp = dim * config.mlp_ratios[s];
            let mlp = MixFfn {
                dense1: archive.linear(&format!("{bp}.mlp.dense1"), dim, exp, device)?,
                dwconv: archive.conv2d(
                    &format!("{bp}.mlp.dwconv.dwconv"),
                    exp,
                    exp,
                    3,
                    1,
                    1,
                    exp,
                    true,
                    device,
                )?,
                dense2: archive.linear(&format!("{bp}.mlp.dense2"), exp, dim, device)?,
            };
            blocks.push(SegformerLayer {
                layer_norm_1: archive.layer_norm(
                    &format!("{bp}.layer_norm_1"),
                    dim,
                    config.layer_norm_eps,
                    device,
                )?,
                attention,
                layer_norm_2: archive.layer_norm(
                    &format!("{bp}.layer_norm_2"),
                    dim,
                    config.layer_norm_eps,
                    device,
                )?,
                mlp,
            });
        }
        let layer_norm = archive.layer_norm(
            &format!("segformer.encoder.layer_norm.{s}"),
            dim,
            config.layer_norm_eps,
            device,
        )?;
        stages.push(SegformerStage {
            patch_embed,
            blocks,
            layer_norm,
        });
    }

    let dec = config.decoder_hidden_size;
    let mut linear_c = Vec::with_capacity(4);
    for s in 0..4 {
        linear_c.push(archive.linear(
            &format!("decode_head.linear_c.{s}.proj"),
            config.hidden_sizes[s],
            dec,
            device,
        )?);
    }
    let linear_fuse = archive.conv2d(
        "decode_head.linear_fuse",
        dec * 4,
        dec,
        1,
        1,
        0,
        1,
        false,
        device,
    )?;
    let bn_w = archive.f32_vec("decode_head.batch_norm.weight", &[dec])?;
    let bn_b = archive.f32_vec("decode_head.batch_norm.bias", &[dec])?;
    let bn_m = archive.f32_vec("decode_head.batch_norm.running_mean", &[dec])?;
    let bn_v = archive.f32_vec("decode_head.batch_norm.running_var", &[dec])?;
    let batch_norm = BatchNorm {
        gamma: Param::from_data(TensorData::new(bn_w, [dec]), device),
        beta: Param::from_data(TensorData::new(bn_b, [dec]), device),
        training: Param::from_bool(true),
        running_mean: RunningState::new(Tensor::from_data(TensorData::new(bn_m, [dec]), device)),
        running_var: RunningState::new(Tensor::from_data(TensorData::new(bn_v, [dec]), device)),
        momentum: 0.1,
        epsilon: 1e-5,
    };

    let classifier = if desired_labels == stored_labels {
        archive.conv2d(
            "decode_head.classifier",
            dec,
            desired_labels,
            1,
            1,
            0,
            1,
            true,
            device,
        )?
    } else {
        deterministic_classifier(dec, desired_labels, device)
    };

    let model = Segformer {
        encoder: SegformerEncoder { stages },
        decode_head: SegformerDecodeHead {
            linear_c,
            linear_fuse,
            batch_norm,
            dropout: DropoutConfig::new(config.classifier_dropout_prob).init(),
            classifier,
        },
        num_labels: desired_labels,
    };
    Ok((model, config, preprocessor))
}

fn deterministic_classifier(dec: usize, num_labels: usize, device: &Device) -> Conv2d {
    let mut w = Vec::with_capacity(num_labels * dec);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    for _ in 0..num_labels * dec {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let unit = ((state >> 32) as f32 / (u32::MAX as f32)) * 2.0 - 1.0;
        w.push(unit * 0.02);
    }
    Conv2d {
        weight: Param::from_data(TensorData::new(w, [num_labels, dec, 1, 1]), device),
        bias: Some(Param::from_data(
            TensorData::new(vec![0.0_f32; num_labels], [num_labels]),
            device,
        )),
        stride: [1, 1],
        kernel_size: [1, 1],
        dilation: [1, 1],
        groups: 1,
        padding: PaddingConfig2d::Valid,
    }
}

#[allow(dead_code)]
fn _keep_flag_import(_: Flag) {}
