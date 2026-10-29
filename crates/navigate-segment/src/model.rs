//! SegFormer semantic segmentation model (`SegformerForSemanticSegmentation`) on `burn`.

pub mod decoder;
pub mod encoder;
mod save;
mod weights;

use crate::{
    SegmentError,
    config::{ColorMode, PreprocessorConfig, SegformerConfig},
};
use burn::{
    module::Module,
    tensor::{Device, Tensor, TensorData, activation},
};
use decoder::{SegformerDecodeHead, bilinear_upsample_align_false};
use encoder::SegformerEncoder;
use image::RgbImage;
use std::path::Path;

/// Full-resolution per-pixel prediction from [`Segformer::predict_image`].
#[derive(Clone, Debug)]
pub struct SegmentationPrediction {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Argmax class index at each pixel (`len == width * height`).
    pub labels: Vec<u8>,
    /// Maximum class probability at each pixel (`len == width * height`).
    pub confidence: Vec<f32>,
    /// Per-pixel probabilities for caller-requested class indices (`[requested.len(), height, width]`).
    pub selected_probs: Vec<Vec<f32>>,
}

/// SegFormer semantic segmentation network (`encoder` + `decode_head`).
#[derive(Module, Debug)]
pub struct Segformer {
    /// Hierarchical MiT encoder.
    pub encoder: SegformerEncoder,
    /// All-MLP segmentation decode head.
    pub decode_head: SegformerDecodeHead,
    /// Number of output segmentation classes.
    pub num_labels: usize,
}

impl Segformer {
    /// Initialize a fresh [`Segformer`] from `config` on `device`.
    pub fn new(config: &SegformerConfig, device: &Device) -> Self {
        Self {
            encoder: SegformerEncoder::new(config, device),
            decode_head: SegformerDecodeHead::new(config, device),
            num_labels: config.num_labels(),
        }
    }

    /// Load a HuggingFace `SegformerForSemanticSegmentation` directory onto `device`.
    ///
    /// Pass `Some(&CLASSES)` in `target_classes` to adapt an ADE20K or ImageNet checkpoint to the
    /// 4-class navigation head (`sky`, `ground`, `airframe`, `propeller`).
    ///
    /// # Errors
    ///
    /// Returns [`SegmentError`] when configuration or SafeTensors weights fail to load.
    pub fn load_pretrained(
        dir: &Path,
        target_classes: Option<&[&str]>,
        device: &Device,
    ) -> Result<(Self, SegformerConfig, PreprocessorConfig), SegmentError> {
        weights::load_dir(dir, target_classes, device)
    }

    /// Save `self`, `config`, and `preprocessor` in HuggingFace `from_pretrained` layout at `dir`.
    ///
    /// # Errors
    ///
    /// Returns [`SegmentError`] when serialization or file writes fail.
    pub fn save_pretrained(
        &self,
        dir: &Path,
        config: &SegformerConfig,
        preprocessor: &PreprocessorConfig,
    ) -> Result<(), SegmentError> {
        save::save_dir(self, config, preprocessor, dir)
    }

    /// Forward pass returning `[B, num_labels, H / 4, W / 4]` logits.
    pub fn forward(&self, pixel_values: Tensor<4>) -> Tensor<4> {
        let features = self.encoder.forward(pixel_values);
        self.decode_head.forward(features)
    }

    /// Forward pass upsampling logits to `(out_h, out_w)` with half-pixel bilinear interpolation.
    pub fn forward_upsampled(
        &self,
        pixel_values: Tensor<4>,
        out_h: usize,
        out_w: usize,
    ) -> Tensor<4> {
        bilinear_upsample_align_false(self.forward(pixel_values), out_h, out_w)
    }

    /// Predict full-resolution semantic labels, confidence, and optional class probability maps.
    ///
    /// # Errors
    ///
    /// Returns [`SegmentError`] when tensor readback fails.
    pub fn predict_image(
        &self,
        rgb: &RgbImage,
        preprocessor: &PreprocessorConfig,
        color_mode: ColorMode,
        requested_classes: &[usize],
        device: &Device,
    ) -> Result<SegmentationPrediction, SegmentError> {
        let orig_w = rgb.width() as usize;
        let orig_h = rgb.height() as usize;
        if orig_w == 0 || orig_h == 0 {
            return Err(SegmentError::Invalid("empty RGB image".into()));
        }
        let (buf, in_h, in_w) = preprocessor.preprocess(rgb, color_mode);
        let input = Tensor::<4>::from_data(TensorData::new(buf, [1, 3, in_h, in_w]), device);
        let logits_low = self.forward(input);
        let logits_full = bilinear_upsample_align_false(logits_low, orig_h, orig_w);
        let probs = activation::softmax(logits_full, 1);

        let selected_probs = requested_classes
            .iter()
            .map(|&cls| {
                if cls >= self.num_labels {
                    return Ok(vec![0.0_f32; orig_h * orig_w]);
                }
                probs
                    .clone()
                    .narrow(1, cls, 1)
                    .into_data()
                    .try_to_vec::<f32>()
                    .map_err(|e| SegmentError::Safetensors(format!("{e:?}")))
            })
            .collect::<Result<Vec<_>, SegmentError>>()?;

        let (max_probs, argmax) = probs.max_dim_with_indices(1);
        let confidence = max_probs
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| SegmentError::Safetensors(format!("{e:?}")))?;
        let raw_labels: Vec<i64> = argmax
            .into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .map_err(|e| SegmentError::Safetensors(format!("{e:?}")))?;
        let labels = raw_labels
            .into_iter()
            .map(|v| v.clamp(0, 254) as u8)
            .collect();

        Ok(SegmentationPrediction {
            width: orig_w,
            height: orig_h,
            labels,
            confidence,
            selected_probs,
        })
    }
}
