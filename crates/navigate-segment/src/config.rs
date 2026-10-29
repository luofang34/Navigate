//! SegFormer-B0 architecture configuration, image preprocessor settings, and class constants.

use crate::SegmentError;
use image::{RgbImage, imageops::FilterType};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};

/// Fine-tuned sky class index.
pub const SKY: u8 = 0;
/// Fine-tuned ground class index.
pub const GROUND: u8 = 1;
/// Fine-tuned airframe class index.
pub const AIRFRAME: u8 = 2;
/// Fine-tuned propeller class index.
pub const PROPELLER: u8 = 3;
/// Ignored target label index during cross-entropy loss evaluation.
pub const IGNORE_INDEX: u8 = 255;

/// Ordered 4-class fine-tuned label names.
pub const CLASSES: [&str; 4] = ["sky", "ground", "airframe", "propeller"];

/// Color handling mode for training and evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ColorMode {
    /// Keep RGB channels unchanged.
    Color,
    /// Convert RGB to replicated luminance with 30% probability during training.
    RandGray,
    /// Always convert RGB to replicated luminance (`0.299 R + 0.587 G + 0.114 B`).
    Gray,
}

impl ColorMode {
    /// Parse a CLI color mode string.
    ///
    /// # Errors
    ///
    /// Returns [`SegmentError::Invalid`] when `value` is not recognized.
    pub fn parse(value: &str) -> Result<Self, SegmentError> {
        match value {
            "color" => Ok(Self::Color),
            "randgray" => Ok(Self::RandGray),
            "gray" => Ok(Self::Gray),
            other => Err(SegmentError::Invalid(format!(
                "unknown color mode '{other}'; expected color, randgray, or gray"
            ))),
        }
    }

    /// Return the canonical lowercase identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Color => "color",
            Self::RandGray => "randgray",
            Self::Gray => "gray",
        }
    }
}

/// HuggingFace `SegformerConfig` (`config.json`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SegformerConfig {
    /// Model architecture names.
    #[serde(default = "default_architectures")]
    pub architectures: Vec<String>,
    /// Attention dropout probability.
    #[serde(default)]
    pub attention_probs_dropout_prob: f64,
    /// Decode-head dropout probability.
    #[serde(default = "default_classifier_dropout")]
    pub classifier_dropout_prob: f64,
    /// Shared channel width in the MLP decode head.
    #[serde(default = "default_decoder_hidden")]
    pub decoder_hidden_size: usize,
    /// Transformer block count per stage.
    pub depths: [usize; 4],
    /// Downsampling factors per stage.
    #[serde(default = "default_downsampling")]
    pub downsampling_rates: [usize; 4],
    /// Maximum stochastic depth rate.
    #[serde(default = "default_drop_path")]
    pub drop_path_rate: f64,
    /// Activation function identifier.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    /// Hidden dropout probability.
    #[serde(default)]
    pub hidden_dropout_prob: f64,
    /// Channel widths per encoder stage.
    pub hidden_sizes: [usize; 4],
    /// Mapping from stringified class index to class name.
    #[serde(default)]
    pub id2label: BTreeMap<String, String>,
    /// Mapping from class name to class index.
    #[serde(default)]
    pub label2id: BTreeMap<String, usize>,
    /// LayerNorm epsilon.
    #[serde(default = "default_ln_eps")]
    pub layer_norm_eps: f64,
    /// MLP expansion ratios per stage.
    #[serde(default = "default_mlp_ratios")]
    pub mlp_ratios: [usize; 4],
    /// HuggingFace model type tag.
    #[serde(default = "default_model_type")]
    pub model_type: String,
    /// Attention head counts per stage.
    pub num_attention_heads: [usize; 4],
    /// Input image channel count.
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    /// Number of encoder stages.
    #[serde(default = "default_num_blocks")]
    pub num_encoder_blocks: usize,
    /// Overlap patch kernel sizes per stage.
    pub patch_sizes: [usize; 4],
    /// Whether stage 3 reshapes tokens back to `[B, C, H, W]`.
    #[serde(default = "default_true")]
    pub reshape_last_stage: bool,
    /// Sequence reduction ratios per stage.
    pub sr_ratios: [usize; 4],
    /// Patch convolution strides per stage.
    pub strides: [usize; 4],
    /// Ignored label index in semantic loss.
    #[serde(default = "default_ignore_index")]
    pub semantic_loss_ignore_index: usize,
}

fn default_architectures() -> Vec<String> {
    vec!["SegformerForSemanticSegmentation".into()]
}
fn default_classifier_dropout() -> f64 {
    0.1
}
fn default_decoder_hidden() -> usize {
    256
}
fn default_downsampling() -> [usize; 4] {
    [1, 4, 8, 16]
}
fn default_drop_path() -> f64 {
    0.1
}
fn default_hidden_act() -> String {
    "gelu".into()
}
fn default_ln_eps() -> f64 {
    1e-6
}
fn default_mlp_ratios() -> [usize; 4] {
    [4, 4, 4, 4]
}
fn default_model_type() -> String {
    "segformer".into()
}
fn default_num_channels() -> usize {
    3
}
fn default_num_blocks() -> usize {
    4
}
fn default_true() -> bool {
    true
}
fn default_ignore_index() -> usize {
    usize::from(IGNORE_INDEX)
}

impl SegformerConfig {
    /// Construct a standard MiT-B0 configuration for `classes`.
    pub fn mit_b0(classes: &[&str]) -> Self {
        let mut id2label = BTreeMap::new();
        let mut label2id = BTreeMap::new();
        for (i, &name) in classes.iter().enumerate() {
            id2label.insert(i.to_string(), name.to_owned());
            label2id.insert(name.to_owned(), i);
        }
        Self {
            architectures: default_architectures(),
            attention_probs_dropout_prob: 0.0,
            classifier_dropout_prob: 0.1,
            decoder_hidden_size: 256,
            depths: [2, 2, 2, 2],
            downsampling_rates: [1, 4, 8, 16],
            drop_path_rate: 0.1,
            hidden_act: "gelu".into(),
            hidden_dropout_prob: 0.0,
            hidden_sizes: [32, 64, 160, 256],
            id2label,
            label2id,
            layer_norm_eps: 1e-6,
            mlp_ratios: [4, 4, 4, 4],
            model_type: "segformer".into(),
            num_attention_heads: [1, 2, 5, 8],
            num_channels: 3,
            num_encoder_blocks: 4,
            patch_sizes: [7, 3, 3, 3],
            reshape_last_stage: true,
            sr_ratios: [8, 4, 2, 1],
            strides: [4, 2, 2, 2],
            semantic_loss_ignore_index: usize::from(IGNORE_INDEX),
        }
    }

    /// Load `config.json` from `path`.
    ///
    /// # Errors
    ///
    /// Returns [`SegmentError`] when reading or JSON parsing fails.
    pub fn load(path: &Path) -> Result<Self, SegmentError> {
        let text = fs::read_to_string(path).map_err(|source| SegmentError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        serde_json::from_str(&text).map_err(|source| SegmentError::Json {
            context: path.display().to_string(),
            source,
        })
    }

    /// Number of output segmentation classes.
    pub fn num_labels(&self) -> usize {
        self.id2label.len().max(1)
    }

    /// Replace class label tables with `classes`.
    pub fn set_classes(&mut self, classes: &[&str]) {
        self.id2label.clear();
        self.label2id.clear();
        for (i, &name) in classes.iter().enumerate() {
            self.id2label.insert(i.to_string(), name.to_owned());
            self.label2id.insert(name.to_owned(), i);
        }
    }

    /// Look up the class index for `label_name`.
    pub fn class_index(&self, label_name: &str) -> Option<usize> {
        self.label2id.get(label_name).copied().or_else(|| {
            self.id2label
                .iter()
                .find_map(|(k, v)| (v == label_name).then(|| k.parse::<usize>().ok())?)
        })
    }
}

/// HuggingFace `preprocessor_config.json` for SegFormer image normalization.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreprocessorConfig {
    /// Normalize RGB values with `image_mean` and `image_std`.
    #[serde(default = "default_true")]
    pub do_normalize: bool,
    /// Resize input images to `(input_size, input_size)` before inference.
    #[serde(default = "default_true")]
    pub do_resize: bool,
    /// Per-channel RGB mean in `[0, 1]`.
    #[serde(default = "default_mean")]
    pub image_mean: [f32; 3],
    /// Per-channel RGB standard deviation in `[0, 1]`.
    #[serde(default = "default_std")]
    pub image_std: [f32; 3],
    /// Whether ADE20K zero-label reduction was used during pretraining.
    #[serde(default)]
    pub reduce_labels: bool,
    /// Square resize target (`size` may be an integer or `{"height": H, "width": W}`).
    #[serde(default = "default_size_value")]
    pub size: serde_json::Value,
}

fn default_mean() -> [f32; 3] {
    [0.485, 0.456, 0.406]
}
fn default_std() -> [f32; 3] {
    [0.229, 0.224, 0.225]
}
fn default_size_value() -> serde_json::Value {
    serde_json::json!(512)
}

impl Default for PreprocessorConfig {
    fn default() -> Self {
        Self {
            do_normalize: true,
            do_resize: true,
            image_mean: default_mean(),
            image_std: default_std(),
            reduce_labels: false,
            size: default_size_value(),
        }
    }
}

impl PreprocessorConfig {
    /// Load `preprocessor_config.json` from `path`, or return [`Self::default`] if missing.
    ///
    /// # Errors
    ///
    /// Returns [`SegmentError`] when an existing file cannot be read or parsed.
    pub fn load_or_default(path: &Path) -> Result<Self, SegmentError> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = fs::read_to_string(path).map_err(|source| SegmentError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        serde_json::from_str(&text).map_err(|source| SegmentError::Json {
            context: path.display().to_string(),
            source,
        })
    }

    /// Return `(width, height)` target size for model input.
    pub fn target_size(&self) -> (u32, u32) {
        if let Some(side) = self.size.as_u64() {
            let side = (side as u32).max(32);
            return (side, side);
        }
        if let Some(obj) = self.size.as_object() {
            let h = obj.get("height").and_then(|v| v.as_u64()).unwrap_or(512) as u32;
            let w = obj.get("width").and_then(|v| v.as_u64()).unwrap_or(512) as u32;
            return (w.max(32), h.max(32));
        }
        (512, 512)
    }

    /// Preprocess `rgb` into a normalized `[1, 3, H, W]` buffer.
    pub fn preprocess(&self, rgb: &RgbImage, mode: ColorMode) -> (Vec<f32>, usize, usize) {
        let (target_w, target_h) = if self.do_resize {
            self.target_size()
        } else {
            rgb.dimensions()
        };
        let resized = if rgb.dimensions() == (target_w, target_h) {
            rgb.clone()
        } else {
            image::imageops::resize(rgb, target_w, target_h, FilterType::Triangle)
        };
        let w = target_w as usize;
        let h = target_h as usize;
        let plane = w * h;
        let mut out = vec![0.0_f32; 3 * plane];
        for y in 0..h {
            for x in 0..w {
                let p = resized.get_pixel(x as u32, y as u32).0;
                let (r, g, b) = if mode == ColorMode::Gray {
                    let luma = (0.299 * f32::from(p[0])
                        + 0.587 * f32::from(p[1])
                        + 0.114 * f32::from(p[2]))
                        / 255.0;
                    (luma, luma, luma)
                } else {
                    (
                        f32::from(p[0]) / 255.0,
                        f32::from(p[1]) / 255.0,
                        f32::from(p[2]) / 255.0,
                    )
                };
                let idx = y * w + x;
                if self.do_normalize {
                    out[idx] = (r - self.image_mean[0]) / self.image_std[0];
                    out[plane + idx] = (g - self.image_mean[1]) / self.image_std[1];
                    out[2 * plane + idx] = (b - self.image_mean[2]) / self.image_std[2];
                } else {
                    out[idx] = r;
                    out[plane + idx] = g;
                    out[2 * plane + idx] = b;
                }
            }
        }
        (out, h, w)
    }
}
