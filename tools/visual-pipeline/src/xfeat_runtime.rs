//! Configurable sparse `XFeat` feature extractor runtimes (`burn-f32`, `burn-f16`, `coreml`, `tensorrt`, `onnx-cpu`).

use crate::{PipelineError, matcher_client::default_burn_device};
use burn::tensor::Device;
use image::GrayImage;
use navigate_match::{
    LighterGlue, ModelSessionExtractor, OrtProvider, OrtSession, SparseFeatureExtractor,
    SparseFeatures, Weights, XFeat, XFeatPrecision, init_ort_runtime,
};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Execution backend for dense `XFeat` feature extraction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum XFeatRuntime {
    /// Pure-Rust `burn` `f32` backbone and device keypoint scorer.
    #[default]
    BurnF32,
    /// Pure-Rust `burn` `f16` backbone with `f32` heads and L2 normalization.
    BurnF16,
    /// ONNX Runtime with the Apple Core ML Neural Engine execution provider.
    CoreMl,
    /// ONNX Runtime with the NVIDIA TensorRT and CUDA execution providers.
    TensorRt,
    /// ONNX Runtime CPU execution provider.
    OnnxCpu,
}

impl XFeatRuntime {
    /// Parse a CLI `--xfeat-runtime` identifier.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Invalid`] when `raw` does not name a supported runtime.
    pub fn parse(raw: &str) -> Result<Self, PipelineError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "burn-f32" | "f32" => Ok(Self::BurnF32),
            "burn-f16" | "f16" => Ok(Self::BurnF16),
            "coreml" | "coreml-ane" => Ok(Self::CoreMl),
            "tensorrt" | "trt" => Ok(Self::TensorRt),
            "onnx-cpu" | "ort-cpu" => Ok(Self::OnnxCpu),
            other => Err(PipelineError::Invalid(format!(
                "unsupported --xfeat-runtime `{other}`; use burn-f32, burn-f16, coreml, tensorrt, or onnx-cpu"
            ))),
        }
    }

    /// Return the canonical telemetry name of this runtime.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::BurnF32 => "burn-f32",
            Self::BurnF16 => "burn-f16",
            Self::CoreMl => "coreml",
            Self::TensorRt => "tensorrt",
            Self::OnnxCpu => "onnx-cpu",
        }
    }

    fn ort_provider(self) -> Option<OrtProvider> {
        match self {
            Self::BurnF32 | Self::BurnF16 => None,
            Self::CoreMl => Some(OrtProvider::CoreMl),
            Self::TensorRt => Some(OrtProvider::TensorRt),
            Self::OnnxCpu => Some(OrtProvider::Cpu),
        }
    }
}

/// Runtime selection and optional ONNX paths for sparse `XFeat` extraction.
#[derive(Clone, Debug, Default)]
pub struct XFeatRuntimeConfig {
    /// Selected sparse extractor runtime.
    pub runtime: XFeatRuntime,
    /// Optional `.onnx` graph path when `runtime` uses ONNX Runtime.
    pub onnx_model_path: Option<PathBuf>,
    /// Optional `libonnxruntime` dynamic library path (falls back to `ORT_DYLIB_PATH`).
    pub onnx_runtime_path: Option<PathBuf>,
}

impl XFeatRuntimeConfig {
    /// Select the `burn` `f16` backbone runtime.
    #[must_use]
    pub fn burn_f16() -> Self {
        Self {
            runtime: XFeatRuntime::BurnF16,
            onnx_model_path: None,
            onnx_runtime_path: None,
        }
    }
}

/// Shared handle to a loaded [`SparseFeatureExtractor`] and its provenance stamps.
#[derive(Clone)]
pub(crate) struct LoadedXFeat {
    extractor: Arc<dyn SparseFeatureExtractor>,
    device: Device,
    weights_sha256: [u8; 32],
}

impl LoadedXFeat {
    pub(crate) fn load(
        safetensors_path: &Path,
        cfg: &XFeatRuntimeConfig,
    ) -> Result<Self, PipelineError> {
        let bytes = fs::read(safetensors_path).map_err(|source| PipelineError::Io {
            path: safetensors_path.to_path_buf(),
            source,
        })?;
        let weights_sha256: [u8; 32] = Sha256::digest(&bytes).into();
        let weights = Weights::from_safetensors(&bytes)?;
        let (device, _) = default_burn_device();

        let extractor: Arc<dyn SparseFeatureExtractor> = match cfg.runtime {
            XFeatRuntime::BurnF32 => Arc::new(XFeat::load_with_precision(
                weights,
                &device,
                XFeatPrecision::F32,
            )?),
            XFeatRuntime::BurnF16 => Arc::new(XFeat::load_with_precision(
                weights,
                &device,
                XFeatPrecision::F16,
            )?),
            runtime => {
                let provider = runtime.ort_provider().unwrap_or(OrtProvider::Cpu);
                let contract = XFeat::encoder_contract_of(&weights);
                let onnx_path = cfg
                    .onnx_model_path
                    .clone()
                    .unwrap_or_else(|| safetensors_path.with_extension("onnx"));
                init_ort_runtime(cfg.onnx_runtime_path.as_deref())?;
                let session = OrtSession::load(&onnx_path, provider)?;
                let input_name = session.primary_input_name().to_owned();
                Arc::new(ModelSessionExtractor::new(
                    session,
                    input_name,
                    contract,
                    runtime.as_str(),
                ))
            }
        };
        Ok(Self {
            extractor,
            device,
            weights_sha256,
        })
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn weights_sha256(&self) -> &[u8; 32] {
        &self.weights_sha256
    }

    pub(crate) fn encoder_contract(&self) -> [u8; 32] {
        self.extractor.encoder_contract()
    }

    pub(crate) fn runtime_name(&self) -> &str {
        self.extractor.runtime_name()
    }

    pub(crate) fn accepts_encoder_contract(&self, stamp: &[u8; 32]) -> bool {
        *stamp == self.encoder_contract() || stamp == &self.weights_sha256
    }

    pub(crate) fn extract(
        &self,
        image: &GrayImage,
        limit: usize,
    ) -> Result<SparseFeatures, PipelineError> {
        Ok(self.extractor.extract_masked_blocking(image, None, limit)?)
    }

    pub(crate) fn extract_masked(
        &self,
        image: &GrayImage,
        valid_mask: Option<&GrayImage>,
        limit: usize,
    ) -> Result<SparseFeatures, PipelineError> {
        Ok(self
            .extractor
            .extract_masked_blocking(image, valid_mask, limit)?)
    }

    pub(crate) fn extract_batch_masked(
        &self,
        items: &[(&GrayImage, Option<&GrayImage>)],
        limit: usize,
    ) -> Result<Vec<SparseFeatures>, PipelineError> {
        Ok(self.extractor.extract_batch_masked_blocking(items, limit)?)
    }

    pub(crate) fn warmup(&self, lighterglue: Option<&LighterGlue>) -> Result<f64, PipelineError> {
        #[cfg(any(feature = "wgpu", feature = "metal", feature = "cuda"))]
        {
            let t0 = std::time::Instant::now();
            let (w, h, h_wide, cap) = if cfg!(debug_assertions) {
                (32_u32, 32_u32, 24_u32, 32_usize)
            } else {
                (800_u32, 600_u32, 450_u32, 2048_usize)
            };
            let fill_noise = |img: &mut GrayImage| {
                for (x, y, p) in img.enumerate_pixels_mut() {
                    p.0[0] = ((x.wrapping_mul(37) ^ y.wrapping_mul(73))
                        .wrapping_add(x.wrapping_mul(y).wrapping_mul(11))
                        & 0xff) as u8;
                }
            };
            let mut patch = GrayImage::new(w, h);
            fill_noise(&mut patch);
            let mut patch_wide = GrayImage::new(w, h_wide);
            fill_noise(&mut patch_wide);
            let mask_std = GrayImage::from_pixel(w, h, image::Luma([255]));
            let mask_wide = GrayImage::from_pixel(w, h_wide, image::Luma([255]));
            let feat = self.extract(&patch, cap)?;
            let _ = self.extract_masked(&patch, Some(&mask_std), cap)?;
            let _ = self.extract(&patch_wide, cap)?;
            let feat_wide = self.extract_masked(&patch_wide, Some(&mask_wide), cap)?;
            if let Some(lg) = lighterglue
                && !feat.is_empty()
                && !feat_wide.is_empty()
            {
                let slice_feat = |src: &SparseFeatures, keep: usize| {
                    let n = keep.max(1).min(src.len());
                    SparseFeatures {
                        pixels: src.pixels[..n].to_vec(),
                        model_pixels: src.model_pixels[..n].to_vec(),
                        scores: src.scores[..n].to_vec(),
                        descriptors: src.descriptors[..n * 64].to_vec(),
                    }
                };
                let f256 = slice_feat(&feat, 240);
                let f512 = slice_feat(&feat, 480);
                let f768 = slice_feat(&feat, 720);
                let f1024 = slice_feat(&feat, 960);
                let cfg_1024 = navigate_match::FastMatchConfig {
                    max_keypoints: cap.min(1024),
                    min_layers: 6,
                    max_layers: 6,
                    exit_check_layers: 0,
                    prune_after_layer: None,
                    ..navigate_match::FastMatchConfig::default()
                };
                let cfg_768 = navigate_match::FastMatchConfig {
                    max_keypoints: cap.min(768),
                    ..cfg_1024
                };
                for single in [&f256, &f512, &f768] {
                    let _ = pollster::block_on(lg.match_features_batch_indexed_fast(
                        &[single],
                        &feat_wide,
                        &cfg_768,
                    ))?;
                }
                for single in [&f512, &f768, &f1024] {
                    let _ = pollster::block_on(lg.match_features_batch_indexed_fast(
                        &[single],
                        &feat_wide,
                        &cfg_1024,
                    ))?;
                }
                for lead in [&f256, &f512, &f768] {
                    let b4 = [lead, &f256, lead, &f256];
                    let _ = pollster::block_on(
                        lg.match_features_batch_indexed_fast(&b4, &feat_wide, &cfg_768),
                    )?;
                }
                let b4_1024 = [&f1024, &f512, &f768, &f256];
                let _ = pollster::block_on(
                    lg.match_features_batch_indexed_fast(&b4_1024, &feat_wide, &cfg_1024),
                )?;
            }
            Ok(t0.elapsed().as_secs_f64() * 1000.0)
        }
        #[cfg(not(any(feature = "wgpu", feature = "metal", feature = "cuda")))]
        {
            let _ = lighterglue;
            Ok(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_parser_accepts_supported_names_and_rejects_unknown() {
        assert_eq!(
            XFeatRuntime::parse("burn-f32").ok(),
            Some(XFeatRuntime::BurnF32)
        );
        assert_eq!(
            XFeatRuntime::parse("burn-f16").ok(),
            Some(XFeatRuntime::BurnF16)
        );
        assert_eq!(
            XFeatRuntime::parse("coreml").ok(),
            Some(XFeatRuntime::CoreMl)
        );
        assert_eq!(
            XFeatRuntime::parse("tensorrt").ok(),
            Some(XFeatRuntime::TensorRt)
        );
        assert_eq!(
            XFeatRuntime::parse("onnx-cpu").ok(),
            Some(XFeatRuntime::OnnxCpu)
        );
        assert!(XFeatRuntime::parse("unknown-npu").is_err());
    }
}
