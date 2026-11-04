//! Hardware-portable sparse feature extractor and matcher runtimes (`Burn` and `ONNX Runtime`).

use crate::error::FlightVisualError;
use burn::tensor::Device;
use image::GrayImage;
#[cfg(any(feature = "wgpu", feature = "metal", feature = "cuda"))]
use navigate_match::FastMatchConfig;
use navigate_match::{
    LighterGlue, ModelSessionExtractor, ModelSessionMatcher, OrtProvider, OrtSession,
    SparseFeatureExtractor, SparseFeatures, SparseMatcher, Weights, XFeat, XFeatPrecision,
    enable_persistent_gpu_cache, init_ort_runtime,
};
use sha2::{Digest, Sha256};
#[cfg(any(feature = "wgpu", feature = "metal", feature = "cuda"))]
use std::time::Instant;
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
};

/// Return the default `Burn` compute device with persistent GPU pipeline caching enabled.
#[must_use]
pub fn default_burn_device() -> (Device, &'static str) {
    enable_persistent_gpu_cache();
    #[cfg(feature = "cuda")]
    let out = (Device::cuda(0), "burn-cuda");
    #[cfg(all(not(feature = "cuda"), feature = "metal"))]
    let out = (Device::default(), "burn-metal");
    #[cfg(all(not(feature = "cuda"), not(feature = "metal"), feature = "wgpu"))]
    let out = (Device::wgpu(0), "burn-wgpu");
    #[cfg(not(any(feature = "cuda", feature = "metal", feature = "wgpu")))]
    let out = (Device::flex(), "burn-flex");
    out
}

/// Execution backend for dense `XFeat` feature extraction or `LighterGlue` matching.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum XFeatRuntime {
    /// Pure-Rust `Burn` `f32` backbone and device keypoint scorer.
    #[default]
    BurnF32,
    /// Pure-Rust `Burn` `f16` backbone with `f32` heads and L2 normalization.
    BurnF16,
    /// `ONNX Runtime` with the Apple `CoreML` Neural Engine execution provider.
    CoreMl,
    /// `ONNX Runtime` with the NVIDIA `TensorRT` and `CUDA` execution providers.
    TensorRt,
    /// `ONNX Runtime` CPU execution provider.
    OnnxCpu,
}

impl XFeatRuntime {
    /// Parse a runtime identifier (`burn-f32`, `burn-f16`, `coreml`, `tensorrt`, `onnx-cpu`).
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError::Invalid`] when `raw` does not name a supported runtime.
    pub fn parse(raw: &str) -> Result<Self, FlightVisualError> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "burn-f32" | "f32" => Ok(Self::BurnF32),
            "burn-f16" | "f16" => Ok(Self::BurnF16),
            "coreml" | "coreml-ane" => Ok(Self::CoreMl),
            "tensorrt" | "trt" => Ok(Self::TensorRt),
            "onnx-cpu" | "ort-cpu" => Ok(Self::OnnxCpu),
            other => Err(FlightVisualError::Invalid(format!(
                "unsupported --xfeat-runtime `{other}`; use burn-f32, burn-f16, coreml, tensorrt, or onnx-cpu"
            ))),
        }
    }

    /// Return the canonical telemetry name of this runtime.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
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

/// Runtime selection and optional `ONNX` paths for sparse `XFeat` extraction.
#[derive(Clone, Debug, Default)]
pub struct XFeatRuntimeConfig {
    /// Selected sparse extractor runtime.
    pub runtime: XFeatRuntime,
    /// Optional `.onnx` graph path when `runtime` uses `ONNX Runtime`.
    pub onnx_model_path: Option<PathBuf>,
    /// Optional `libonnxruntime` dynamic library path (falls back to `ORT_DYLIB_PATH`).
    pub onnx_runtime_path: Option<PathBuf>,
}

impl XFeatRuntimeConfig {
    /// Select the `Burn` `f16` backbone runtime.
    #[must_use]
    pub fn burn_f16() -> Self {
        Self {
            runtime: XFeatRuntime::BurnF16,
            onnx_model_path: None,
            onnx_runtime_path: None,
        }
    }
}

/// Shared handle to a loaded [`SparseFeatureExtractor`] and its provenance digests.
#[derive(Clone)]
pub struct LoadedXFeat {
    extractor: Arc<dyn SparseFeatureExtractor>,
    device: Device,
    weights_sha256: [u8; 32],
}

impl LoadedXFeat {
    /// Wrap an in-memory [`SparseFeatureExtractor`] without reading disk files.
    #[must_use]
    pub fn from_extractor(
        extractor: Arc<dyn SparseFeatureExtractor>,
        device: Device,
        weights_sha256: [u8; 32],
    ) -> Self {
        Self {
            extractor,
            device,
            weights_sha256,
        }
    }

    /// Load `XFeat` weights from `safetensors_path` on the configured runtime.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] when reading weights or initializing the backend fails.
    pub fn load(
        safetensors_path: &Path,
        cfg: &XFeatRuntimeConfig,
    ) -> Result<Self, FlightVisualError> {
        let bytes = fs::read(safetensors_path).map_err(|source| FlightVisualError::Io {
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

    /// Borrow the underlying `Burn` device.
    #[must_use]
    pub const fn device(&self) -> &Device {
        &self.device
    }

    /// Return the SHA-256 digest of the `.safetensors` file.
    #[must_use]
    pub const fn weights_sha256(&self) -> &[u8; 32] {
        &self.weights_sha256
    }

    /// Return the canonical encoder contract digest.
    #[must_use]
    pub fn encoder_contract(&self) -> [u8; 32] {
        self.extractor.encoder_contract()
    }

    /// Return the active runtime identifier.
    #[must_use]
    pub fn runtime_name(&self) -> &str {
        self.extractor.runtime_name()
    }

    /// Return true when `stamp` matches the encoder contract or weights digest.
    #[must_use]
    pub fn accepts_encoder_contract(&self, stamp: &[u8; 32]) -> bool {
        *stamp == self.encoder_contract() || stamp == &self.weights_sha256
    }

    /// Extract up to `limit` features from an unmasked grayscale image.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] when extraction fails.
    pub fn extract(
        &self,
        image: &GrayImage,
        limit: usize,
    ) -> Result<SparseFeatures, FlightVisualError> {
        Ok(self.extractor.extract_masked_blocking(image, None, limit)?)
    }

    /// Extract up to `limit` features from `image` masked by `valid_mask`.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] when extraction fails.
    pub fn extract_masked(
        &self,
        image: &GrayImage,
        valid_mask: Option<&GrayImage>,
        limit: usize,
    ) -> Result<SparseFeatures, FlightVisualError> {
        Ok(self
            .extractor
            .extract_masked_blocking(image, valid_mask, limit)?)
    }

    /// Extract up to `limit` features per image across a uniform-resolution batch.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] when extraction fails.
    pub fn extract_batch_masked(
        &self,
        items: &[(&GrayImage, Option<&GrayImage>)],
        limit: usize,
    ) -> Result<Vec<SparseFeatures>, FlightVisualError> {
        Ok(self.extractor.extract_batch_masked_blocking(items, limit)?)
    }

    /// Compile the `XFeat` shader pipeline with one masked extraction pass.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] when extraction fails.
    pub fn warmup_xfeat_only(
        &self,
        render_size: [u32; 2],
        limit: usize,
    ) -> Result<f64, FlightVisualError> {
        #[cfg(any(feature = "wgpu", feature = "metal", feature = "cuda"))]
        {
            let t0 = Instant::now();
            let (w, h, cap) = if cfg!(debug_assertions) {
                (32_u32, 32_u32, 32_usize)
            } else {
                (
                    render_size[0].max(32),
                    render_size[1].max(32),
                    limit.clamp(32, 2048),
                )
            };
            let patch = synthetic_noise_patch(w, h);
            let mask = GrayImage::from_pixel(w, h, image::Luma([255]));
            let _ = self.extract_masked(&patch, Some(&mask), cap)?;
            Ok(t0.elapsed().as_secs_f64() * 1000.0)
        }
        #[cfg(not(any(feature = "wgpu", feature = "metal", feature = "cuda")))]
        {
            let _ = (render_size, limit);
            Ok(0.0)
        }
    }

    /// Compile `XFeat` and the two primary static `LighterGlue` buckets (`B=1, 512x512` and `B=4, 512x1024`).
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] when device execution fails.
    pub fn warmup(&self, matcher: Option<&dyn SparseMatcher>) -> Result<f64, FlightVisualError> {
        #[cfg(any(feature = "wgpu", feature = "metal", feature = "cuda"))]
        {
            let t0 = Instant::now();
            let (w, h, cap) = if cfg!(debug_assertions) {
                (32_u32, 32_u32, 32_usize)
            } else {
                (800_u32, 600_u32, 1024_usize)
            };
            let patch = synthetic_noise_patch(w, h);
            let mask = GrayImage::from_pixel(w, h, image::Luma([255]));
            let feat = self.extract_masked(&patch, Some(&mask), cap)?;
            if let Some(m) = matcher
                && !feat.is_empty()
            {
                let f512 = slice_features(&feat, 480);
                let f1024 = slice_features(&feat, 960);
                let cfg = FastMatchConfig {
                    max_keypoints: cap.min(1024),
                    min_layers: 6,
                    max_layers: 6,
                    exit_check_layers: 0,
                    prune_after_layer: None,
                    ..FastMatchConfig::default()
                };
                let _ = m.match_batch_indexed_blocking(&[&f512], &f512, &cfg)?;
                let b4 = [&f512, &f512, &f512, &f512];
                let _ = m.match_batch_indexed_blocking(&b4, &f1024, &cfg)?;
            }
            Ok(t0.elapsed().as_secs_f64() * 1000.0)
        }
        #[cfg(not(any(feature = "wgpu", feature = "metal", feature = "cuda")))]
        {
            let _ = matcher;
            Ok(0.0)
        }
    }
}

/// Runtime configuration for loading a [`SparseMatcher`] (`Burn` `.safetensors` or `ORT` `.onnx`).
#[derive(Clone, Debug, Default)]
pub struct MatcherRuntimeConfig {
    /// Selected execution runtime (`BurnF32`/`BurnF16` uses `Burn`; `CoreMl`/`TensorRt`/`OnnxCpu` uses `ORT` when `.onnx` is supplied).
    pub runtime: XFeatRuntime,
    /// Optional `.onnx` graph path for `ModelSessionMatcher`.
    pub onnx_model_path: Option<PathBuf>,
    /// Optional `libonnxruntime` dynamic library path.
    pub onnx_runtime_path: Option<PathBuf>,
}

/// Shared handle to a hardware-portable [`SparseMatcher`].
#[derive(Clone)]
pub struct LoadedMatcher {
    matcher: Arc<dyn SparseMatcher>,
}

impl LoadedMatcher {
    /// Wrap an already-constructed [`SparseMatcher`] implementation.
    #[must_use]
    pub fn from_arc(matcher: Arc<dyn SparseMatcher>) -> Self {
        Self { matcher }
    }

    /// Load `LighterGlue` from `.safetensors` on `Burn` or from `.onnx` via `ModelSessionMatcher`.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] when weights or `ORT` session initialization fails.
    pub fn load(
        weights_or_onnx_path: &Path,
        device: &Device,
        cfg: &MatcherRuntimeConfig,
    ) -> Result<Self, FlightVisualError> {
        let is_onnx_ext = weights_or_onnx_path
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("onnx"));
        if let Some(provider) = cfg.runtime.ort_provider()
            && (is_onnx_ext || cfg.onnx_model_path.is_some())
        {
            let onnx_path = cfg
                .onnx_model_path
                .as_deref()
                .unwrap_or(weights_or_onnx_path);
            init_ort_runtime(cfg.onnx_runtime_path.as_deref())?;
            let session = OrtSession::load(onnx_path, provider)?;
            let matcher = ModelSessionMatcher::from_ort(session, cfg.runtime.as_str());
            return Ok(Self {
                matcher: Arc::new(matcher),
            });
        }
        if is_onnx_ext {
            init_ort_runtime(cfg.onnx_runtime_path.as_deref())?;
            let session = OrtSession::load(weights_or_onnx_path, OrtProvider::Cpu)?;
            let matcher = ModelSessionMatcher::from_ort(session, "onnx-cpu");
            return Ok(Self {
                matcher: Arc::new(matcher),
            });
        }
        let bytes = fs::read(weights_or_onnx_path).map_err(|source| FlightVisualError::Io {
            path: weights_or_onnx_path.to_path_buf(),
            source,
        })?;
        let weights = Weights::from_safetensors(&bytes)?;
        let lg = LighterGlue::load(weights, device)?;
        Ok(Self {
            matcher: Arc::new(lg),
        })
    }

    /// Borrow the underlying [`SparseMatcher`] trait object.
    #[must_use]
    pub fn as_matcher(&self) -> &dyn SparseMatcher {
        &*self.matcher
    }
}

#[cfg(any(feature = "wgpu", feature = "metal", feature = "cuda"))]
fn synthetic_noise_patch(w: u32, h: u32) -> GrayImage {
    let mut img = GrayImage::new(w, h);
    for (x, y, p) in img.enumerate_pixels_mut() {
        p.0[0] = ((x.wrapping_mul(37) ^ y.wrapping_mul(73))
            .wrapping_add(x.wrapping_mul(y).wrapping_mul(11))
            & 0xff) as u8;
    }
    img
}

#[cfg(any(feature = "wgpu", feature = "metal", feature = "cuda"))]
fn slice_features(src: &SparseFeatures, keep: usize) -> SparseFeatures {
    let n = keep.max(1).min(src.len());
    SparseFeatures {
        pixels: src.pixels[..n].to_vec(),
        model_pixels: src.model_pixels[..n].to_vec(),
        scores: src.scores[..n].to_vec(),
        descriptors: src.descriptors[..n * 64].to_vec(),
    }
}
