//! Configurable sparse `XFeat` feature extractor runtimes (`burn-f32`, `burn-f16`, `coreml`, `tensorrt`, `onnx-cpu`).

use crate::{PipelineError, matcher_client::default_burn_device};
use burn::tensor::Device;
use image::GrayImage;
use navigate_match::{
    LighterGlue, ModelSessionExtractor, SparseFeatureExtractor, SparseFeatures, Weights, XFeat,
    XFeatPrecision,
};
use navigate_visual::{EngineKind, HostTensor, ModelInference, NamedTensor};
use ort::{
    ep,
    session::{Session, builder::AutoDevicePolicy},
    value::TensorRef,
};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::{Arc, OnceLock},
};

static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();

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
            XFeatRuntime::CoreMl | XFeatRuntime::TensorRt | XFeatRuntime::OnnxCpu => {
                let contract = XFeat::encoder_contract_of(&weights);
                let onnx_path = cfg
                    .onnx_model_path
                    .clone()
                    .unwrap_or_else(|| safetensors_path.with_extension("onnx"));
                init_ort_once(cfg.onnx_runtime_path.as_deref())?;
                let session = OrtDenseSession::load(&onnx_path, cfg.runtime)?;
                Arc::new(ModelSessionExtractor::new(
                    session,
                    "image",
                    contract,
                    cfg.runtime.as_str(),
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

    pub(crate) fn warmup(&self, lighterglue: Option<&LighterGlue>) -> Result<f64, PipelineError> {
        #[cfg(any(feature = "wgpu", feature = "metal", feature = "cuda"))]
        {
            let t0 = std::time::Instant::now();
            let (w, h, cap) = if cfg!(debug_assertions) {
                (32_u32, 32_u32, 32_usize)
            } else {
                (800_u32, 600_u32, 2048_usize)
            };
            let mut patch = GrayImage::new(w, h);
            for (x, y, p) in patch.enumerate_pixels_mut() {
                p.0[0] = ((x.wrapping_mul(37) ^ y.wrapping_mul(73))
                    .wrapping_add(x.wrapping_mul(y).wrapping_mul(11))
                    & 0xff) as u8;
            }
            let feat = self.extract(&patch, cap)?;
            if let Some(lg) = lighterglue
                && !feat.is_empty()
            {
                let cfg_full = navigate_match::FastMatchConfig {
                    max_keypoints: cap,
                    ..navigate_match::FastMatchConfig::balanced_search()
                };
                let _ = pollster::block_on(lg.match_features_fast(&feat, &feat, &cfg_full))?;
                let cfg_batch = navigate_match::FastMatchConfig {
                    max_keypoints: cap.min(1024),
                    ..navigate_match::FastMatchConfig::balanced_search()
                };
                let batch = [&feat, &feat, &feat, &feat];
                let _ = pollster::block_on(
                    lg.match_features_batch_indexed_fast(&batch, &feat, &cfg_batch),
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

fn init_ort_once(explicit_lib: Option<&Path>) -> Result<(), PipelineError> {
    let resolved = explicit_lib
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from))
        .ok_or_else(|| {
            PipelineError::Invalid(
                "ONNX XFeat runtime requires --onnx-runtime or ORT_DYLIB_PATH".into(),
            )
        })?;
    let outcome = ORT_INIT.get_or_init(|| {
        let builder = ort::init_from(&resolved)
            .map_err(|e| format!("load ONNX Runtime {}: {e}", resolved.display()))?;
        let _ = builder.with_telemetry(false).commit();
        Ok(())
    });
    outcome.clone().map_err(PipelineError::Invalid)
}

struct OrtDenseSession {
    session: Session,
    output_names: Vec<String>,
}

impl OrtDenseSession {
    fn load(onnx_path: &Path, runtime: XFeatRuntime) -> Result<Self, PipelineError> {
        if !onnx_path.is_file() {
            return Err(PipelineError::Invalid(format!(
                "missing XFeat ONNX model {} (pass --xfeat-onnx)",
                onnx_path.display()
            )));
        }
        let providers = match runtime {
            XFeatRuntime::CoreMl => vec![
                ep::CoreML::default()
                    .with_compute_units(ep::coreml::ComputeUnits::CPUAndNeuralEngine)
                    .with_static_input_shapes(true)
                    .with_model_format(ep::coreml::ModelFormat::MLProgram)
                    .build()
                    .error_on_failure(),
            ],
            XFeatRuntime::TensorRt => vec![
                ep::TensorRT::default()
                    .with_max_workspace_size(256 * 1024 * 1024)
                    .build()
                    .error_on_failure(),
                ep::CUDA::default().build().error_on_failure(),
            ],
            _ => Vec::new(),
        };
        let session = Session::builder()
            .map_err(|e| PipelineError::Invalid(format!("create ORT session: {e}")))?
            .with_auto_device(AutoDevicePolicy::Default)
            .map_err(|e| PipelineError::Invalid(format!("configure ORT auto device: {e}")))?
            .with_intra_threads(4)
            .map_err(|e| PipelineError::Invalid(format!("set ORT threads: {e}")))?
            .with_intra_op_spinning(false)
            .map_err(|e| PipelineError::Invalid(format!("set ORT spin policy: {e}")))?
            .with_execution_providers(providers)
            .map_err(|e| {
                PipelineError::Invalid(format!("register {} provider: {e}", runtime.as_str()))
            })?
            .commit_from_file(onnx_path)
            .map_err(|e| PipelineError::Invalid(format!("load {}: {e}", onnx_path.display())))?;
        let output_names = session
            .outputs()
            .iter()
            .map(|o| o.name().to_owned())
            .collect();
        Ok(Self {
            session,
            output_names,
        })
    }
}

impl ModelInference for OrtDenseSession {
    type Tensor = HostTensor;
    type Error = String;

    fn engine_kind(&self) -> EngineKind {
        EngineKind::Ort
    }

    fn forward(
        &mut self,
        inputs: &[NamedTensor<HostTensor>],
    ) -> Result<Vec<NamedTensor<HostTensor>>, String> {
        let mut ort_inputs = Vec::with_capacity(inputs.len());
        for named in inputs {
            let view = TensorRef::from_array_view((
                named.tensor.shape().to_vec(),
                named.tensor.as_slice(),
            ))
            .map_err(|e| format!("borrow ORT input {}: {e}", named.name))?
            .into_dyn();
            ort_inputs.push((named.name.clone(), view));
        }
        let outputs = self
            .session
            .run(ort_inputs)
            .map_err(|e| format!("run ORT XFeat session: {e}"))?;
        let mut results = Vec::with_capacity(self.output_names.len());
        for name in &self.output_names {
            let (shape, slice) = outputs
                .get(name.as_str())
                .ok_or_else(|| format!("missing ORT output {name}"))?
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("extract ORT output {name}: {e}"))?;
            let dims: Vec<usize> = shape
                .iter()
                .map(|&d| usize::try_from(d).map_err(|_| format!("negative dimension on {name}")))
                .collect::<Result<_, _>>()?;
            let tensor = HostTensor::new(dims, slice.to_vec())
                .map_err(|_| format!("non-finite ORT output {name}"))?;
            results.push(NamedTensor::new(name.clone(), tensor));
        }
        Ok(results)
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
