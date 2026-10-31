//! Optional ONNX Runtime session implementing [`ModelInference`].
//!
//! Host and embedded Linux targets enable the `ort` feature to run fixed-shape dense
//! convolutional graphs on Apple Core ML, NVIDIA TensorRT, CUDA, or CPU providers
//! while sharing [`crate::PreparedXFeatImage`] preprocessing and sparse keypoint decoding.
use crate::{
    MatchError,
    traits::{EngineKind, HostTensor, ModelInference, NamedTensor},
};
use ort::{
    ep,
    session::{Session, builder::AutoDevicePolicy},
    value::TensorRef,
};
use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
};

static ORT_INIT: OnceLock<Result<(), String>> = OnceLock::new();

/// Execution provider selection for [`OrtSession`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OrtProvider {
    /// ONNX Runtime CPU execution provider.
    #[default]
    Cpu,
    /// Apple Core ML CPU and Neural Engine execution provider.
    CoreMl,
    /// NVIDIA TensorRT execution provider with CUDA fallback.
    TensorRt,
}

impl OrtProvider {
    /// Return the stable runtime identifier of this provider.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "onnx-cpu",
            Self::CoreMl => "coreml",
            Self::TensorRt => "tensorrt",
        }
    }
}

/// Initialize the shared ONNX Runtime dynamic library once per process.
///
/// The loader uses `explicit_dylib` when supplied and falls back to `ORT_DYLIB_PATH`.
///
/// # Errors
///
/// Returns [`MatchError::Device`] when no dynamic library path is configured or loading fails.
pub fn init_ort_runtime(explicit_dylib: Option<&Path>) -> Result<(), MatchError> {
    let resolved = explicit_dylib
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os("ORT_DYLIB_PATH").map(PathBuf::from))
        .ok_or_else(|| MatchError::Device {
            stage: "ort init",
            message: "ONNX Runtime requires an explicit library path or ORT_DYLIB_PATH".into(),
        })?;
    let outcome = ORT_INIT.get_or_init(|| {
        let builder = ort::init_from(&resolved)
            .map_err(|e| format!("load ONNX Runtime {}: {e}", resolved.display()))?;
        let _ = builder.with_telemetry(false).commit();
        Ok(())
    });
    outcome.clone().map_err(|message| MatchError::Device {
        stage: "ort init",
        message,
    })
}

/// Fixed-shape ONNX Runtime graph session implementing [`ModelInference`].
pub struct OrtSession {
    session: Session,
    primary_input: String,
    output_names: Vec<String>,
}

impl OrtSession {
    /// Load `onnx_path` on `provider` after [`init_ort_runtime`] succeeds.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `onnx_path` does not exist, provider registration fails,
    /// or the graph declares no input tensors.
    pub fn load(onnx_path: &Path, provider: OrtProvider) -> Result<Self, MatchError> {
        if !onnx_path.is_file() {
            return Err(MatchError::Device {
                stage: "ort load",
                message: format!("missing ONNX model {}", onnx_path.display()),
            });
        }
        let providers = match provider {
            OrtProvider::CoreMl => vec![
                ep::CoreML::default()
                    .with_compute_units(ep::coreml::ComputeUnits::CPUAndNeuralEngine)
                    .with_static_input_shapes(true)
                    .with_model_format(ep::coreml::ModelFormat::MLProgram)
                    .build()
                    .error_on_failure(),
            ],
            OrtProvider::TensorRt => vec![
                ep::TensorRT::default()
                    .with_max_workspace_size(256 * 1024 * 1024)
                    .build()
                    .error_on_failure(),
                ep::CUDA::default().build().error_on_failure(),
            ],
            OrtProvider::Cpu => Vec::new(),
        };
        fn dev_err(stage: &'static str, err: impl std::fmt::Display) -> MatchError {
            MatchError::Device {
                stage,
                message: err.to_string(),
            }
        }
        let session = Session::builder()
            .map_err(|e| dev_err("ort session builder", e))?
            .with_auto_device(AutoDevicePolicy::Default)
            .map_err(|e| dev_err("ort auto device", e))?
            .with_intra_threads(4)
            .map_err(|e| dev_err("ort threads", e))?
            .with_intra_op_spinning(false)
            .map_err(|e| dev_err("ort spin policy", e))?
            .with_execution_providers(providers)
            .map_err(|e| MatchError::Device {
                stage: "ort provider",
                message: format!("{}: {e}", provider.as_str()),
            })?
            .commit_from_file(onnx_path)
            .map_err(|e| MatchError::Device {
                stage: "ort commit",
                message: format!("{}: {e}", onnx_path.display()),
            })?;
        let primary_input = session
            .inputs()
            .first()
            .map(|input| input.name().to_owned())
            .ok_or(MatchError::Output("ORT model declares no input tensors"))?;
        let output_names = session
            .outputs()
            .iter()
            .map(|output| output.name().to_owned())
            .collect();
        Ok(Self {
            session,
            primary_input,
            output_names,
        })
    }

    /// Return the primary input tensor name discovered from the ONNX graph.
    #[must_use]
    pub fn primary_input_name(&self) -> &str {
        &self.primary_input
    }

    /// Return the ordered output tensor names declared by the ONNX graph.
    #[must_use]
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }
}

impl ModelInference for OrtSession {
    type Tensor = HostTensor;
    type Error = MatchError;

    fn engine_kind(&self) -> EngineKind {
        EngineKind::Ort
    }

    fn forward(
        &mut self,
        inputs: &[NamedTensor<HostTensor>],
    ) -> Result<Vec<NamedTensor<HostTensor>>, MatchError> {
        let mut ort_inputs = Vec::with_capacity(inputs.len());
        for named in inputs {
            let view = TensorRef::from_array_view((
                named.tensor.shape().to_vec(),
                named.tensor.as_slice(),
            ))
            .map_err(|e| MatchError::Device {
                stage: "ort input",
                message: format!("{}: {e}", named.name),
            })?
            .into_dyn();
            ort_inputs.push((named.name.clone(), view));
        }
        let outputs = self
            .session
            .run(ort_inputs)
            .map_err(|e| MatchError::Device {
                stage: "ort forward",
                message: e.to_string(),
            })?;
        let mut results = Vec::with_capacity(self.output_names.len());
        for name in &self.output_names {
            let (shape, slice) = outputs
                .get(name.as_str())
                .ok_or(MatchError::Output("missing declared ORT output tensor"))?
                .try_extract_tensor::<f32>()
                .map_err(|e| MatchError::Device {
                    stage: "ort output",
                    message: format!("{name}: {e}"),
                })?;
            let dims: Vec<usize> = shape
                .iter()
                .map(|&d| usize::try_from(d).map_err(|_| MatchError::Output("negative ORT dim")))
                .collect::<Result<_, _>>()?;
            let tensor = HostTensor::new(dims, slice.to_vec())
                .map_err(|_| MatchError::Output("non-finite ORT output tensor"))?;
            results.push(NamedTensor::new(name.clone(), tensor));
        }
        Ok(results)
    }
}
