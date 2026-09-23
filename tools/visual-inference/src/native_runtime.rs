//! Explicit native runtime setup and contextual adapter errors.
use ort::{ep, session::Session};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// A native adapter could not produce correspondences.
#[derive(Debug, Error)]
pub enum InferenceError {
    /// The host runtime library could not load.
    #[error("load runtime {path}: {source}")]
    Load {
        /// Runtime library path.
        path: PathBuf,
        /// Dynamic loader failure.
        #[source]
        source: ort::LoadDynamicError,
    },
    /// A model or runtime file could not be read.
    #[error("read {path}: {source}")]
    Io {
        /// Input path.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// A device cache directory could not be created.
    #[error("create device cache {path}: {source}")]
    Cache {
        /// Requested cache directory.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// ONNX Runtime rejected an operation.
    #[error("{operation}: {source}")]
    Runtime {
        /// Operation and model context.
        operation: String,
        /// Runtime error.
        #[source]
        source: ort::Error,
    },
    /// The supplied data does not satisfy this export's contract.
    #[error("invalid inference input or output: {0}")]
    Invalid(String),
}

pub(crate) fn runtime(
    operation: impl Into<String>,
    source: impl Into<ort::Error>,
) -> InferenceError {
    InferenceError::Runtime {
        operation: operation.into(),
        source: source.into(),
    }
}

/// Requested native execution provider. Unsupported operators can use CPU.
#[derive(Clone, Copy, Debug)]
pub enum Provider {
    /// CPU execution.
    Cpu,
    /// Core ML with CPU and GPU compute units.
    CoreMlGpu,
    /// Core ML with CPU and Neural Engine compute units.
    CoreMlAne,
    /// NVIDIA CUDA. Provider registration must succeed.
    Cuda,
    /// NVIDIA TensorRT, then CUDA for unsupported TensorRT operators.
    TensorRt,
}

/// Execution choices owned by the adapter, outside the geometry contract.
#[derive(Clone, Debug)]
pub struct ExecutionConfig {
    /// Requested provider. This is not a claim of exclusive device use.
    pub provider: Provider,
    /// ONNX Runtime CPU thread count.
    pub threads: usize,
    /// NVIDIA device index. Used only with CUDA or TensorRT.
    pub nvidia_device_id: i32,
    /// Optional TensorRT engine cache. Keep this separate for each runtime and GPU.
    /// Model content determines the cache prefix. Runtime upgrades require a new directory.
    pub engine_cache_directory: Option<PathBuf>,
    /// Maximum TensorRT builder workspace. This does not cap total GPU memory.
    pub tensor_rt_workspace_bytes: usize,
}
impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            provider: Provider::Cpu,
            threads: 4,
            nvidia_device_id: 0,
            engine_cache_directory: None,
            tensor_rt_workspace_bytes: 256 * 1024 * 1024,
        }
    }
}

/// Load the host-selected ONNX Runtime once before constructing adapters.
///
/// # Errors
/// Returns an error if the library cannot load or the environment already exists.
pub fn initialize_blocking(library: &Path) -> Result<(), InferenceError> {
    let initialized = ort::init_from(library)
        .map_err(|source| InferenceError::Load {
            path: library.to_owned(),
            source,
        })?
        .with_telemetry(false)
        .commit();
    if !initialized {
        return Err(InferenceError::Invalid(
            "ONNX environment is already initialized".into(),
        ));
    }
    Ok(())
}

pub(crate) fn session_blocking(
    path: &Path,
    config: &ExecutionConfig,
) -> Result<Session, InferenceError> {
    validate_config(config)?;
    let providers = providers_blocking(path, config)?;
    Session::builder()
        .map_err(|e| runtime("create session", e))?
        .with_intra_threads(config.threads)
        .map_err(|e| runtime("set CPU threads", e))?
        .with_execution_providers(providers)
        .map_err(|e| {
            runtime(
                format!("configure requested {:?} provider", config.provider),
                e,
            )
        })?
        .commit_from_file(path)
        .map_err(|e| runtime(format!("load model {}", path.display()), e))
}

fn validate_config(config: &ExecutionConfig) -> Result<(), InferenceError> {
    if !(1..=64).contains(&config.threads) {
        return Err(InferenceError::Invalid(
            "CPU threads must be 1 through 64".into(),
        ));
    }
    if matches!(config.provider, Provider::Cuda | Provider::TensorRt) && config.nvidia_device_id < 0
    {
        return Err(InferenceError::Invalid(
            "NVIDIA device index must be nonnegative".into(),
        ));
    }
    if matches!(config.provider, Provider::TensorRt) && config.tensor_rt_workspace_bytes == 0 {
        return Err(InferenceError::Invalid(
            "TensorRT workspace must be positive".into(),
        ));
    }
    Ok(())
}

fn providers_blocking(
    path: &Path,
    config: &ExecutionConfig,
) -> Result<Vec<ep::ExecutionProviderDispatch>, InferenceError> {
    let cuda = || {
        ep::CUDA::default()
            .with_device_id(config.nvidia_device_id)
            .build()
            .error_on_failure()
    };
    Ok(match config.provider {
        Provider::Cpu => vec![],
        Provider::Cuda => vec![cuda()],
        Provider::TensorRt => {
            let mut provider = ep::TensorRT::default()
                .with_device_id(config.nvidia_device_id)
                .with_max_workspace_size(config.tensor_rt_workspace_bytes);
            if let Some(directory) = &config.engine_cache_directory {
                std::fs::create_dir_all(directory).map_err(|source| InferenceError::Cache {
                    path: directory.clone(),
                    source,
                })?;
                let bytes = std::fs::read(path).map_err(|source| InferenceError::Io {
                    path: path.to_owned(),
                    source,
                })?;
                provider = provider
                    .with_engine_cache(true)
                    .with_engine_cache_path(directory.display())
                    .with_engine_cache_prefix(format!("{:x}", Sha256::digest(bytes)));
            }
            vec![provider.build().error_on_failure(), cuda()]
        }
        Provider::CoreMlAne | Provider::CoreMlGpu => {
            let units = match config.provider {
                Provider::CoreMlAne => ep::coreml::ComputeUnits::CPUAndNeuralEngine,
                _ => ep::coreml::ComputeUnits::CPUAndGPU,
            };
            vec![
                ep::CoreML::default()
                    .with_compute_units(units)
                    .with_static_input_shapes(true)
                    .with_model_format(ep::coreml::ModelFormat::MLProgram)
                    .build()
                    .error_on_failure(),
            ]
        }
    })
}

#[cfg(test)]
mod tests;
