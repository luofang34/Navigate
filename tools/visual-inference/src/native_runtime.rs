//! Explicit native runtime setup and contextual adapter errors.
use ort::{ep, session::Session};
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
}

/// Execution choices owned by the adapter, outside the geometry contract.
#[derive(Clone, Debug)]
pub struct ExecutionConfig {
    /// Requested provider. This is not a claim of exclusive device use.
    pub provider: Provider,
    /// ONNX Runtime CPU thread count.
    pub threads: usize,
}
impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            provider: Provider::Cpu,
            threads: 4,
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
    if !(1..=64).contains(&config.threads) {
        return Err(InferenceError::Invalid(
            "CPU threads must be 1 through 64".into(),
        ));
    }
    let mut builder = Session::builder()
        .map_err(|e| runtime("create session", e))?
        .with_intra_threads(config.threads)
        .map_err(|e| runtime("set CPU threads", e))?;
    if !matches!(config.provider, Provider::Cpu) {
        let units = match config.provider {
            Provider::CoreMlAne => ep::coreml::ComputeUnits::CPUAndNeuralEngine,
            _ => ep::coreml::ComputeUnits::CPUAndGPU,
        };
        builder = builder
            .with_execution_providers([ep::CoreML::default()
                .with_compute_units(units)
                .with_static_input_shapes(true)
                .with_model_format(ep::coreml::ModelFormat::MLProgram)
                .build()
                .error_on_failure()])
            .map_err(|e| runtime("configure Core ML", e))?;
    }
    builder
        .commit_from_file(path)
        .map_err(|e| runtime(format!("load model {}", path.display()), e))
}
