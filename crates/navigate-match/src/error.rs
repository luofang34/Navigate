//! Matcher errors with the name and the size that failed.
use thiserror::Error;

/// A failure to load weights or to run a matcher.
#[derive(Debug, Error)]
pub enum MatchError {
    /// The weight file is not a valid safetensors file.
    #[error("weight file is not valid safetensors")]
    WeightFile(#[source] safetensors::SafeTensorError),
    /// A tensor that the model needs is not in the weight file.
    #[error("weight tensor {name} is missing")]
    MissingWeight {
        /// Tensor name.
        name: String,
    },
    /// A weight tensor has a different shape than the model needs.
    #[error("weight tensor {name} has shape {actual:?}, the model needs {expected:?}")]
    WeightShape {
        /// Tensor name.
        name: String,
        /// Shape in the file.
        actual: Vec<usize>,
        /// Shape that the model needs.
        expected: Vec<usize>,
    },
    /// A weight tensor has a data type that the loader does not read.
    #[error("weight tensor {name} has data type {dtype}; use F32 or F16")]
    WeightType {
        /// Tensor name.
        name: String,
        /// Data type in the file.
        dtype: String,
    },
    /// The weight file contains tensors that the model does not use.
    #[error("weight file has {count} tensors that the model does not use, for example {example}")]
    UnusedWeights {
        /// Number of unused tensors.
        count: usize,
        /// One unused tensor name.
        example: String,
    },
    /// The serialized weight file could not be written.
    #[error("weights could not be serialized")]
    Serialize(#[source] safetensors::SafeTensorError),
    /// An input image has a size that the matcher does not accept.
    #[error("image of {width} × {height} pixels is outside 16 through 8192 pixels per side")]
    ImageSize {
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
    },
    /// The device could not run the model or read back a result.
    #[error("device failed during {stage}: {message}")]
    Device {
        /// Model stage.
        stage: &'static str,
        /// Backend message.
        message: String,
    },
    /// The model returned values that are not finite or that have a wrong length.
    #[error("model output is not valid: {0}")]
    Output(&'static str),
}
