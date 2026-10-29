//! Error types for semantic segmentation, `.npy`/`.npz` archives, and fine-tuning.

use std::path::PathBuf;
use thiserror::Error;

/// Errors returned by [`crate`] operations.
#[derive(Debug, Error)]
pub enum SegmentError {
    /// Filesystem read or write failed.
    #[error("I/O failed for {}: {source}", path.display())]
    Io {
        /// Path associated with the I/O operation.
        path: PathBuf,
        /// Underlying operating-system error.
        #[source]
        source: std::io::Error,
    },
    /// JSON configuration or report failed to parse or serialize.
    #[error("JSON failure for {context}: {source}")]
    Json {
        /// Context string describing the JSON document.
        context: String,
        /// Underlying parser error.
        #[source]
        source: serde_json::Error,
    },
    /// SafeTensors model archive could not be parsed or written.
    #[error("SafeTensors failure: {0}")]
    Safetensors(String),
    /// Image decoding or encoding failed.
    #[error("image failure for {}: {source}", path.display())]
    Image {
        /// Image path or label.
        path: PathBuf,
        /// Underlying decoder/encoder error.
        #[source]
        source: image::ImageError,
    },
    /// ZIP archive failure when reading or writing `.npz` files.
    #[error("NPZ archive failure: {0}")]
    Archive(String),
    /// Tensor shape or dtype mismatch.
    #[error("shape mismatch for {name}: expected {expected:?}, got {actual:?}")]
    Shape {
        /// Tensor or array identifier.
        name: String,
        /// Expected dimensions.
        expected: Vec<usize>,
        /// Actual dimensions.
        actual: Vec<usize>,
    },
    /// Invalid configuration, policy, or training input.
    #[error("invalid segmentation input: {0}")]
    Invalid(String),
}
