use thiserror::Error;

/// Failure to plan or construct verified reference data.
#[derive(Debug, Error)]
pub enum ImageryError {
    /// A selection or source violates the supported data contract.
    #[error("Invalid imagery data: {reason}")]
    Invalid {
        /// The failed requirement and its context.
        reason: String,
    },
    /// Serialization of a manifest failed.
    #[error("Cannot encode imagery manifest")]
    Json(#[from] serde_json::Error),
    /// Image decoding or encoding failed.
    #[error("Cannot decode or encode imagery tile")]
    Image(#[from] image::ImageError),
    /// A native file operation failed.
    #[cfg(feature = "native")]
    #[error("Cannot access {path}")]
    Io {
        /// File path without provider credentials.
        path: std::path::PathBuf,
        /// The operating system failure.
        #[source]
        source: std::io::Error,
    },
    /// A provider request failed. URLs are removed from the source error.
    #[cfg(feature = "native")]
    #[error("Imagery provider request failed: {operation}")]
    Provider {
        /// The request purpose.
        operation: &'static str,
        /// The HTTP failure without a signed URL.
        #[source]
        source: reqwest::Error,
    },
    /// A raster operation failed.
    #[cfg(feature = "native")]
    #[error("Raster processing failed: {operation}")]
    Raster {
        /// The operation purpose; signed URLs must not be logged.
        operation: &'static str,
        /// The GDAL failure, which can contain private provider credentials.
        #[source]
        source: gdal::errors::GdalError,
    },
}

pub(crate) fn invalid(reason: impl Into<String>) -> ImageryError {
    ImageryError::Invalid {
        reason: reason.into(),
    }
}
