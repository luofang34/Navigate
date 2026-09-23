//! Failures of imagery providers and file package publication.

use navigate_imagery::ImageryError;
use thiserror::Error;

/// Failure to fetch, render, or publish reference imagery.
#[derive(Debug, Error)]
pub enum ProviderError {
    /// The package or coverage contract refused the data.
    #[error(transparent)]
    Imagery(#[from] ImageryError),
    /// A file operation failed.
    #[error("Cannot access {path}")]
    Io {
        /// File path without provider credentials.
        path: std::path::PathBuf,
        /// The operating system failure.
        #[source]
        source: std::io::Error,
    },
    /// A provider request failed. URLs are removed from the source error.
    #[error("Imagery provider request failed: {operation}")]
    Provider {
        /// The request purpose.
        operation: &'static str,
        /// The HTTP failure without a signed URL.
        #[source]
        source: reqwest::Error,
    },
    /// A raster operation failed.
    #[error("Raster processing failed: {operation}")]
    Raster {
        /// The operation purpose; signed URLs must not be logged.
        operation: &'static str,
        /// The GDAL failure, which can contain private provider credentials.
        #[source]
        source: gdal::errors::GdalError,
    },
    /// Manifest encoding failed.
    #[error("Cannot encode imagery manifest")]
    Json(#[from] serde_json::Error),
    /// Image decoding or encoding failed.
    #[error("Cannot decode or encode imagery tile")]
    Image(#[from] image::ImageError),
}

pub(crate) fn invalid(reason: impl Into<String>) -> ProviderError {
    ProviderError::Imagery(ImageryError::Invalid {
        reason: reason.into(),
    })
}
