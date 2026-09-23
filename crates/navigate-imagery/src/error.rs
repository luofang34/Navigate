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
}

pub(crate) fn invalid(reason: impl Into<String>) -> ImageryError {
    ImageryError::Invalid {
        reason: reason.into(),
    }
}
