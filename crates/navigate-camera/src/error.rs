//! Errors of the camera model.

/// A camera description or an image that the model cannot use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CameraError {
    /// A field has a value outside its valid range.
    #[error("invalid camera field: {field}")]
    Invalid {
        /// The name of the field.
        field: &'static str,
    },
    /// A pixel buffer is shorter than its size and stride need.
    #[error("pixel buffer holds {actual} samples; {needed} are needed")]
    BufferTooShort {
        /// Samples that the size and the stride need.
        needed: usize,
        /// Samples in the buffer.
        actual: usize,
    },
    /// An output buffer has a different length from the image.
    #[error("output buffer holds {actual} samples; the image has {needed}")]
    OutputSize {
        /// Samples in the image.
        needed: usize,
        /// Samples in the output buffer.
        actual: usize,
    },
    /// The iteration that removes lens distortion did not converge.
    #[error("lens distortion inversion did not converge")]
    NotConverged,
}
