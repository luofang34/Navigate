//! Rectification failures preserve the rejected image or memory constraint.
use thiserror::Error;

/// A calibrated pixel map or its input image cannot be used.
#[derive(Debug, Error)]
pub enum RectificationError {
    /// Invalid pinhole calibration.
    #[error(transparent)]
    Calibration(#[from] crate::VisualError),
    /// Source or target exceeds the supported dimensions.
    #[error("rectification dimensions exceed 8192 pixels per side")]
    Dimensions,
    /// Non-finite distortion coefficients.
    #[error("rectification coefficients must be finite")]
    Coefficients,
    /// Map storage exceeds the caller's budget.
    #[error("rectification map requires {required} bytes; limit is {limit}")]
    Budget {
        /// Required map bytes.
        required: usize,
        /// Caller-supplied limit.
        limit: usize,
    },
    /// A target pixel folds or leaves the source image.
    #[error("rectification pixel {pixel:?} has a folded or uncovered mapping")]
    Mapping {
        /// Target pixel that failed validation.
        pixel: [u32; 2],
    },
    /// The source image is not the size of the calibrated sensor.
    #[error("rectification source size {received:?}; expected {expected:?}")]
    SourceSize {
        /// Bound sensor dimensions.
        expected: (u32, u32),
        /// Supplied image dimensions.
        received: (u32, u32),
    },
    /// Output storage does not have the declared dimensions.
    #[error("rectification output size does not match its camera")]
    OutputSize,
    /// The allocator could not reserve map or image storage.
    #[error("cannot reserve rectification storage")]
    Allocation(#[from] std::collections::TryReserveError),
}
