//! Errors from visual observation validation and pose estimation.

use thiserror::Error;

/// A visual observation cannot produce an accepted camera pose.
#[derive(Debug, Error)]
pub enum VisualError {
    /// A parameter is invalid.
    #[error("invalid visual parameter: {field}")]
    Invalid {
        /// Name of the invalid parameter.
        field: &'static str,
    },
    /// The reference and query images have different dimensions.
    #[error("image dimensions differ: query {query:?}, reference {reference:?}")]
    Dimensions {
        /// Query image dimensions.
        query: (u32, u32),
        /// Reference image dimensions.
        reference: (u32, u32),
    },
    /// Too few correspondences pass the geometric checks.
    #[error("only {found} visual inliers; need {required}")]
    InsufficientMatches {
        /// Number of accepted correspondences.
        found: usize,
        /// Minimum required number.
        required: usize,
    },
    /// The image does not constrain a full pose.
    #[error("visual geometry does not constrain the camera pose")]
    DegenerateGeometry,
    /// The estimated pose is outside the declared prior bounds.
    #[error("visual correction exceeds prior: {meters:.2} m, {radians:.4} rad")]
    OutsidePrior {
        /// Translation from the prior in metres.
        meters: f64,
        /// Rotation from the prior in radians.
        radians: f64,
    },
    /// Frame times must increase within one stream.
    #[error("frame time {received_ns} does not follow {previous_ns}")]
    FrameOrder {
        /// Last admitted capture time.
        previous_ns: u64,
        /// Incoming capture time.
        received_ns: u64,
    },
    /// A hardware or model backend failed.
    #[error("image matcher {backend} failed")]
    Backend {
        /// Backend name.
        backend: String,
        /// Backend failure.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
}
