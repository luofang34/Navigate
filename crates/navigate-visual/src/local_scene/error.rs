//! Input and numerical failures in local scene refinement.
use crate::VisualError;
use thiserror::Error;
/// A local scene cannot be refined under its declared conditions.
#[derive(Debug, Error)]
pub enum LocalSceneError {
    /// The common calibration is invalid.
    #[error("invalid local scene camera calibration")]
    Camera {
        /// Camera validation failure.
        #[source]
        source: VisualError,
    },
    /// The requested problem exceeds this solver's allocation or iteration limits.
    #[error(
        "local refinement limits exceeded: {cameras} free cameras, {points} points, {iterations} iterations"
    )]
    Limits {
        /// Number of free camera poses.
        cameras: usize,
        /// Number of scene points.
        points: usize,
        /// Requested iterations.
        iterations: usize,
    },
    /// An observation has an invalid identity, pose, or duplicate identity.
    #[error("invalid local scene camera {index}: {reason}")]
    Observation {
        /// Camera index in the input.
        index: usize,
        /// Failed invariant.
        reason: &'static str,
    },
    /// A point or one of its source observations is invalid.
    #[error("invalid local scene point {feature_id}: {reason}")]
    Point {
        /// Persistent scene point identity.
        feature_id: u64,
        /// Failed invariant.
        reason: &'static str,
    },
    /// The declared coordinate constraints do not support a scene component.
    #[error("local scene has no observed coordinate constraints for cameras {cameras:?}")]
    Unconstrained {
        /// Camera indices in the unconstrained component.
        cameras: Vec<usize>,
    },
    /// The declared coordinate gauge is invalid.
    #[error("invalid scene coordinate gauge ({origin_camera}, {scale_camera}): {reason}")]
    Gauge {
        /// Camera that defines the coordinate origin.
        origin_camera: usize,
        /// Camera that defines the arbitrary baseline length.
        scale_camera: usize,
        /// Failed gauge invariant.
        reason: &'static str,
    },
    /// The numerical step could not be computed from the supplied geometry.
    #[error("local refinement failed for {cameras} cameras and {points} points")]
    Numerical {
        /// Number of camera poses.
        cameras: usize,
        /// Number of scene points.
        points: usize,
    },
}
