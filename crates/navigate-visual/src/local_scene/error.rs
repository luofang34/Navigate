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
    /// Fixed poses do not constrain a connected scene component.
    #[error("local scene has no observed fixed baseline for cameras {cameras:?}")]
    Unconstrained {
        /// Camera indices in the unconstrained component.
        cameras: Vec<usize>,
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
