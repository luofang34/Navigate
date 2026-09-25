//! Context for invalid reconstruction inputs and unsupported geometry.
use crate::{LocalSceneError, VisualError};
/// A reconstruction cannot proceed with these inputs or geometry.
#[derive(Debug, thiserror::Error)]
pub enum ReconstructionError {
    /// The pinhole camera is invalid.
    #[error("invalid reconstruction camera: {source}")]
    Camera {
        /// Intrinsic validation failure.
        #[source]
        source: VisualError,
    },
    /// The supplied poses do not cover the exact source graph.
    #[error("scene has {supplied} cameras; source graph requires {required}")]
    CameraCount {
        /// Number of supplied camera estimates.
        supplied: usize,
        /// Number of source observations.
        required: usize,
    },
    /// A supplied scene camera has an invalid pose.
    #[error("invalid scene camera {index}: {source}")]
    SceneCamera {
        /// Source camera index.
        index: usize,
        /// Pose validation failure.
        #[source]
        source: VisualError,
    },
    /// Resource bounds apply to one local group, not to the source video.
    #[error(
        "reconstruction group exceeds bounds: {cameras} cameras, {tracks} tracks, {observations} observations"
    )]
    Limits {
        /// Number of source cameras.
        cameras: usize,
        /// Number of feature tracks.
        tracks: usize,
        /// Number of image links.
        observations: usize,
    },
    /// An observation digest is invalid or repeated.
    #[error("invalid reconstruction observation {index}: {reason}")]
    Observation {
        /// Source camera index.
        index: usize,
        /// Validation failure.
        reason: &'static str,
    },
    /// A proposed feature has invalid or repeated image support.
    #[error("invalid image track {feature_id}: {reason}")]
    Track {
        /// Host feature identity.
        feature_id: u64,
        /// Validation failure.
        reason: &'static str,
    },
    /// The initial camera pair or relative pose is invalid.
    #[error("invalid reconstruction seed {camera_indices:?}: {reason}")]
    Seed {
        /// Source pair.
        camera_indices: [usize; 2],
        /// Validation failure.
        reason: &'static str,
    },
    /// This geometric adapter could not propose a relative pose.
    #[error("five-point geometry failed for {camera_indices:?}: {source}")]
    TwoView {
        /// Source pair.
        camera_indices: [usize; 2],
        /// Adapter failure.
        #[source]
        source: kornia_3d::pose::TwoViewError,
    },
    /// A seed lacks enough triangulation support.
    #[error("seed {camera_indices:?} has only {points} triangulated points")]
    Support {
        /// Source pair.
        camera_indices: [usize; 2],
        /// Supported scene points.
        points: usize,
    },
    /// Joint refinement failed without changing the input graph.
    #[error("local reconstruction refinement failed: {source}")]
    Refinement {
        /// Original refinement failure.
        #[source]
        source: LocalSceneError,
    },
}
