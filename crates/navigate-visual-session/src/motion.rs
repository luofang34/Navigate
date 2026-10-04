//! Relative camera motion between two frames and the source of its scale.

use crate::{FrameKey, pose::Pose, pose::from_camera};
use navigate_visual::{CameraPose, MapRevision, SurfaceTrackUpdate, TrackingProposal};

/// The source of the metric scale of one motion step.
///
/// Image matches alone fix rotation and the direction of travel. Metric
/// distance needs depth or another sensor. Map-depth tracking reads depth
/// from the map render, so it is not map-independent odometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScaleSource {
    /// Depth rendered from the map at the previous estimated pose.
    ///
    /// Map elevation errors and errors in the previous pose change the scale.
    /// The step needs map elevation near the camera, but no imagery match.
    MapDepth {
        /// Map revision of the rendered depth.
        map: MapRevision,
    },
    /// World points that were fixed when map depth seeded them.
    ///
    /// Scale persists while the seeded points stay in view. The points carry
    /// the map and pose errors of the frame that seeded them.
    SeededMapPoints {
        /// Map revision of the seed depth.
        map: MapRevision,
    },
    /// A host sensor with metric output, for example a rangefinder or wheel odometry.
    External {
        /// Host name of the sensor and its processing.
        sensor: String,
    },
    /// No metric scale, for example a five-point relative pose.
    Unknown,
}

impl ScaleSource {
    /// True when the step has a metric scale.
    pub fn is_metric(&self) -> bool {
        !matches!(self, Self::Unknown)
    }
}

/// Measured motion of the camera from one frame to a later frame.
#[derive(Clone, Debug, PartialEq)]
pub struct RelativeMotion {
    /// Start frame. It must be the newest frame with odometry in its epoch.
    pub from: FrameKey,
    /// End frame. It must be a later frame of the same epoch.
    pub to: FrameKey,
    /// Pose of the `to` camera in the `from` camera frame.
    pub from_to: Pose,
    /// Source of the metric scale.
    pub scale: ScaleSource,
    /// Evidence digest of the start frame.
    pub from_observation_sha256: String,
    /// Evidence digest of the end frame.
    pub to_observation_sha256: String,
}

impl RelativeMotion {
    /// Motion from a map-depth tracking result.
    ///
    /// `previous` is the pose that the host used to render the surface. Only
    /// the relative motion enters the session. The absolute proposal pose is
    /// conditional on `previous` and is not a map measurement.
    pub fn from_tracking(
        from: FrameKey,
        to: FrameKey,
        previous: &CameraPose,
        proposal: &TrackingProposal,
    ) -> Self {
        Self {
            from,
            to,
            from_to: from_camera(previous).inv_mul(&from_camera(&proposal.pose)),
            scale: ScaleSource::MapDepth {
                map: proposal.map.clone(),
            },
            from_observation_sha256: proposal.reference_observation_sha256.clone(),
            to_observation_sha256: proposal.observation_sha256.clone(),
        }
    }

    /// Motion from a `SurfaceTracks` update.
    pub fn from_surface_tracks(
        from: FrameKey,
        to: FrameKey,
        previous: &CameraPose,
        update: &SurfaceTrackUpdate,
    ) -> Self {
        let mut motion = Self::from_tracking(from, to, previous, &update.proposal);
        motion.scale = ScaleSource::SeededMapPoints {
            map: update.proposal.map.clone(),
        };
        motion
    }
}
