//! Continuous camera odometry, map anchoring, revisit closure, and per-frame
//! height for one camera session.
//!
//! [`VisualSession`] keeps three layers apart:
//!
//! - Odometry: each continuity segment has a continuous camera pose that
//!   relative motion builds. A map correction never changes it.
//! - Map anchoring: map matches (`navigate_visual::Estimate`) enter a pose
//!   graph with their declared error. Each segment gets a `map_from_odom`
//!   transform, and [`VisualSession::pose_at`] gives the map pose of a frame.
//! - Revision: anchors, revisit closures, and the memory bound change past
//!   map poses. Each change is a [`TrajectoryRevision`] with the affected frame
//!   ranges, so consumers re-project those frames.
//!
//! Map-depth tracking (`navigate_visual::PoseVerifier::track`) reads depth
//! rendered from the map. It needs
//! map elevation but no imagery match. It is not map-independent odometry,
//! and its error grows from the last anchor. Motion without metric scale is
//! refused. See [`ScaleSource`].
//!
//! The session does not estimate inertial states. It is not visual-inertial
//! odometry. The navigation filter accepts only map anchors as position fixes;
//! [`FusionEligibility`] labels the anchors that share map error.
//!
//! The session is sans-IO and deterministic. It never reads a clock, never
//! decodes media, and never runs a matcher. Media playback stays independent.

mod anchor;
mod calibration;
mod config;
mod error;
mod events;
mod evidence;
mod frame;
mod graph;
mod height;
mod motion;
mod pose;
mod session;
mod store;

pub use anchor::{AnchorBudget, AnchorObservation, MapReliability, RegionChange, SurfaceModel};
pub use calibration::{Calibration, CameraMount, LensModel, VerticalReference};
pub use config::{
    AnchorPolicy, DriftModel, KeyframePolicy, RevisitPolicy, SessionConfig, SessionLimits,
};
pub use error::SessionError;
pub use events::{
    AnchorRejection, EventBatch, RevisedRange, RevisionCause, SegmentId, SegmentStart,
    SessionEvent, TrajectoryRevision,
};
pub use evidence::{FusionEligibility, MapCell};
pub use frame::{CaptureTime, ContinuityEpoch, FrameKey, FrameRecord, MediaTime, StreamId};
pub use height::{
    AxisRange, Height, HeightBasis, HeightEstimate, HeightInputs, HeightSample, HeightSensor,
    RangeSource, TerrainSample, TerrainSource, Unobservable, estimate_height,
};
pub use motion::{RelativeMotion, ScaleSource};
pub use pose::{Pose, from_camera, to_camera};
pub use session::{
    AnchorDecision, Confirmation, FramePose, MapPose, MotionOutcome, OdometryPose,
    RevisitCandidate, RevisitConstraint, RevisitDecision, RevisitEvidence, SessionUsage, Unlocated,
    VisualSession,
};
