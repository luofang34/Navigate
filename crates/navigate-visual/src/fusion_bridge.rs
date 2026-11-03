//! Admission of visual map-relative camera poses as fusion observations.
//!
//! [`position_fix_observation`] converts a [`crate::Estimate`] into a
//! [`navigate_fusion::Observation`] with a `PositionFix` value.
//! [`visual_pose_observation`] converts it into a `VisualPose` value with the
//! camera position and the vehicle body attitude (ADR-0020).
//! [`navigate_fusion::InertialNavigationFilter`] applies a visual pose as one
//! update of dimension six. [`VisualFixSource`] applies both conversions to
//! one capture stream and refuses repeated evidence and frames out of order.
//!
//! The host supplies the source identity, the clock domain, and a
//! [`VisualErrorBudget`]. A visual pose also needs a [`VisualAttitudeBudget`]
//! with the camera mounting. The adapter adds the budgets and the
//! point-coordinate displacement to the image-geometry covariance. The
//! required geometry-model term covers the fitted-position effect of map
//! distortion. The adapter does not estimate map, geometry-model,
//! calibration, or datum errors. A budget of zero is refused.
//!
//! The position is the position of the camera optical centre. The host
//! removes the camera lever arm before the observation enters a
//! vehicle-position filter, or includes it in the calibration term of the
//! budget.
//!
//! Each observation declares the body-fixed frame of
//! [`crate::LocalFrame::reference_ellipsoid`]. A
//! [`crate::LocalFrame::BodyTangentEnu`] frame on the Moon or on Mars gives
//! an observation that a filter on that body admits (ADR-0013).
//!
//! Each accepted fix names its [`crate::MapRevision`]. Fixes from
//! one map release share map error. The filter treats them as independent,
//! so the budget must be conservative. See ADR-0007.
//!
//! [`EvidenceIndependence::restricted_by`] applies the fusion label of a
//! [`crate::session`] anchor. Anchors whose map cell already gave a
//! fix in the session are refused.

mod adapter;
mod budget;
mod capture;
mod error;
mod observation;
mod prior;

pub use adapter::{
    DEFAULT_MAX_RETAINED_DIGESTS, EvidenceIndependence, VisualFix, VisualFixSource,
    VisualSourceIdentity,
};
pub use budget::{VerticalDatum, VisualAttitudeBudget, VisualErrorBudget};
pub use error::VisualFusionError;
pub use observation::{position_fix_observation, visual_pose_observation};
pub use prior::{FrameCapture, PropagationBudget, search_prior};

pub use capture::{
    AttitudeSample, CameraAlignment, CaptureError, ClockAlignment, TimedCapture, camera_capture,
};
