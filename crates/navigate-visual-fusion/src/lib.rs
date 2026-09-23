//! Admission of visual map-relative camera poses as fusion position fixes.
//!
//! [`VisualFixSource`] converts a [`navigate_visual::Estimate`] into a
//! [`navigate_fusion::Observation`] with a `PositionFix` value. The host
//! supplies the source identity, the clock domain, and a
//! [`VisualErrorBudget`]. The adapter adds the budget and the frame-model
//! error to the image-geometry covariance. The adapter does not estimate map,
//! calibration, or datum errors. A budget of zero is refused.
//!
//! The fix is the position of the camera optical centre. Attitude is not
//! transferred, because the filter has no attitude state. The host removes
//! the camera lever arm before the fix enters a vehicle-position filter, or
//! includes it in the calibration term of the budget.
//!
//! Each accepted fix names its [`navigate_visual::MapRevision`]. Fixes from
//! one map release share map error. The filter treats them as independent,
//! so the budget must be conservative. See ADR-0007.

mod adapter;
mod budget;
mod error;

pub use adapter::{EvidenceIndependence, VisualFix, VisualFixSource, VisualSourceIdentity};
pub use budget::{VerticalDatum, VisualErrorBudget};
pub use error::VisualFusionError;
