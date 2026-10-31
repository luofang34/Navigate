//! Bounded visual localization stages with reusable reference and query features.

mod error;
mod localizer;
mod pair;
#[cfg(test)]
mod test_scene;

pub use error::WorkflowError;
pub use localizer::{
    CandidateReport, FeatureLocalizer, FrameReport, PreparedReference, ResourceLimits, StageTimes,
};
pub use pair::PairLocalizer;
