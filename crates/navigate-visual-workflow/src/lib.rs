//! Bounded camera localization with reusable reference and query features.

mod backend;
mod error;
mod localizer;
mod pair;
pub use pair::PairLocalizer;

pub use backend::{FeatureBackend, FeatureMetadata};
pub use error::WorkflowError;
pub use localizer::{
    CandidateReport, FeatureLocalizer, FrameReport, PreparedReference, ResourceLimits, StageTimes,
};

#[cfg(test)]
mod test_scene;
