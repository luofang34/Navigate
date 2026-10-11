//! Native ONNX adapters for image matching and reference retrieval.
//!
//! The host supplies model files and initializes ONNX Runtime once. Model files
//! retain their own licences. No weights are embedded or downloaded by this crate.
//! Matchers return pixels. [`navigate_visual::PoseVerifier`] owns pose acceptance.
//! Provider selection can include CPU operators. It is not device telemetry.

mod adapter;
mod camp;
mod diagnostics;
mod feature_cache;
mod features;
mod lighterglue;
mod lightglue;
mod loftr;
mod native_runtime;
mod preprocessing;
#[cfg(all(feature = "rustnn-coreml", target_os = "macos"))]
mod rustnn_dense;
mod selection;
mod staged;
mod superpoint;
mod xfeat;
mod xfeat_dense;
mod xfeat_image;

pub use adapter::{MatcherFiles, OnnxMatcher};
pub use camp::{CampIndex, CampRetriever};
pub use diagnostics::TensorIoStatistics;
pub use lightglue::LightGlueMatcher;
pub use loftr::LoFtrMatcher;
pub use native_runtime::{ExecutionConfig, InferenceError, Provider, initialize_blocking};

pub use staged::{XFeatFeatures, XFeatGraph, XFeatStages};

pub use selection::{CandidateFailure, MatcherCandidate, MatcherSelection, SelectionError};
