//! Native ONNX image matchers for the Navigate visual pipeline.
//!
//! The host supplies model files and initializes ONNX Runtime once. Model files
//! retain their own licences. No weights are embedded or downloaded by this crate.
//! Matchers return pixels. [`navigate_visual::PoseVerifier`] owns pose acceptance.
//! Provider selection can include CPU operators. It is not device telemetry.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

mod adapter;
mod features;
mod lighterglue;
mod loftr;
mod native_runtime;
mod preprocessing;
mod selection;
mod superpoint;
mod xfeat;
mod xfeat_dense;

pub use adapter::{MatcherFiles, OnnxMatcher};
pub use loftr::LoFtrMatcher;
pub use native_runtime::{ExecutionConfig, InferenceError, Provider, initialize_blocking};

pub use selection::{CandidateFailure, MatcherCandidate, MatcherSelection, SelectionError};
