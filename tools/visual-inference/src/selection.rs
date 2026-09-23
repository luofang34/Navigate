//! Ordered model and runtime selection for a packaged native application.
use crate::{ExecutionConfig, InferenceError, MatcherFiles, OnnxMatcher, Provider};
use thiserror::Error;

/// One packaged model and runtime choice, in host preference order.
///
/// Keep hardware-specific artifacts here, outside the image matcher contract.
pub struct MatcherCandidate {
    /// Model files compatible with this runtime choice.
    pub files: MatcherFiles,
    /// Requested runtime configuration. Unsupported operators can use CPU.
    pub execution: ExecutionConfig,
}

/// One model and runtime choice could not initialize.
#[derive(Debug, Error)]
#[error("matcher candidate {index} with {requested_provider:?}: {source}")]
pub struct CandidateFailure {
    /// Zero-based position in the host's preference list.
    pub index: usize,
    /// Provider requested for this candidate.
    pub requested_provider: Provider,
    /// Model, configuration, or runtime failure with its original context.
    #[source]
    pub source: InferenceError,
}

/// Session initialization could not select a matcher.
#[derive(Debug, Error)]
pub enum SelectionError {
    /// The host supplied no choices.
    #[error("no matcher candidates were supplied")]
    Empty,
    /// Every supplied choice failed to initialize.
    #[error("all matcher candidates failed; final attempt: {last}")]
    AllFailed {
        /// Earlier failures, in preference order.
        previous: Vec<CandidateFailure>,
        /// Final failure, retained as the error source.
        #[source]
        last: CandidateFailure,
    },
}

/// Requested configuration selected by successful session initialization.
///
/// This is not operator placement or hardware utilization telemetry. Model
/// operators can still use CPU. Inference failures remain errors.
#[derive(Debug)]
pub struct MatcherSelection {
    /// Zero-based position of the selected candidate.
    pub index: usize,
    /// Provider requested for the selected candidate.
    pub requested_provider: Provider,
    /// Earlier initialization failures, in preference order.
    pub rejected: Vec<CandidateFailure>,
}

impl OnnxMatcher {
    /// Try packaged model and runtime choices in host preference order.
    ///
    /// Include an explicit CPU candidate to allow CPU fallback. Each attempt
    /// creates real sessions. A runtime capability flag alone is insufficient.
    /// The selected matcher keeps its model digests and requested provider in its
    /// identity. Selection does not run geometric verification or add evidence.
    /// Use [`Self::load_blocking`] for a strict single-provider request.
    ///
    /// # Errors
    /// Returns every initialization failure if no candidate succeeds. A selected
    /// session can still fail during inference; this method does not retry frames.
    pub fn load_preferred_blocking(
        candidates: Vec<MatcherCandidate>,
        keypoints: usize,
    ) -> Result<(Self, MatcherSelection), SelectionError> {
        let mut rejected = Vec::new();
        for (index, candidate) in candidates.into_iter().enumerate() {
            let requested_provider = candidate.execution.provider;
            match Self::load_blocking(candidate.files, candidate.execution, keypoints) {
                Ok(matcher) => {
                    tracing::info!(
                        index,
                        ?requested_provider,
                        rejected = rejected.len(),
                        "selected native matcher configuration"
                    );
                    return Ok((
                        matcher,
                        MatcherSelection {
                            index,
                            requested_provider,
                            rejected,
                        },
                    ));
                }
                Err(source) => {
                    tracing::debug!(index, ?requested_provider, %source, "matcher candidate failed");
                    rejected.push(CandidateFailure {
                        index,
                        requested_provider,
                        source,
                    });
                }
            }
        }
        match rejected.pop() {
            Some(last) => Err(SelectionError::AllFailed {
                previous: rejected,
                last,
            }),
            None => Err(SelectionError::Empty),
        }
    }
}

#[cfg(test)]
mod tests;
