//! Preserve geometric alternatives without fusing shared observations.
use crate::{Estimate, Frame, VisualError};
use std::collections::BTreeMap;
/// Host-assigned search hypothesis identity within one observation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CandidateId(
    /// Opaque identifier assigned by the candidate search.
    pub u64,
);
/// Visual selection among the candidates that were actually evaluated.
///
/// A unique result does not establish absolute accuracy or exclude places that
/// retrieval did not search. This decision contains no fused pose or covariance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateDecision {
    /// No evaluated candidate passed geometry and prior bounds.
    Rejected,
    /// Exactly one evaluated candidate passed.
    Unique(CandidateId),
    /// Multiple candidates remain possible. Preserve them for later evidence.
    Unresolved(Vec<CandidateId>),
}
/// Latest geometric result for each hypothesis of one observation.
///
/// Refinement replaces a hypothesis result. It does not add an independent
/// measurement or combine precision. Different hypotheses remain separate.
pub struct CandidateResults {
    evidence: String,
    results: BTreeMap<CandidateId, Result<Estimate, VisualError>>,
}
impl CandidateResults {
    /// Start a hypothesis set bound to exact observation content and calibration.
    pub fn new(frame: &Frame) -> Self {
        Self {
            evidence: frame.evidence_sha256(),
            results: BTreeMap::new(),
        }
    }
    /// Replace the latest result for a hypothesis without accumulating confidence.
    ///
    /// # Errors
    /// Rejects an estimate from different pixels, calibration or capture stamp.
    pub fn record(
        &mut self,
        id: CandidateId,
        result: Result<Estimate, VisualError>,
    ) -> Result<(), VisualError> {
        if result
            .as_ref()
            .is_ok_and(|estimate| estimate.observation_sha256 != self.evidence)
        {
            return Err(VisualError::Invalid {
                field: "candidate observation identity",
            });
        }
        self.results.insert(id, result);
        Ok(())
    }
    /// Results retain their individual map revision, backend and local covariance.
    pub fn iter(&self) -> impl Iterator<Item = (CandidateId, &Result<Estimate, VisualError>)> {
        self.results.iter().map(|(id, value)| (*id, value))
    }
    /// Select only when there is one geometrically accepted hypothesis.
    pub fn decision(&self) -> CandidateDecision {
        let accepted: Vec<_> = self
            .iter()
            .filter_map(|(id, result)| result.is_ok().then_some(id))
            .collect();
        match accepted.as_slice() {
            [] => CandidateDecision::Rejected,
            [id] => CandidateDecision::Unique(*id),
            _ => CandidateDecision::Unresolved(accepted),
        }
    }
}
