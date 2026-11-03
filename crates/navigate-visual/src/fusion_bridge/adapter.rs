//! Conversion of one visual estimate into one fusion observation.

use navigate_contract::{ClockDomainId, SourceEpoch, SourceId};
use navigate_fusion::Observation;
use std::collections::{HashSet, VecDeque};

use crate::fusion_bridge::{
    VisualAttitudeBudget, VisualErrorBudget, VisualFusionError, position_fix_observation,
    visual_pose_observation,
};
use crate::{Estimate, MapRevision};

/// Default maximum count of frame digests retained in the duplicate screen.
pub const DEFAULT_MAX_RETAINED_DIGESTS: usize = 16_384;

/// Host-assigned identity of one camera capture stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisualSourceIdentity {
    /// Source instance of the camera stream.
    pub source: SourceId,
    /// Restart epoch of the camera stream.
    pub epoch: SourceEpoch,
    /// Clock domain of the frame capture times.
    pub clock: ClockDomainId,
}

/// One fusion observation with the map release and the frame evidence it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct VisualFix {
    /// A position fix for [`navigate_fusion::NavigationFilter::ingest`], or a
    /// visual pose for [`navigate_fusion::InertialNavigationFilter::ingest`].
    pub observation: Observation,
    /// Map release of the reference that located the frame.
    pub map: MapRevision,
    /// Evidence digest of the frame.
    pub observation_sha256: String,
}

/// Host assessment of shared errors between this fix and the filter state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvidenceIndependence {
    /// Shared image, map, calibration, or prior error has not been excluded.
    Unknown,
    /// The host has validated independence from all evidence in the filter state.
    /// This includes residual map and calibration error across earlier fixes.
    ValidatedIndependent,
}

impl EvidenceIndependence {
    /// Restrict a host assessment with the label of a visual session anchor.
    ///
    /// A session label can refuse independence but cannot give it. The label
    /// covers only the map cells of one session ledger. The host assessment
    /// must cover all other shared error, for example calibration, earlier
    /// sessions, other map revisions, and other sources in the filter. Anchors
    /// that share a map cell with an earlier anchor, and anchors that the
    /// ledger could not track, become `Unknown`.
    pub fn restricted_by(self, label: &crate::session::FusionEligibility) -> Self {
        match label {
            crate::session::FusionEligibility::Independent => self,
            crate::session::FusionEligibility::SharedMapError { .. }
            | crate::session::FusionEligibility::Untracked => Self::Unknown,
        }
    }
}

/// Converts the visual estimates of one capture stream into position fixes or visual poses.
///
/// One frame gives at most one observation. A second estimate from the same frame
/// evidence is refused, because re-evaluation is not a new measurement.
/// Use a new source for a new stream epoch.
#[derive(Debug)]
pub struct VisualFixSource {
    identity: VisualSourceIdentity,
    budget: VisualErrorBudget,
    last: Option<u64>,
    admitted: HashSet<String>,
    admitted_order: VecDeque<String>,
    max_retained_digests: usize,
}

impl VisualFixSource {
    /// Declare a capture stream and the error budget of its map and camera.
    ///
    /// # Errors
    /// Refuses a budget with a term that is not finite and positive.
    pub fn new(
        identity: VisualSourceIdentity,
        budget: VisualErrorBudget,
    ) -> Result<Self, VisualFusionError> {
        Self::with_capacity(identity, budget, DEFAULT_MAX_RETAINED_DIGESTS)
    }

    /// Declare a capture stream with an explicit duplicate-digest window size.
    ///
    /// # Errors
    /// Refuses a zero `max_retained_digests` or a budget with a term that is
    /// not finite and positive.
    pub fn with_capacity(
        identity: VisualSourceIdentity,
        budget: VisualErrorBudget,
        max_retained_digests: usize,
    ) -> Result<Self, VisualFusionError> {
        if max_retained_digests == 0 {
            return Err(VisualFusionError::InvalidBudget {
                field: "max_retained_digests",
            });
        }
        budget.validate()?;
        Ok(Self {
            identity,
            budget,
            last: None,
            admitted: HashSet::new(),
            admitted_order: VecDeque::new(),
            max_retained_digests,
        })
    }

    /// Convert one accepted visual estimate into a fusion position fix.
    ///
    /// The fix declares the body-fixed frame of the reference ellipsoid of
    /// the visual frame, for example [`navigate_contract::FrameId::EARTH_ITRF93`]
    /// for an Earth frame (ADR-0013). See [`position_fix_observation`].
    ///
    /// # Errors
    /// Refuses unknown correlation, repeated frame evidence, frames out of capture order,
    /// positions outside geodetic range, and unusable covariances. A refusal
    /// does not change the state of the source.
    pub fn convert(
        &mut self,
        estimate: &Estimate,
        independence: EvidenceIndependence,
    ) -> Result<VisualFix, VisualFusionError> {
        self.admit(estimate, independence, |estimate, identity, budget| {
            position_fix_observation(estimate, identity, budget)
        })
    }

    /// Convert one accepted visual estimate into a six-degree-of-freedom visual pose
    /// for [`navigate_fusion::InertialNavigationFilter::ingest`] (ADR-0020).
    ///
    /// The pose and a position fix of one frame are the same evidence. The
    /// source gives at most one of them for each frame. See
    /// [`visual_pose_observation`].
    ///
    /// # Errors
    /// Refuses the conditions of [`Self::convert`] and an invalid attitude
    /// budget. A refusal does not change the state of the source.
    pub fn convert_pose(
        &mut self,
        estimate: &Estimate,
        independence: EvidenceIndependence,
        attitude_budget: &VisualAttitudeBudget,
    ) -> Result<VisualFix, VisualFusionError> {
        self.admit(estimate, independence, |estimate, identity, budget| {
            visual_pose_observation(estimate, identity, budget, attitude_budget)
        })
    }

    fn admit(
        &mut self,
        estimate: &Estimate,
        independence: EvidenceIndependence,
        build: impl FnOnce(
            &Estimate,
            &VisualSourceIdentity,
            &VisualErrorBudget,
        ) -> Result<Observation, VisualFusionError>,
    ) -> Result<VisualFix, VisualFusionError> {
        if independence != EvidenceIndependence::ValidatedIndependent {
            return Err(VisualFusionError::UnknownCorrelation);
        }
        self.check_order(estimate)?;
        let observation = build(estimate, &self.identity, &self.budget)?;
        self.last = Some(estimate.stamp.capture_time_ns);
        self.record_digest(estimate.observation_sha256.clone());
        Ok(VisualFix {
            observation,
            map: estimate.map.clone(),
            observation_sha256: estimate.observation_sha256.clone(),
        })
    }

    fn record_digest(&mut self, digest: String) {
        if self.admitted.len() >= self.max_retained_digests
            && let Some(oldest) = self.admitted_order.pop_front()
        {
            self.admitted.remove(&oldest);
        }
        self.admitted.insert(digest.clone());
        self.admitted_order.push_back(digest);
    }

    fn check_order(&self, estimate: &Estimate) -> Result<(), VisualFusionError> {
        if self.admitted.contains(&estimate.observation_sha256) {
            return Err(VisualFusionError::RepeatedEvidence {
                observation_sha256: estimate.observation_sha256.clone(),
            });
        }
        let Some(previous_ns) = self.last else {
            return Ok(());
        };
        if estimate.stamp.capture_time_ns <= previous_ns {
            return Err(VisualFusionError::FrameOrder {
                previous_ns,
                received_ns: estimate.stamp.capture_time_ns,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
