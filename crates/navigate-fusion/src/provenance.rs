//! Per-source ordering, the assessment window, and the provenance of one
//! filter epoch (ADR-0003, ADR-0004). [`crate::NavigationFilter`] and
//! [`crate::InertialNavigationFilter`] keep one ledger each.

use std::collections::HashMap;

use navigate_contract::{
    DurationNanos, MonotonicNanos, ObservationStamp, SourceComposition, SourceEpoch, SourceId,
    WrappingSequence,
};

use crate::rejection::RejectionReason;

/// Per-source admission tracking, valid within one source epoch.
#[derive(Debug, Clone, Copy)]
struct SourceTrack {
    epoch: SourceEpoch,
    sequence: WrappingSequence,
    acquired_at: MonotonicNanos,
}

/// One admitted observation's contribution to the assessment window.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AdmittedContribution {
    pub(crate) source: SourceId,
    pub(crate) composition: SourceComposition,
    admitted_at: MonotonicNanos,
}

/// The admission bookkeeping of one filter epoch.
#[derive(Debug, Clone)]
pub(crate) struct ProvenanceLedger {
    sources: HashMap<SourceId, SourceTrack>,
    admitted: Vec<AdmittedContribution>,
    // Union of every admitted composition this filter epoch: published
    // provenance describes everything the state ever absorbed, not just
    // the assessment window.
    epoch_composition: SourceComposition,
    pub(crate) last_admitted_at: Option<MonotonicNanos>,
}

impl Default for ProvenanceLedger {
    fn default() -> Self {
        Self {
            sources: HashMap::new(),
            admitted: Vec::new(),
            epoch_composition: SourceComposition::empty(),
            last_admitted_at: None,
        }
    }
}

impl ProvenanceLedger {
    /// Forgets all per-source tracking, the window, and the provenance.
    pub(crate) fn clear(&mut self) {
        self.sources.clear();
        self.admitted.clear();
        self.epoch_composition = SourceComposition::empty();
        self.last_admitted_at = None;
    }

    /// The union of every admitted composition in this filter epoch.
    pub(crate) const fn epoch_composition(&self) -> SourceComposition {
        self.epoch_composition
    }

    /// The admissions inside the assessment window.
    pub(crate) fn window(&self) -> &[AdmittedContribution] {
        &self.admitted
    }

    /// Time from the last admitted aiding observation to `now`. `None`
    /// before the first admission or when `now` is earlier.
    pub(crate) fn silence(&self, now: MonotonicNanos) -> Option<DurationNanos> {
        self.last_admitted_at.and_then(|at| now.elapsed_since(at))
    }

    /// The wrap-aware sequence and the acquisition time of a source must
    /// advance within one source epoch.
    pub(crate) fn check_source_ordering(
        &self,
        stamp: &ObservationStamp,
    ) -> Result<(), RejectionReason> {
        let Some(track) = self.sources.get(&stamp.source) else {
            // A source's first observation is always sequence-admissible.
            return Ok(());
        };
        if track.epoch != stamp.epoch {
            // Only a strictly advancing epoch resets the source's
            // sequence and time tracking; anything else replays a prior
            // incarnation whose stamps were already judged.
            if track.epoch.advances(stamp.epoch) {
                return Ok(());
            }
            return Err(RejectionReason::EpochRegression {
                last: track.epoch,
                got: stamp.epoch,
            });
        }
        if !track.sequence.admits(stamp.sequence) {
            return Err(RejectionReason::SequenceNotAdmitted {
                last: track.sequence,
                got: stamp.sequence,
            });
        }
        if stamp.acquired_at < track.acquired_at {
            return Err(RejectionReason::AcquisitionTimeRegression {
                last: track.acquired_at,
                got: stamp.acquired_at,
            });
        }
        Ok(())
    }

    /// Records an admitted aiding observation: the source track, the
    /// window entry, the provenance, and the silence reference.
    pub(crate) fn commit(
        &mut self,
        stamp: &ObservationStamp,
        composition: SourceComposition,
        now: MonotonicNanos,
        window: DurationNanos,
    ) {
        self.track(stamp, composition);
        self.admitted.push(AdmittedContribution {
            source: stamp.source,
            composition,
            admitted_at: now,
        });
        self.prune_window(now, window);
        self.last_admitted_at = Some(now);
    }

    /// Records an admitted sample of the propagation sensor. The sample
    /// enters the source track and the provenance. It does not enter the
    /// assessment window, and it does not reset the silence: a sensor
    /// that only propagates the state cannot cross-check it (ADR-0004).
    pub(crate) fn track(&mut self, stamp: &ObservationStamp, composition: SourceComposition) {
        self.sources.insert(
            stamp.source,
            SourceTrack {
                epoch: stamp.epoch,
                sequence: stamp.sequence,
                acquired_at: stamp.acquired_at,
            },
        );
        self.epoch_composition = self.epoch_composition.union(composition);
    }

    /// Records the composition of an explicit initialization at `at`. The
    /// initialization is an aiding fix, so it sets the silence reference.
    pub(crate) fn initialize(&mut self, composition: SourceComposition, at: MonotonicNanos) {
        self.epoch_composition = self.epoch_composition.union(composition);
        self.last_admitted_at = Some(at);
    }

    pub(crate) fn prune_window(&mut self, now: MonotonicNanos, window: DurationNanos) {
        self.admitted.retain(|c| {
            now.elapsed_since(c.admitted_at)
                .is_none_or(|age| age <= window)
        });
    }
}
