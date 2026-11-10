//! Per-source ordering, the assessment window, and the provenance of one
//! filter epoch (ADR-0003, ADR-0004). [`crate::NavigationFilter`] and
//! [`crate::InertialNavigationFilter`] keep one ledger each.
//!
//! The ledger has a fixed inline capacity (ADR-0016). The source table
//! holds [`MAX_SOURCES`] sources, and the filter refuses the first
//! observation of one more source. Without a track, the sequence and
//! time checks of that source would not run. The assessment window holds
//! [`MAX_WINDOW_ADMISSIONS`] admissions. A full window drops its oldest
//! admission, so the window can only show less redundancy, never more.

use navigate_contract::{
    DurationNanos, MonotonicNanos, ObservationStamp, SourceComposition, SourceEpoch, SourceId,
    WrappingSequence,
};

use crate::rejection::RejectionReason;

/// Largest number of sources that one filter epoch tracks.
pub const MAX_SOURCES: usize = 32;

/// Largest number of admissions in the assessment window.
pub const MAX_WINDOW_ADMISSIONS: usize = 128;

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

/// Bitmask of the state components that an observation directly constrains.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ConstrainedState(u8);

impl ConstrainedState {
    /// Vertical position (`[1]`).
    pub(crate) const VERTICAL_POSITION: Self = Self(1 << 1);
    /// Full 3D position (`Horizontal | Vertical`, `[0, 1]`).
    pub(crate) const POSITION: Self = Self((1 << 0) | (1 << 1));
    /// Translational velocity (`[2]`).
    pub(crate) const VELOCITY: Self = Self(1 << 2);
    /// Vehicle attitude (`[3]`).
    pub(crate) const ATTITUDE: Self = Self(1 << 3);
    /// Full 6-DoF pose (`Position | Attitude`, `[0, 1, 3]`).
    pub(crate) const POSE: Self = Self((1 << 0) | (1 << 1) | (1 << 3));

    const fn contains_bit(self, bit: usize) -> bool {
        (self.0 & (1_u8 << bit)) != 0
    }
}

/// The admission bookkeeping of one filter epoch.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ProvenanceLedger {
    sources: [Option<(SourceId, SourceTrack)>; MAX_SOURCES],
    admitted: [Option<AdmittedContribution>; MAX_WINDOW_ADMISSIONS],
    // Union of every admitted composition this filter epoch: published
    // provenance describes everything the state ever absorbed, not just
    // the assessment window.
    epoch_composition: SourceComposition,
    pub(crate) last_admitted_at: Option<MonotonicNanos>,
    // Per-component innovation-gate refusals since the last admitted
    // observation that constrained that state component (`[horizontal_pos,
    // vertical_pos, velocity, attitude]`).
    gate_rejection_streaks: [u32; 4],
}

impl Default for ProvenanceLedger {
    fn default() -> Self {
        Self {
            sources: [None; MAX_SOURCES],
            admitted: [None; MAX_WINDOW_ADMISSIONS],
            epoch_composition: SourceComposition::empty(),
            last_admitted_at: None,
            gate_rejection_streaks: [0; 4],
        }
    }
}

impl ProvenanceLedger {
    /// Forgets all per-source tracking, the window, and the provenance.
    pub(crate) fn clear(&mut self) {
        self.sources = [None; MAX_SOURCES];
        self.admitted = [None; MAX_WINDOW_ADMISSIONS];
        self.epoch_composition = SourceComposition::empty();
        self.last_admitted_at = None;
        self.gate_rejection_streaks = [0; 4];
    }

    /// The largest consecutive innovation-gate refusal streak across
    /// constrained state components in this filter epoch.
    pub(crate) const fn gate_rejection_streak(&self) -> u32 {
        const fn max_u32(a: u32, b: u32) -> u32 {
            if a >= b { a } else { b }
        }
        let [h, v, vel, att] = self.gate_rejection_streaks;
        max_u32(max_u32(h, v), max_u32(vel, att))
    }

    /// Records one innovation-gate refusal on the state components
    /// constrained by the refused observation. Each streak saturates: a
    /// wrap to 0 would report the longest storm as no storm.
    pub(crate) fn record_gate_rejection(&mut self, constrained: ConstrainedState) {
        for (bit, streak) in self.gate_rejection_streaks.iter_mut().enumerate() {
            if constrained.contains_bit(bit) {
                *streak = streak.saturating_add(1);
            }
        }
    }

    /// The union of every admitted composition in this filter epoch.
    pub(crate) const fn epoch_composition(&self) -> SourceComposition {
        self.epoch_composition
    }

    /// The admissions inside the assessment window, in no particular
    /// sequence.
    pub(crate) fn window(&self) -> impl Iterator<Item = &AdmittedContribution> + Clone {
        self.admitted.iter().flatten()
    }

    /// The track of `source`.
    fn track_of(&self, source: SourceId) -> Option<&SourceTrack> {
        self.sources
            .iter()
            .flatten()
            .find_map(|(id, track)| (*id == source).then_some(track))
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
        let Some(track) = self.track_of(stamp.source) else {
            // A source's first observation is always sequence-admissible
            // when the table has room to track it.
            if self.sources.iter().all(Option::is_some) {
                return Err(RejectionReason::SourceCapacityExceeded {
                    capacity: MAX_SOURCES,
                });
            }
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
    /// window entry, the provenance, the silence reference, and recovery
    /// of the state components that this observation constrains.
    pub(crate) fn commit(
        &mut self,
        stamp: &ObservationStamp,
        composition: SourceComposition,
        constrained: ConstrainedState,
        now: MonotonicNanos,
        window: DurationNanos,
    ) {
        self.track(stamp, composition);
        self.prune_window(now, window);
        let entry = AdmittedContribution {
            source: stamp.source,
            composition,
            admitted_at: now,
        };
        let slot = match self.admitted.iter().position(Option::is_none) {
            Some(free) => free,
            // A full window drops its oldest admission.
            None => oldest_admission(&self.admitted),
        };
        if let Some(target) = self.admitted.get_mut(slot) {
            *target = Some(entry);
        }
        self.last_admitted_at = Some(now);
        for (bit, streak) in self.gate_rejection_streaks.iter_mut().enumerate() {
            if constrained.contains_bit(bit) {
                *streak = 0;
            }
        }
    }

    /// Records an admitted sample of the propagation sensor. The sample
    /// enters the source track and the provenance. It does not enter the
    /// assessment window, and it does not reset the silence: a sensor
    /// that only propagates the state cannot cross-check it (ADR-0004).
    pub(crate) fn track(&mut self, stamp: &ObservationStamp, composition: SourceComposition) {
        let track = SourceTrack {
            epoch: stamp.epoch,
            sequence: stamp.sequence,
            acquired_at: stamp.acquired_at,
        };
        let existing = self
            .sources
            .iter()
            .position(|entry| entry.is_some_and(|(id, _)| id == stamp.source));
        // `check_source_ordering` refused a new source when the table is
        // full, so a new source always finds a free slot here.
        let slot = existing.or_else(|| self.sources.iter().position(Option::is_none));
        if let Some(entry) = slot.and_then(|index| self.sources.get_mut(index)) {
            *entry = Some((stamp.source, track));
        }
        self.epoch_composition = self.epoch_composition.union(composition);
    }

    /// Records the composition of an explicit initialization at `at`. The
    /// initialization is an aiding fix, so it sets the silence reference.
    pub(crate) fn initialize(&mut self, composition: SourceComposition, at: MonotonicNanos) {
        self.epoch_composition = self.epoch_composition.union(composition);
        self.last_admitted_at = Some(at);
        self.gate_rejection_streaks = [0; 4];
    }

    pub(crate) fn prune_window(&mut self, now: MonotonicNanos, window: DurationNanos) {
        for slot in &mut self.admitted {
            let expired = slot.is_some_and(|c| {
                now.elapsed_since(c.admitted_at)
                    .is_some_and(|age| age > window)
            });
            if expired {
                *slot = None;
            }
        }
    }
}

/// The slot of the admission with the earliest admission time.
fn oldest_admission(admitted: &[Option<AdmittedContribution>]) -> usize {
    admitted
        .iter()
        .enumerate()
        .filter_map(|(index, slot)| slot.map(|c| (index, c.admitted_at)))
        .min_by_key(|(_, at)| *at)
        .map_or(0, |(index, _)| index)
}

#[cfg(test)]
mod tests;
