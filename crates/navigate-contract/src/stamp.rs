//! Identity and ordering stamps for observations and solutions.
//!
//! The discipline mirrors the sibling systems' ingress rules: a stamp
//! carries who produced a value, in which incarnation-epoch, at which
//! wrap-aware sequence position, and when on which clock. Republication
//! preserves the stamp; freshness never comes from arrival time.
//!
//! A stamp can also carry the same event as a [`PhysicalInstant`] on a
//! coordinate time scale (ADR-0015). [`RadiometricStamps`] carries the
//! events of one light-time measurement.

use crate::time::{ClockDomainId, MonotonicNanos, PhysicalInstant};

/// Identifies one observation source instance (one receiver, one camera
/// rig), stable within a source epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceId(u32);

impl SourceId {
    /// Wraps a raw source identifier.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Raw identifier value.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// A source's restart epoch: ordering of sequences is meaningful only
/// within one epoch, and a new epoch clears everything learned in the
/// previous one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SourceEpoch(u32);

impl SourceEpoch {
    /// Wraps a raw epoch value.
    #[must_use]
    pub const fn new(epoch: u32) -> Self {
        Self(epoch)
    }

    /// Raw epoch value.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The next epoch, wrapping.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0.wrapping_add(1))
    }

    /// Epoch-admission predicate: whether `candidate` is strictly ahead
    /// of `self` under serial arithmetic, ahead by at least one and by
    /// no more than half the counter range. `u32::MAX → 0` is an
    /// advance; a forward jump of `2^31 - 1` is admissible; exactly
    /// `2^31` is the ambiguous midpoint and refused.
    #[must_use]
    pub const fn advances(self, candidate: Self) -> bool {
        let delta = candidate.0.wrapping_sub(self.0);
        delta != 0 && delta <= (u32::MAX / 2)
    }
}

/// A wrap-aware monotonic sequence counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WrappingSequence(u32);

impl WrappingSequence {
    /// Wraps a raw sequence value.
    #[must_use]
    pub const fn new(value: u32) -> Self {
        Self(value)
    }

    /// Raw sequence value.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }

    /// The next sequence value, wrapping.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0.wrapping_add(1))
    }

    /// Whether `candidate` is strictly ahead of `self` under serial
    /// arithmetic: ahead by at least one and by no more than half the
    /// counter range. `u32::MAX → 0` is an advance; a forward jump of
    /// `2^31 - 1` is admissible; exactly `2^31` is the ambiguous
    /// midpoint and refused.
    #[must_use]
    pub const fn admits(self, candidate: Self) -> bool {
        let delta = candidate.0.wrapping_sub(self.0);
        delta != 0 && delta <= (u32::MAX / 2)
    }
}

/// Stamp carried by every observation entering fusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ObservationStamp {
    /// Producing source instance.
    pub source: SourceId,
    /// Source restart epoch.
    pub epoch: SourceEpoch,
    /// Wrap-aware per-source sequence.
    pub sequence: WrappingSequence,
    /// Monotonic acquisition time on `clock`.
    pub acquired_at: MonotonicNanos,
    /// Clock domain `acquired_at` was read from.
    pub clock: ClockDomainId,
    /// The acquisition event on a coordinate time scale, when the source
    /// knows it. It is absent by default.
    pub physical_at: Option<PhysicalInstant>,
}

impl ObservationStamp {
    /// Builds a stamp from its parts. [`Self::physical_at`] starts
    /// absent.
    #[must_use]
    pub const fn new(
        source: SourceId,
        epoch: SourceEpoch,
        sequence: WrappingSequence,
        acquired_at: MonotonicNanos,
        clock: ClockDomainId,
    ) -> Self {
        Self {
            source,
            epoch,
            sequence,
            acquired_at,
            clock,
            physical_at: None,
        }
    }

    /// This stamp with the acquisition event on a coordinate time scale.
    #[must_use]
    pub const fn with_physical_at(mut self, physical_at: PhysicalInstant) -> Self {
        self.physical_at = Some(physical_at);
        self
    }
}

/// Stamp carried by every published navigation solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SolutionStamp {
    /// Filter restart epoch: solutions across epochs never mix.
    pub epoch: SourceEpoch,
    /// Wrap-aware solution sequence within the epoch. It advances when
    /// the published solution changes. Two publications that are equal
    /// in every other field, such as two reads of one state at one time,
    /// have the same sequence.
    pub sequence: WrappingSequence,
    /// Monotonic time the solution state refers to, on `clock`.
    pub solved_at: MonotonicNanos,
    /// Clock domain `solved_at` was read from.
    pub clock: ClockDomainId,
    /// The solution time on a coordinate time scale, when the producer
    /// knows it. It is absent by default.
    pub physical_at: Option<PhysicalInstant>,
}

impl SolutionStamp {
    /// Builds a stamp from its parts. [`Self::physical_at`] starts
    /// absent.
    #[must_use]
    pub const fn new(
        epoch: SourceEpoch,
        sequence: WrappingSequence,
        solved_at: MonotonicNanos,
        clock: ClockDomainId,
    ) -> Self {
        Self {
            epoch,
            sequence,
            solved_at,
            clock,
            physical_at: None,
        }
    }

    /// This stamp with the solution time on a coordinate time scale.
    #[must_use]
    pub const fn with_physical_at(mut self, physical_at: PhysicalInstant) -> Self {
        self.physical_at = Some(physical_at);
        self
    }
}

/// The events of one light-time measurement on coordinate time scales.
///
/// A one-way measurement has a transmission and a reception. A two-way
/// measurement also has a turnaround at the transponder. A three-way
/// measurement has the form of a two-way measurement with a receiver
/// that is not the transmitter. Each event keeps its own scale, so a
/// ground transmission on TAI and a spacecraft turnaround on TDB do not
/// mix without an explicit conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct RadiometricStamps {
    /// The transmission event, when the measurement states it.
    pub transmitted_at: Option<PhysicalInstant>,
    /// The turnaround event of a two-way or three-way measurement.
    pub turnaround_at: Option<PhysicalInstant>,
    /// The reception event.
    pub received_at: PhysicalInstant,
}

impl RadiometricStamps {
    /// The events of a one-way measurement.
    #[must_use]
    pub const fn one_way(transmitted_at: PhysicalInstant, received_at: PhysicalInstant) -> Self {
        Self {
            transmitted_at: Some(transmitted_at),
            turnaround_at: None,
            received_at,
        }
    }

    /// The events of a two-way or three-way measurement.
    #[must_use]
    pub const fn two_way(
        transmitted_at: PhysicalInstant,
        turnaround_at: PhysicalInstant,
        received_at: PhysicalInstant,
    ) -> Self {
        Self {
            transmitted_at: Some(transmitted_at),
            turnaround_at: Some(turnaround_at),
            received_at,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ObservationStamp, RadiometricStamps, SolutionStamp, SourceEpoch, SourceId, WrappingSequence,
    };
    use crate::time::{ClockDomainId, MonotonicNanos, PhysicalInstant, TimeScale};

    #[test]
    fn a_stamp_carries_no_physical_instant_until_one_is_attached() {
        let at = PhysicalInstant::from_j2000(TimeScale::Tai, 10, 0).expect("in range");
        let observation = ObservationStamp::new(
            SourceId::new(1),
            SourceEpoch::new(0),
            WrappingSequence::new(0),
            MonotonicNanos::from_nanos(5),
            ClockDomainId::new(0),
        );
        assert_eq!(observation.physical_at, None);
        assert_eq!(observation.with_physical_at(at).physical_at, Some(at));
        let solution = SolutionStamp::new(
            SourceEpoch::new(0),
            WrappingSequence::new(0),
            MonotonicNanos::from_nanos(5),
            ClockDomainId::new(0),
        );
        assert_eq!(solution.physical_at, None);
        assert_eq!(solution.with_physical_at(at).physical_at, Some(at));
    }

    #[test]
    fn radiometric_stamps_name_each_light_time_event() {
        let up = PhysicalInstant::from_j2000(TimeScale::Tai, 0, 0).expect("in range");
        let turn = PhysicalInstant::from_j2000(TimeScale::Tdb, 1, 0).expect("in range");
        let down = PhysicalInstant::from_j2000(TimeScale::Tai, 2, 0).expect("in range");
        let one_way = RadiometricStamps::one_way(up, down);
        assert_eq!(one_way.transmitted_at, Some(up));
        assert_eq!(one_way.turnaround_at, None);
        assert_eq!(one_way.received_at, down);
        let two_way = RadiometricStamps::two_way(up, turn, down);
        assert_eq!(two_way.turnaround_at, Some(turn));
        assert_eq!(
            two_way.received_at.elapsed_since(up).map(|d| d.seconds()),
            Some(2)
        );
    }

    #[test]
    fn sequence_wrap_is_an_advance() {
        let last = WrappingSequence::new(u32::MAX);
        assert!(last.admits(WrappingSequence::new(0)));
    }

    #[test]
    fn duplicate_and_regression_are_refused() {
        let seq = WrappingSequence::new(7);
        assert!(!seq.admits(WrappingSequence::new(7)));
        assert!(!seq.admits(WrappingSequence::new(6)));
    }

    #[test]
    fn half_range_jump_is_ambiguous_and_refused() {
        let seq = WrappingSequence::new(0);
        assert!(seq.admits(WrappingSequence::new(u32::MAX / 2)));
        assert!(!seq.admits(WrappingSequence::new(u32::MAX / 2 + 1)));
    }

    #[test]
    fn epoch_wrap_is_an_advance() {
        let last = SourceEpoch::new(u32::MAX);
        assert!(last.advances(SourceEpoch::new(0)));
    }

    #[test]
    fn epoch_duplicate_and_regression_are_refused() {
        let epoch = SourceEpoch::new(7);
        assert!(!epoch.advances(SourceEpoch::new(7)));
        assert!(!epoch.advances(SourceEpoch::new(6)));
    }

    #[test]
    fn epoch_half_range_jump_is_ambiguous_and_refused() {
        let epoch = SourceEpoch::new(0);
        assert!(epoch.advances(SourceEpoch::new(u32::MAX / 2)));
        assert!(!epoch.advances(SourceEpoch::new(u32::MAX / 2 + 1)));
    }
}
