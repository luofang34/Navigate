//! Identity and ordering stamps for observations and solutions.
//!
//! The discipline mirrors the sibling systems' ingress rules: a stamp
//! carries who produced a value, in which incarnation-epoch, at which
//! wrap-aware sequence position, and when on which clock. Republication
//! preserves the stamp; freshness never comes from arrival time.

use crate::time::{ClockDomainId, MonotonicNanos};

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
}

impl ObservationStamp {
    /// Builds a stamp from its parts.
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
        }
    }
}

/// Stamp carried by every published navigation solution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct SolutionStamp {
    /// Filter restart epoch: solutions across epochs never mix.
    pub epoch: SourceEpoch,
    /// Wrap-aware solution sequence within the epoch.
    pub sequence: WrappingSequence,
    /// Monotonic time the solution state refers to, on `clock`.
    pub solved_at: MonotonicNanos,
    /// Clock domain `solved_at` was read from.
    pub clock: ClockDomainId,
}

impl SolutionStamp {
    /// Builds a stamp from its parts.
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
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::{SourceEpoch, WrappingSequence};

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
