//! Ingest outcomes, typed rejection reasons, and per-reason counters.
//!
//! Every admission-gate refusal is counted (ADR-0003): the counters are
//! first-class observability, part of the filter's public state rather
//! than a logging concern.

use navigate_contract::{
    ClockDomainId, DurationNanos, MonotonicNanos, SourceEpoch, WrappingSequence,
};
use thiserror::Error;

/// The result of offering one observation to the filter.
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use]
pub enum IngestOutcome {
    /// The observation passed every gate and updated the state.
    Accepted,
    /// The observation was refused; the state is unchanged.
    Rejected(RejectionReason),
}

impl IngestOutcome {
    /// Whether the observation was admitted.
    #[must_use]
    pub const fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted)
    }
}

/// Why an observation was refused admission.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum RejectionReason {
    /// The stamp's clock domain is not the filter's; their timestamps
    /// are not comparable.
    #[error(
        "observation clock domain {} is not the filter's domain {}",
        got.get(),
        expected.get()
    )]
    ClockDomainMismatch {
        /// The filter's clock domain.
        expected: ClockDomainId,
        /// The observation stamp's clock domain.
        got: ClockDomainId,
    },
    /// The observation declares no source composition; an empty
    /// declaration is inadmissible, never presumed independent.
    #[error("observation declares an empty source composition")]
    EmptyComposition,
    /// The composition marks the value as derived from an estimator
    /// output (FC state or a published solution) — circular as an input.
    #[error("observation composition is estimator-derived and would double-count")]
    EstimatorDerived,
    /// A value component is non-finite or outside geodetic range.
    #[error("observation value has a non-finite or out-of-range component")]
    NonFiniteValue,
    /// The covariance is not finite and positive definite, or the
    /// innovation covariance it produced could not be inverted. Merely
    /// semidefinite is refused: a zero-variance direction would let one
    /// observation collapse the published uncertainty to exactly zero.
    #[error("observation covariance is not finite and positive definite")]
    ImplausibleCovariance,
    /// The observation is older than the staleness bound at admission.
    #[error(
        "observation is stale: age {} ns exceeds the bound {} ns",
        age.as_nanos(),
        bound.as_nanos()
    )]
    Stale {
        /// Age of the observation at the admission reference time.
        age: DurationNanos,
        /// The configured staleness bound the age exceeds.
        bound: DurationNanos,
    },
    /// The source's epoch neither matches the tracked epoch nor advances
    /// past it — a replayed prior incarnation, whose stamps were already
    /// judged, is never re-admissible.
    #[error(
        "source epoch {} does not advance past tracked epoch {}",
        got.get(),
        last.get()
    )]
    EpochRegression {
        /// Last tracked epoch for the source.
        last: SourceEpoch,
        /// The refused candidate epoch.
        got: SourceEpoch,
    },
    /// The source's wrap-aware sequence does not admit the candidate
    /// (duplicate, regression, or ambiguous half-range jump).
    #[error("sequence {} is not admitted after {}", got.get(), last.get())]
    SequenceNotAdmitted {
        /// Last admitted sequence for the source in this epoch.
        last: WrappingSequence,
        /// The refused candidate sequence.
        got: WrappingSequence,
    },
    /// Acquisition time runs backward: earlier than the source's last
    /// admitted acquisition time, or ahead of the admission reference
    /// `now` (a claim of acquisition in the future).
    #[error(
        "acquisition time {} ns regresses against reference {} ns",
        got.as_nanos(),
        last.as_nanos()
    )]
    AcquisitionTimeRegression {
        /// The reference the candidate violates: the source's last
        /// admitted acquisition time, or `now` for a future claim.
        last: MonotonicNanos,
        /// The refused acquisition time.
        got: MonotonicNanos,
    },
    /// A velocity fix arrived before any position fix anchored the
    /// filter's origin.
    #[error("filter is not initialized: a position fix must be admitted first")]
    NotInitialized,
    /// The innovation chi-square statistic exceeded the configured gate.
    #[error("innovation chi-square {chi2} exceeds the gate threshold {threshold}")]
    InnovationGate {
        /// Observed chi-square statistic (3 degrees of freedom).
        chi2: f64,
        /// The configured gate threshold.
        threshold: f64,
    },
}

/// Per-reason rejection counters; every refusal increments exactly one.
///
/// Counters wrap rather than saturate and persist across
/// [`crate::NavigationFilter::reset`]: they describe the process
/// lifetime, not one filter epoch.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct RejectionCounters {
    /// Refusals for a foreign clock domain.
    pub clock_domain_mismatch: u64,
    /// Refusals for an empty source composition.
    pub empty_composition: u64,
    /// Refusals for an estimator-derived composition.
    pub estimator_derived: u64,
    /// Refusals for non-finite or out-of-range value components.
    pub non_finite_value: u64,
    /// Refusals for implausible covariances.
    pub implausible_covariance: u64,
    /// Refusals for staleness beyond the configured bound.
    pub stale: u64,
    /// Refusals for a source epoch that neither matches nor advances.
    pub epoch_regression: u64,
    /// Refusals for inadmissible sequences.
    pub sequence_not_admitted: u64,
    /// Refusals for acquisition-time regression.
    pub acquisition_time_regression: u64,
    /// Refusals of velocity fixes before initialization.
    pub not_initialized: u64,
    /// Refusals by the innovation gate.
    pub innovation_gate: u64,
}

impl RejectionCounters {
    pub(crate) fn record(&mut self, reason: &RejectionReason) {
        let counter = match reason {
            RejectionReason::ClockDomainMismatch { .. } => &mut self.clock_domain_mismatch,
            RejectionReason::EmptyComposition => &mut self.empty_composition,
            RejectionReason::EstimatorDerived => &mut self.estimator_derived,
            RejectionReason::NonFiniteValue => &mut self.non_finite_value,
            RejectionReason::ImplausibleCovariance => &mut self.implausible_covariance,
            RejectionReason::Stale { .. } => &mut self.stale,
            RejectionReason::EpochRegression { .. } => &mut self.epoch_regression,
            RejectionReason::SequenceNotAdmitted { .. } => &mut self.sequence_not_admitted,
            RejectionReason::AcquisitionTimeRegression { .. } => {
                &mut self.acquisition_time_regression
            }
            RejectionReason::NotInitialized => &mut self.not_initialized,
            RejectionReason::InnovationGate { .. } => &mut self.innovation_gate,
        };
        *counter = counter.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use navigate_contract::DurationNanos;

    use super::{RejectionCounters, RejectionReason};

    #[test]
    fn counters_wrap_instead_of_overflowing() {
        let mut counters = RejectionCounters {
            stale: u64::MAX,
            ..RejectionCounters::default()
        };
        counters.record(&RejectionReason::Stale {
            age: DurationNanos::from_nanos(1),
            bound: DurationNanos::from_nanos(0),
        });
        assert_eq!(counters.stale, 0);
        counters.record(&RejectionReason::NotInitialized);
        assert_eq!(counters.not_initialized, 1);
    }
}
