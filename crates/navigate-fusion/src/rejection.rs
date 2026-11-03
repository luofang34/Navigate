//! Ingest outcomes, typed rejection reasons, and per-reason counters.
//!
//! Every admission-gate refusal is counted (ADR-0003): the counters are
//! first-class observability, part of the filter's public state rather
//! than a logging concern.

use navigate_contract::{
    ClockDomainId, DurationNanos, FrameId, MonotonicNanos, SourceEpoch, WrappingSequence,
};

use crate::observation::MeasurementKind;
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
    /// The observation declares a frame that is not the frame of the
    /// filter ellipsoid (ADR-0013). Two frames with the same center and
    /// different orientations, such as `MOON_ME` and `MOON_PA`, do not
    /// agree.
    #[error(
        "observation frame {}/{} is not the filter frame {}/{}",
        got.center_naif_id,
        got.orientation_naif_id,
        expected.center_naif_id,
        expected.orientation_naif_id
    )]
    FrameMismatch {
        /// The frame of the filter ellipsoid.
        expected: FrameId,
        /// The frame that the observation declares.
        got: FrameId,
    },
    /// The observation declares no frame, and the filter requires an
    /// explicit frame (ADR-0013).
    #[error(
        "observation declares no frame; the filter requires frame {}/{}",
        expected.center_naif_id,
        expected.orientation_naif_id
    )]
    MissingFrame {
        /// The frame of the filter ellipsoid.
        expected: FrameId,
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
    /// A measurement that needs a state arrived before any position fix
    /// anchored the filter's origin.
    #[error("filter is not initialized: a position fix must be admitted first")]
    NotInitialized,
    /// The innovation chi-square statistic exceeded the configured gate.
    #[error("innovation chi-square {chi2} exceeds the gate threshold {threshold}")]
    InnovationGate {
        /// Observed chi-square statistic. A vector fix has 3 degrees of
        /// freedom. A scalar measurement has 1 degree of freedom.
        chi2: f64,
        /// The configured gate threshold.
        threshold: f64,
    },
    /// The observation needs a measurement model that this build does not
    /// have. The value is reserved vocabulary (ADR-0008), not silently dropped.
    #[error("measurement model {kind:?} is not available in this build")]
    UnsupportedMeasurement {
        /// The measurement model the observation needs.
        kind: MeasurementKind,
    },
    /// The state can be outside the region where the map linearization is
    /// valid (ADR-0012). The filter adds the offset from the linearization
    /// point to `linearization_sigma_multiplier` horizontal 1-sigma
    /// values. The sum exceeds the valid radius of the model.
    #[error(
        "offset plus sigma margin {distance_plus_ksigma_m} m exceeds the valid radius {valid_radius_m} m"
    )]
    LinearizationRegionExceeded {
        /// Offset from the linearization point plus the sigma margin, in
        /// meters.
        distance_plus_ksigma_m: f64,
        /// Valid radius of the linearized map model, in meters.
        valid_radius_m: f64,
    },
    /// The map gradient gives too little position information
    /// (ADR-0012). The information ratio `gᵀ P_pos g / R` is below the
    /// configured floor, or it is not finite.
    #[error("gradient information ratio {information_ratio} is below the floor {floor}")]
    UninformativeGradient {
        /// The information ratio. It has no unit.
        information_ratio: f64,
        /// The configured floor, `min_information_ratio`.
        floor: f64,
    },
    /// A delayed observation is older than the retained history
    /// (ADR-0012). The filter cannot apply it at its acquisition time.
    #[error(
        "delayed observation age {} ns is outside the history depth {} ns",
        age.as_nanos(),
        depth.as_nanos()
    )]
    OutsideHistory {
        /// Time from the acquisition time to the state time.
        age: DurationNanos,
        /// Time from the oldest retained rewind point to the state time.
        /// The oldest rewind point is the state before the oldest retained
        /// update. Without retained updates, it is the initialization
        /// time. The depth is zero when the history capacity is zero.
        depth: DurationNanos,
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
    /// Refusals for a declared frame that is not the filter frame.
    pub frame_mismatch: u64,
    /// Refusals for a missing frame when the filter requires one.
    pub missing_frame: u64,
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
    /// Refusals of measurements whose model this build does not have.
    pub unsupported_measurement: u64,
    /// Refusals of map measurements outside their linearization region.
    pub linearization_region_exceeded: u64,
    /// Refusals of map measurements with an uninformative gradient.
    pub uninformative_gradient: u64,
    /// Refusals of delayed observations older than the retained history.
    pub outside_history: u64,
}

impl RejectionCounters {
    pub(crate) fn record(&mut self, reason: &RejectionReason) {
        let counter = match reason {
            RejectionReason::ClockDomainMismatch { .. } => &mut self.clock_domain_mismatch,
            RejectionReason::FrameMismatch { .. } => &mut self.frame_mismatch,
            RejectionReason::MissingFrame { .. } => &mut self.missing_frame,
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
            RejectionReason::UnsupportedMeasurement { .. } => &mut self.unsupported_measurement,
            RejectionReason::LinearizationRegionExceeded { .. } => {
                &mut self.linearization_region_exceeded
            }
            RejectionReason::UninformativeGradient { .. } => &mut self.uninformative_gradient,
            RejectionReason::OutsideHistory { .. } => &mut self.outside_history,
        };
        *counter = counter.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use navigate_contract::{DurationNanos, FrameId};

    use super::{RejectionCounters, RejectionReason};

    #[test]
    fn frame_refusals_each_increment_one_counter_and_wrap() {
        let mut counters = RejectionCounters {
            missing_frame: u64::MAX,
            ..RejectionCounters::default()
        };
        counters.record(&RejectionReason::FrameMismatch {
            expected: FrameId::MOON_ME,
            got: FrameId::MOON_PA,
        });
        counters.record(&RejectionReason::MissingFrame {
            expected: FrameId::MOON_ME,
        });
        assert_eq!(
            counters,
            RejectionCounters {
                frame_mismatch: 1,
                missing_frame: 0,
                ..RejectionCounters::default()
            }
        );
    }

    #[test]
    fn frame_refusals_name_both_naif_codes() {
        let text = RejectionReason::FrameMismatch {
            expected: FrameId::MOON_ME,
            got: FrameId::MOON_PA,
        }
        .to_string();
        assert!(text.contains("301/31000"), "{text}");
        assert!(text.contains("301/31001"), "{text}");
        let missing = RejectionReason::MissingFrame {
            expected: FrameId::MARS_IAU,
        }
        .to_string();
        assert!(missing.contains("499/10014"), "{missing}");
    }

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

    #[test]
    fn map_and_history_refusals_each_increment_one_counter() {
        let mut counters = RejectionCounters::default();
        counters.record(&RejectionReason::LinearizationRegionExceeded {
            distance_plus_ksigma_m: 600.0,
            valid_radius_m: 500.0,
        });
        counters.record(&RejectionReason::UninformativeGradient {
            information_ratio: 1e-6,
            floor: 1e-3,
        });
        counters.record(&RejectionReason::UninformativeGradient {
            information_ratio: f64::NAN,
            floor: 1e-3,
        });
        counters.record(&RejectionReason::OutsideHistory {
            age: DurationNanos::from_millis(300),
            depth: DurationNanos::from_millis(100),
        });
        assert_eq!(
            counters,
            RejectionCounters {
                linearization_region_exceeded: 1,
                uninformative_gradient: 2,
                outside_history: 1,
                ..RejectionCounters::default()
            }
        );
    }

    #[test]
    fn new_counters_wrap_instead_of_overflowing() {
        let mut counters = RejectionCounters {
            linearization_region_exceeded: u64::MAX,
            uninformative_gradient: u64::MAX,
            outside_history: u64::MAX,
            ..RejectionCounters::default()
        };
        counters.record(&RejectionReason::LinearizationRegionExceeded {
            distance_plus_ksigma_m: 1.0,
            valid_radius_m: 0.5,
        });
        counters.record(&RejectionReason::UninformativeGradient {
            information_ratio: 0.0,
            floor: 1e-3,
        });
        counters.record(&RejectionReason::OutsideHistory {
            age: DurationNanos::from_nanos(1),
            depth: DurationNanos::from_nanos(0),
        });
        assert_eq!(counters.linearization_region_exceeded, 0);
        assert_eq!(counters.uninformative_gradient, 0);
        assert_eq!(counters.outside_history, 0);
    }

    #[test]
    fn new_reasons_name_their_values() {
        let text = RejectionReason::OutsideHistory {
            age: DurationNanos::from_millis(300),
            depth: DurationNanos::from_millis(100),
        }
        .to_string();
        assert!(text.contains("300000000"), "{text}");
        assert!(text.contains("100000000"), "{text}");
    }
}
