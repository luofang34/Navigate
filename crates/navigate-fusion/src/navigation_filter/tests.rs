#![allow(clippy::expect_used, clippy::panic)]

mod publication;
mod range;
mod reserved;

use navigate_contract::{
    ClockDomainId, GeodeticPosition, MonotonicNanos, NedVelocity, ObservationStamp, SensorClass,
    SourceComposition, SourceEpoch, SourceId, SymmetricCov3, WrappingSequence,
};
use navigate_geodesy::{LocalTangentPlane, NedOffset};

use super::NavigationFilter;
use crate::config::FusionConfig;
use crate::observation::{Observation, ObservationValue};
use crate::rejection::{IngestOutcome, RejectionReason};

const CLOCK: ClockDomainId = ClockDomainId::new(7);

fn t(ms: u64) -> MonotonicNanos {
    MonotonicNanos::from_nanos(ms.saturating_mul(1_000_000))
}

fn origin() -> GeodeticPosition {
    GeodeticPosition::new(47.0_f64.to_radians(), 8.0_f64.to_radians(), 500.0)
}

fn position_north_m(north_m: f64) -> GeodeticPosition {
    LocalTangentPlane::new(origin())
        .expect("plausible origin")
        .from_ned(&NedOffset::new(north_m, 0.0, 0.0))
}

fn stamp(source: u32, epoch: u32, sequence: u32, at_ms: u64) -> ObservationStamp {
    ObservationStamp::new(
        SourceId::new(source),
        SourceEpoch::new(epoch),
        WrappingSequence::new(sequence),
        t(at_ms),
        CLOCK,
    )
}

fn position_fix(
    stamp: ObservationStamp,
    position: GeodeticPosition,
    composition: SourceComposition,
) -> Observation {
    Observation::new(
        stamp,
        ObservationValue::PositionFix {
            position,
            covariance: SymmetricCov3::from_diagonal(25.0, 25.0, 25.0),
        },
        composition,
    )
}

fn gnss_fix(source: u32, sequence: u32, at_ms: u64, position: GeodeticPosition) -> Observation {
    position_fix(
        stamp(source, 1, sequence, at_ms),
        position,
        SourceComposition::of(SensorClass::Gnss),
    )
}

fn velocity_fix(source: u32, sequence: u32, at_ms: u64, velocity: NedVelocity) -> Observation {
    Observation::new(
        stamp(source, 1, sequence, at_ms),
        ObservationValue::VelocityFix {
            velocity,
            covariance: SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
        },
        SourceComposition::of(SensorClass::Gnss),
    )
}

fn zero_covariance_fix(sequence: u32, at_ms: u64) -> Observation {
    Observation::new(
        stamp(1, 1, sequence, at_ms),
        ObservationValue::PositionFix {
            position: origin(),
            covariance: SymmetricCov3::from_diagonal(0.0, 0.0, 0.0),
        },
        SourceComposition::of(SensorClass::Gnss),
    )
}

fn filter() -> NavigationFilter {
    NavigationFilter::new(FusionConfig::default(), CLOCK)
}

fn assert_accepted(outcome: IngestOutcome) {
    assert!(
        outcome.is_accepted(),
        "expected acceptance, got {outcome:?}"
    );
}

fn rejection(outcome: IngestOutcome) -> RejectionReason {
    match outcome {
        IngestOutcome::Rejected(reason) => reason,
        IngestOutcome::Accepted => panic!("expected rejection, got acceptance"),
    }
}

#[test]
fn kilometer_outlier_is_gated_and_state_untouched() {
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    let outlier = gnss_fix(1, 2, 100, position_north_m(1_000.0));
    let reason = rejection(f.ingest(&outlier, t(100)));
    assert!(
        matches!(reason, RejectionReason::InnovationGate { chi2, threshold } if chi2 > threshold)
    );
    assert_eq!(f.rejections().innovation_gate, 1);
    // The rejection changed nothing: the same sequence position is still
    // open, and the state is still at the origin.
    assert_accepted(f.ingest(&gnss_fix(1, 2, 200, origin()), t(200)));
    let solution = f.tick(t(200)).expect("initialized");
    assert!((solution.position.latitude_rad - origin().latitude_rad).abs() < 1e-7);
    assert!(solution.integrity.horizontal_1sigma_m < 10.0);
}

#[test]
fn observation_beyond_the_staleness_bound_is_rejected() {
    let mut f = filter();
    let reason = rejection(f.ingest(&gnss_fix(1, 1, 0, origin()), t(600)));
    assert!(matches!(reason, RejectionReason::Stale { .. }));
    assert_eq!(f.rejections().stale, 1);
    assert!(!f.is_initialized());
}

#[test]
fn duplicate_and_regressed_sequences_are_refused_within_an_epoch() {
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 5, 0, origin()), t(0)));
    assert!(matches!(
        rejection(f.ingest(&gnss_fix(1, 5, 100, origin()), t(100))),
        RejectionReason::SequenceNotAdmitted { .. }
    ));
    assert!(matches!(
        rejection(f.ingest(&gnss_fix(1, 4, 200, origin()), t(200))),
        RejectionReason::SequenceNotAdmitted { .. }
    ));
    assert_eq!(f.rejections().sequence_not_admitted, 2);
}

#[test]
fn sequence_wrap_is_an_advance() {
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, u32::MAX, 0, origin()), t(0)));
    assert_accepted(f.ingest(&gnss_fix(1, 0, 100, origin()), t(100)));
    assert_eq!(f.rejections().sequence_not_admitted, 0);
}

#[test]
fn a_new_source_epoch_resets_sequence_tracking() {
    let mut f = filter();
    let gnss = SourceComposition::of(SensorClass::Gnss);
    assert_accepted(f.ingest(&position_fix(stamp(1, 1, 5, 100), origin(), gnss), t(100)));
    // The same sequence within epoch 1 would be a duplicate; an
    // advancing epoch makes it admissible again.
    assert_accepted(f.ingest(&position_fix(stamp(1, 2, 5, 150), origin(), gnss), t(150)));
    assert_eq!(f.rejections().sequence_not_admitted, 0);
    assert_eq!(f.rejections().epoch_regression, 0);
}

#[test]
fn epoch_regression_is_rejected_and_state_untouched() {
    let mut f = filter();
    let gnss = SourceComposition::of(SensorClass::Gnss);
    assert_accepted(f.ingest(&position_fix(stamp(1, 2, 1, 0), origin(), gnss), t(0)));
    let before = f.tick(t(50)).expect("initialized");
    let reason = rejection(f.ingest(&position_fix(stamp(1, 1, 9, 50), origin(), gnss), t(50)));
    assert!(matches!(
        reason,
        RejectionReason::EpochRegression { last, got }
            if last == SourceEpoch::new(2) && got == SourceEpoch::new(1)
    ));
    assert_eq!(f.rejections().epoch_regression, 1);
    let after = f.tick(t(50)).expect("initialized");
    assert!(
        (after.integrity.horizontal_1sigma_m - before.integrity.horizontal_1sigma_m).abs() < 1e-12
    );
}

#[test]
fn replayed_prior_epoch_observation_is_rejected() {
    let mut f = filter();
    let gnss = SourceComposition::of(SensorClass::Gnss);
    let epoch_one = position_fix(stamp(1, 1, 5, 0), origin(), gnss);
    assert_accepted(f.ingest(&epoch_one, t(0)));
    assert_accepted(f.ingest(&position_fix(stamp(1, 2, 1, 50), origin(), gnss), t(50)));
    // A byte-identical replay of the epoch-1 observation must not
    // flip-flop the tracked epoch back and re-enter the state.
    assert!(matches!(
        rejection(f.ingest(&epoch_one, t(100))),
        RejectionReason::EpochRegression { .. }
    ));
    assert_eq!(f.rejections().epoch_regression, 1);
}

#[test]
fn undeclared_or_estimator_derived_composition_is_inadmissible() {
    let mut f = filter();
    let empty = position_fix(stamp(1, 1, 1, 0), origin(), SourceComposition::empty());
    assert!(matches!(
        rejection(f.ingest(&empty, t(0))),
        RejectionReason::EmptyComposition
    ));
    let circular = position_fix(
        stamp(1, 1, 2, 10),
        origin(),
        SourceComposition::of(SensorClass::FcState),
    );
    assert!(matches!(
        rejection(f.ingest(&circular, t(10))),
        RejectionReason::EstimatorDerived
    ));
    assert_eq!(f.rejections().empty_composition, 1);
    assert_eq!(f.rejections().estimator_derived, 1);
}

#[test]
fn foreign_clock_domain_is_rejected() {
    let mut f = filter();
    let foreign = ObservationStamp::new(
        SourceId::new(1),
        SourceEpoch::new(1),
        WrappingSequence::new(1),
        t(0),
        ClockDomainId::new(8),
    );
    let obs = position_fix(foreign, origin(), SourceComposition::of(SensorClass::Gnss));
    let reason = rejection(f.ingest(&obs, t(0)));
    assert!(matches!(
        reason,
        RejectionReason::ClockDomainMismatch { expected, got }
            if expected == CLOCK && got == ClockDomainId::new(8)
    ));
    assert_eq!(f.rejections().clock_domain_mismatch, 1);
}

#[test]
fn non_finite_values_and_implausible_covariances_are_rejected() {
    let mut f = filter();
    let nan_position = GeodeticPosition::new(f64::NAN, 0.0, 0.0);
    let obs = position_fix(
        stamp(1, 1, 1, 0),
        nan_position,
        SourceComposition::of(SensorClass::Gnss),
    );
    assert!(matches!(
        rejection(f.ingest(&obs, t(0))),
        RejectionReason::NonFiniteValue
    ));
    // A correlation exceeding the variances it links cannot come from a
    // real measurement: xy² > xx·yy fails positive definiteness.
    let overcorrelated = SymmetricCov3::from_upper_triangle([25.0, 100.0, 0.0, 25.0, 0.0, 25.0]);
    let obs = Observation::new(
        stamp(1, 1, 2, 10),
        ObservationValue::PositionFix {
            position: origin(),
            covariance: overcorrelated,
        },
        SourceComposition::of(SensorClass::Gnss),
    );
    assert!(matches!(
        rejection(f.ingest(&obs, t(10))),
        RejectionReason::ImplausibleCovariance
    ));
    assert_eq!(f.rejections().non_finite_value, 1);
    assert_eq!(f.rejections().implausible_covariance, 1);
}

#[test]
fn zero_covariance_is_rejected_and_cannot_collapse_sigma() {
    let mut f = filter();
    assert!(matches!(
        rejection(f.ingest(&zero_covariance_fix(1, 0), t(0))),
        RejectionReason::ImplausibleCovariance
    ));
    assert!(!f.is_initialized());
    // A properly uncertain fix initializes; the zero-covariance claim of
    // exact certainty is refused again afterward, so no single
    // observation can drive the published sigma to exactly zero.
    assert_accepted(f.ingest(&gnss_fix(1, 2, 10, origin()), t(10)));
    assert!(matches!(
        rejection(f.ingest(&zero_covariance_fix(3, 20), t(20))),
        RejectionReason::ImplausibleCovariance
    ));
    let solution = f.tick(t(20)).expect("initialized");
    assert!(solution.integrity.horizontal_1sigma_m > 0.0);
    assert!(solution.integrity.vertical_1sigma_m > 0.0);
    assert_eq!(f.rejections().implausible_covariance, 2);
}

#[test]
fn velocity_fix_before_any_position_fix_is_not_initialized() {
    let mut f = filter();
    let reason = rejection(f.ingest(
        &velocity_fix(1, 1, 0, NedVelocity::new(1.0, 0.0, 0.0)),
        t(0),
    ));
    assert!(matches!(reason, RejectionReason::NotInitialized));
    assert_eq!(f.rejections().not_initialized, 1);
}

#[test]
fn acquisition_time_never_runs_backward() {
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 100, origin()), t(100)));
    // Earlier than the source's admitted acquisition time.
    assert!(matches!(
        rejection(f.ingest(&gnss_fix(1, 2, 50, origin()), t(150))),
        RejectionReason::AcquisitionTimeRegression { .. }
    ));
    // Ahead of the admission reference `now`.
    assert!(matches!(
        rejection(f.ingest(&gnss_fix(1, 2, 300, origin()), t(200))),
        RejectionReason::AcquisitionTimeRegression { .. }
    ));
    assert_eq!(f.rejections().acquisition_time_regression, 2);
}
