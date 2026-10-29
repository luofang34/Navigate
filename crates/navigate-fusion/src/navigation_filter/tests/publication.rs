//! Publication-side behavior: initialization, tick semantics, integrity
//! derivation, provenance, and determinism.

use navigate_contract::{
    FaultDetection, NavigationSolution, NedVelocity, Redundancy, SensorClass, SolutionQuality,
    SourceComposition, SourceEpoch, WrappingSequence,
};

use super::{
    CLOCK, assert_accepted, filter, gnss_fix, origin, position_fix, position_north_m, stamp, t,
    velocity_fix,
};
use crate::config::FusionConfig;
use crate::navigation_filter::NavigationFilter;

#[test]
fn no_solution_before_initialization() {
    let mut f = filter();
    assert!(f.tick(t(10)).is_none());
    assert!(!f.is_initialized());
}

#[test]
fn first_admitted_fix_initializes_at_the_origin() {
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    assert!(f.is_initialized());
    let solution = f.tick(t(0)).expect("initialized");
    assert!((solution.position.latitude_rad - origin().latitude_rad).abs() < 1e-9);
    assert!((solution.position.longitude_rad - origin().longitude_rad).abs() < 1e-9);
    assert!((solution.position.altitude_m - origin().altitude_m).abs() < 1e-3);
    assert!(solution.attitude.is_none());
    assert!(solution.integrity.protection_levels.is_none());
    assert!(matches!(solution.integrity.quality, SolutionQuality::Good));
}

#[test]
fn first_solution_uncertainty_is_the_initializing_fix_uncertainty() {
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    let solution = f.tick(t(0)).expect("initialized");
    // The helper fix carries diag(25) m² — 5 m 1-sigma. The published
    // uncertainty is the fix's own, not the fix folded into a diffuse
    // prior the fix itself centers.
    assert!((solution.integrity.horizontal_1sigma_m - 5.0).abs() < 1e-9);
    assert!((solution.integrity.vertical_1sigma_m - 5.0).abs() < 1e-9);
}

#[test]
fn second_fix_shrinks_horizontal_uncertainty() {
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    let first = f.tick(t(0)).expect("initialized");
    assert_accepted(f.ingest(&gnss_fix(1, 2, 100, origin()), t(100)));
    let second = f.tick(t(100)).expect("initialized");
    assert!(second.integrity.horizontal_1sigma_m < first.integrity.horizontal_1sigma_m);
}

#[test]
fn velocity_fix_after_initialization_steers_the_velocity_state() {
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    assert_accepted(f.ingest(
        &velocity_fix(1, 2, 50, NedVelocity::new(10.0, 0.0, 0.0)),
        t(50),
    ));
    let solution = f.tick(t(50)).expect("initialized");
    assert!(solution.velocity.north_mps > 9.0);
}

#[test]
fn single_source_reports_no_redundancy_and_unavailable_fault_detection() {
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    let solution = f.tick(t(0)).expect("initialized");
    assert!(matches!(solution.integrity.quality, SolutionQuality::Good));
    assert!(matches!(solution.integrity.redundancy, Redundancy::None));
    assert!(matches!(
        solution.integrity.fault_detection,
        FaultDetection::Unavailable
    ));
    assert!(solution.integrity.contributing.contains(SensorClass::Gnss));
    assert_eq!(solution.integrity.contributing.class_count(), 1);
}

#[test]
fn single_source_declaring_many_classes_is_still_not_redundant() {
    let mut f = filter();
    let multi = SourceComposition::of(SensorClass::Gnss).with(SensorClass::Barometric);
    assert_accepted(f.ingest(&position_fix(stamp(1, 1, 1, 0), origin(), multi), t(0)));
    let solution = f.tick(t(0)).expect("initialized");
    // One source corroborates nothing, however many classes it declares.
    assert!(matches!(solution.integrity.redundancy, Redundancy::None));
    assert!(matches!(
        solution.integrity.fault_detection,
        FaultDetection::Unavailable
    ));
    assert_eq!(solution.integrity.contributing.class_count(), 2);
}

#[test]
fn a_second_independent_class_enables_monitoring() {
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    let landmark = position_fix(
        stamp(2, 1, 1, 50),
        origin(),
        SourceComposition::of(SensorClass::VisualLandmark),
    );
    assert_accepted(f.ingest(&landmark, t(50)));
    let solution = f.tick(t(50)).expect("initialized");
    assert!(matches!(
        solution.integrity.redundancy,
        Redundancy::IndependentClasses
    ));
    assert!(matches!(
        solution.integrity.fault_detection,
        FaultDetection::Monitoring
    ));
    assert!(solution.composition.contains(SensorClass::Gnss));
    assert!(solution.composition.contains(SensorClass::VisualLandmark));
}

#[test]
fn two_sources_of_one_class_report_same_class_redundancy() {
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    assert_accepted(f.ingest(&gnss_fix(2, 1, 50, origin()), t(50)));
    let solution = f.tick(t(50)).expect("initialized");
    assert!(matches!(
        solution.integrity.redundancy,
        Redundancy::SameClass
    ));
    assert!(matches!(
        solution.integrity.fault_detection,
        FaultDetection::Monitoring
    ));
}

#[test]
fn published_composition_is_epoch_cumulative_not_window_scoped() {
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    // Past the assessment window the contributing set empties, but the
    // state still embodies the GNSS history it absorbed.
    let solution = f.tick(t(6_000)).expect("initialized");
    assert!(solution.integrity.contributing.is_empty());
    assert!(solution.composition.contains(SensorClass::Gnss));
    // Reset clears the cumulative provenance with everything else.
    f.reset();
    let landmark = position_fix(
        stamp(9, 1, 1, 6_050),
        origin(),
        SourceComposition::of(SensorClass::VisualLandmark),
    );
    assert_accepted(f.ingest(&landmark, t(6_050)));
    let fresh = f.tick(t(6_050)).expect("re-initialized");
    assert!(!fresh.composition.contains(SensorClass::Gnss));
    assert!(fresh.composition.contains(SensorClass::VisualLandmark));
}

#[test]
fn silence_degrades_quality_through_degraded_to_unusable() {
    let config = FusionConfig {
        // Uncertainty bounds far out of reach isolate the silence
        // horizons from covariance growth.
        good_horizontal_1sigma_m: 1.0e9,
        degraded_horizontal_1sigma_m: 2.0e9,
        good_vertical_1sigma_m: 1.0e9,
        degraded_vertical_1sigma_m: 2.0e9,
        ..FusionConfig::default()
    };
    let mut f = NavigationFilter::new(config, CLOCK);
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    let fresh = f.tick(t(1_000)).expect("initialized");
    assert!(matches!(fresh.integrity.quality, SolutionQuality::Good));
    let degraded = f.tick(t(2_500)).expect("initialized");
    assert!(matches!(
        degraded.integrity.quality,
        SolutionQuality::Degraded
    ));
    let unusable = f.tick(t(5_500)).expect("initialized");
    assert!(matches!(
        unusable.integrity.quality,
        SolutionQuality::Unusable
    ));
}

#[test]
fn a_measurement_after_an_intervening_tick_keeps_its_true_dt() {
    fn run(tick_between: bool) -> NavigationSolution {
        let mut f = filter();
        assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
        if tick_between {
            assert!(f.tick(t(150)).is_some());
        }
        assert_accepted(f.ingest(&gnss_fix(1, 2, 100, origin()), t(150)));
        f.tick(t(150)).expect("initialized")
    }
    let with_tick = run(true);
    let without_tick = run(false);
    // The intervening tick committed nothing: the measurement applied
    // with its true 100 ms dt in both runs.
    assert_eq!(with_tick.position, without_tick.position);
    assert_eq!(with_tick.position_cov, without_tick.position_cov);
    // And that dt is genuinely nonzero: a dt = 0 application of the same
    // fix yields a different covariance.
    let mut dt_zero = filter();
    assert_accepted(dt_zero.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    assert_accepted(dt_zero.ingest(&gnss_fix(1, 2, 0, origin()), t(0)));
    let zero = dt_zero.tick(t(150)).expect("initialized");
    assert!(
        (zero.integrity.horizontal_1sigma_m - with_tick.integrity.horizontal_1sigma_m).abs() > 1e-6
    );
}

#[test]
fn tick_earlier_than_the_state_time_is_clock_misuse_and_yields_none() {
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 100, origin()), t(100)));
    assert!(f.tick(t(50)).is_none());
    let solution = f.tick(t(150)).expect("initialized");
    assert_eq!(solution.stamp.solved_at, t(150));
}

#[test]
fn reset_starts_a_new_epoch_and_forgets_initialization() {
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    let before = f.tick(t(0)).expect("initialized");
    assert_eq!(before.stamp.epoch, SourceEpoch::new(0));
    assert_eq!(before.stamp.sequence, WrappingSequence::new(0));
    f.reset();
    assert!(f.tick(t(10)).is_none());
    assert_eq!(f.epoch(), SourceEpoch::new(1));
    // Per-source tracking is gone: the same stamp identity is admissible.
    assert_accepted(f.ingest(&gnss_fix(1, 1, 20, origin()), t(20)));
    let after = f.tick(t(20)).expect("re-initialized");
    assert_eq!(after.stamp.epoch, SourceEpoch::new(1));
    assert_eq!(after.stamp.sequence, WrappingSequence::new(0));
}

#[test]
fn solution_sequence_advances_per_publication() {
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    let a = f.tick(t(10)).expect("initialized");
    let b = f.tick(t(20)).expect("initialized");
    assert_eq!(a.stamp.sequence.next(), b.stamp.sequence);
    assert_eq!(a.stamp.solved_at, t(10));
    assert_eq!(b.stamp.solved_at, t(20));
}

#[test]
fn identical_observation_scripts_yield_field_identical_solutions() {
    fn run() -> Vec<Option<NavigationSolution>> {
        let mut f = filter();
        let mut published = Vec::new();
        assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
        published.push(f.tick(t(10)));
        assert_accepted(f.ingest(
            &velocity_fix(1, 2, 40, NedVelocity::new(3.0, -1.0, 0.5)),
            t(40),
        ));
        assert_accepted(f.ingest(&gnss_fix(2, 1, 80, position_north_m(5.0)), t(80)));
        published.push(f.tick(t(100)));
        published.push(f.tick(t(400)));
        published
    }
    assert_eq!(run(), run());
}

#[test]
fn vertical_uncertainty_degrades_quality_even_when_horizontal_is_tight() {
    use crate::observation::{Observation, ObservationValue};
    use navigate_contract::SymmetricCov3;

    let mut f = filter();
    let degraded_vertical = Observation::new(
        stamp(1, 1, 1, 0),
        ObservationValue::PositionFix {
            position: origin(),
            covariance: SymmetricCov3::from_diagonal(4.0, 4.0, 900.0),
        },
        SourceComposition::of(SensorClass::Gnss),
    );
    assert_accepted(f.ingest(&degraded_vertical, t(0)));
    let s1 = f.tick(t(0)).expect("initialized");
    assert!(s1.integrity.horizontal_1sigma_m <= 2.5);
    assert!((s1.integrity.vertical_1sigma_m - 30.0).abs() < 1e-9);
    assert!(matches!(s1.integrity.quality, SolutionQuality::Degraded));

    f.reset();
    let unusable_vertical = Observation::new(
        stamp(1, 1, 1, 0),
        ObservationValue::PositionFix {
            position: origin(),
            covariance: SymmetricCov3::from_diagonal(4.0, 4.0, 10_000.0),
        },
        SourceComposition::of(SensorClass::Gnss),
    );
    assert_accepted(f.ingest(&unusable_vertical, t(0)));
    let s2 = f.tick(t(0)).expect("initialized");
    assert!(matches!(s2.integrity.quality, SolutionQuality::Unusable));
}

#[test]
fn long_range_flight_reanchors_tangent_plane_without_losing_track() {
    use navigate_geodesy::{LocalTangentPlane, NedOffset};

    let config = FusionConfig {
        initial_velocity_variance_m2_per_s2: 10_000.0,
        reanchor_distance_m: 50_000.0,
        ..FusionConfig::default()
    };
    let mut f = NavigationFilter::new(config, CLOCK);
    let mut current = origin();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, current), t(0)));
    assert_accepted(f.ingest(
        &velocity_fix(1, 2, 0, NedVelocity::new(250.0, 0.0, 0.0)),
        t(0),
    ));

    // Fly 200 km north at 250 m/s in 1-second steps across 4 re-anchors.
    for step in 1_u32..=800 {
        let now_ms = u64::from(step) * 1_000;
        let local = LocalTangentPlane::new(current).expect("valid");
        current = local.from_ned(&NedOffset::new(250.0, 0.0, 0.0));
        let seq = step * 2 + 1;
        assert_accepted(f.ingest(&gnss_fix(1, seq, now_ms, current), t(now_ms)));
        assert_accepted(f.ingest(
            &velocity_fix(1, seq + 1, now_ms, NedVelocity::new(250.0, 0.0, 0.0)),
            t(now_ms),
        ));
    }
    let final_sol = f.tick(t(800_000)).expect("solution after 200 km");
    let residual = LocalTangentPlane::new(current)
        .expect("valid")
        .to_ned(&final_sol.position);
    assert!(
        residual.north_m.hypot(residual.east_m) < 5.0,
        "200 km horizontal error: {residual:?}"
    );
    assert!(
        residual.down_m.abs() < 5.0,
        "200 km vertical error: {residual:?}"
    );
    assert!(matches!(final_sol.integrity.quality, SolutionQuality::Good));
}
