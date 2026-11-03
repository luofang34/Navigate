//! Magnetic anomaly updates at survey scale in SI tesla.

use navigate_contract::{Redundancy, SolutionQuality};

use super::*;
use crate::observation::MeasurementKind;

/// 50 nT anomaly at the linearization point.
const MAP_T: f64 = 50e-9;
/// 3 nT 1-sigma.
const VARIANCE_T2: f64 = 9e-18;
/// 0.05 nT/m.
const SURVEY_GRADIENT_T_PER_M: f64 = 5e-11;

fn magnetic(
    sequence: u32,
    at_ms: u64,
    measured_t: f64,
    gradient_ned_t_per_m: [f64; 3],
    linearization_point: GeodeticPosition,
) -> Observation {
    Observation::new(
        stamp(30, 1, sequence, at_ms),
        ObservationValue::MagneticAnomaly {
            measured_t,
            map_t_at_lin: MAP_T,
            gradient_ned_t_per_m,
            linearization_point,
            valid_radius_m: 500.0,
            variance_t2: VARIANCE_T2,
        },
        SourceComposition::of(SensorClass::MagneticAnomaly),
    )
}

fn fix_with_variance(
    sequence: u32,
    at_ms: u64,
    position: GeodeticPosition,
    variance: f64,
) -> Observation {
    Observation::new(
        stamp(1, 1, sequence, at_ms),
        ObservationValue::PositionFix {
            position,
            covariance: SymmetricCov3::from_diagonal(variance, variance, variance),
        },
        SourceComposition::of(SensorClass::Gnss),
    )
}

fn core(f: &NavigationFilter) -> super::super::Core {
    f.core.expect("initialized")
}

#[test]
fn a_survey_scale_anomaly_pulls_the_position_along_the_gradient() {
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    let before = core(&f);
    // The vehicle reads the map value 10 m north of the fix.
    let measured = MAP_T + SURVEY_GRADIENT_T_PER_M * 10.0;
    let obs = magnetic(
        1,
        0,
        measured,
        [SURVEY_GRADIENT_T_PER_M, 0.0, 0.0],
        origin(),
    );
    assert_accepted(f.ingest(&obs, t(0)));
    let after = core(&f);
    assert!(after.state.x[0] > before.state.x[0]);
    assert!(after.state.p[(0, 0)] < before.state.p[(0, 0)]);
    assert!((after.state.p[(1, 1)] - before.state.p[(1, 1)]).abs() < 1e-12);
    assert!((after.state.p[(2, 2)] - before.state.p[(2, 2)]).abs() < 1e-12);
    let stats = f
        .innovation_stats(MeasurementKind::MagneticAnomaly)
        .expect("one update");
    assert_eq!(stats.admitted_count, 1);
    // ν = 0.5 nT, S = 25·(5e-11)² + 9e-18.
    let expected_nis = 25e-20 / (25.0 * 25e-22 + VARIANCE_T2);
    assert!((stats.last_nis - expected_nis).abs() < 1e-12, "{stats:?}");
    assert!(f.innovation_stats(MeasurementKind::PositionFix).is_none());
}

#[test]
fn a_rotated_gradient_50_km_from_the_origin_predicts_the_state_exactly() {
    let config = FusionConfig {
        reanchor_distance_m: 0.0,
        ..FusionConfig::default()
    };
    let mut f = NavigationFilter::new(config, CLOCK);
    let plane = LocalTangentPlane::new(origin()).expect("plausible origin");
    let far = plane.from_ned(&NedOffset::new(48_000.0, 3_000.0, 0.0));
    assert_accepted(f.ingest(&fix_with_variance(1, 0, origin(), 1e10), t(0)));
    assert_accepted(f.ingest(&fix_with_variance(2, 0, far, 25.0), t(0)));
    let before = core(&f);
    let x = before.state.x;
    let vehicle = plane.from_ned(&NedOffset::new(x[0], x[1], x[2]));
    // The map is linearized 364 m from the vehicle, inside the 500 m
    // radius.
    let vehicle_local = LocalTangentPlane::new(vehicle).expect("plausible point");
    let lin_point = vehicle_local.from_ned(&NedOffset::new(300.0, -200.0, 50.0));
    let local = LocalTangentPlane::new(lin_point).expect("plausible point");
    let gradient = [3e-11, -2e-11, 4e-11];
    let offset = local.to_ned(&vehicle);
    let measured = MAP_T
        + gradient[0] * offset.north_m
        + gradient[1] * offset.east_m
        + gradient[2] * offset.down_m;
    assert_accepted(f.ingest(&magnetic(1, 0, measured, gradient, lin_point), t(0)));
    let after = core(&f);
    // Zero innovation: the mean stays. An unrotated gradient gives an
    // innovation of approximately 1e-13 T, which moves the mean by
    // approximately 1e-5 m.
    let moved_m = (after.state.x - before.state.x).fixed_rows::<3>(0).norm();
    assert!(moved_m < 1e-7, "{moved_m}");
    assert!(after.state.p[(2, 2)] < before.state.p[(2, 2)]);
    assert!(after.state.p[(0, 0)] < before.state.p[(0, 0)]);
}

#[test]
fn map_refusals_name_the_reason_and_change_nothing() {
    let mut f = filter();
    // A 1 km 1-sigma fix exceeds the 500 m radius by the sigma margin.
    assert_accepted(f.ingest(&fix_with_variance(1, 0, origin(), 1e6), t(0)));
    let gradient = [SURVEY_GRADIENT_T_PER_M, 0.0, 0.0];
    assert!(matches!(
        rejection(f.ingest(&magnetic(1, 0, MAP_T, gradient, origin()), t(0))),
        RejectionReason::LinearizationRegionExceeded { .. }
    ));
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    let before = core(&f);
    // A tight state 1 km from the linearization point.
    let far_point = position_north_m(1_000.0);
    assert!(matches!(
        rejection(f.ingest(&magnetic(1, 0, MAP_T, gradient, far_point), t(0))),
        RejectionReason::LinearizationRegionExceeded { .. }
    ));
    // A flat cell gives an information ratio far below the floor.
    let flat = [1e-13, 0.0, 0.0];
    assert!(matches!(
        rejection(f.ingest(&magnetic(1, 0, MAP_T, flat, origin()), t(0))),
        RejectionReason::UninformativeGradient { information_ratio, .. } if information_ratio < 1e-6
    ));
    assert_eq!(core(&f).state, before.state);
    assert_eq!(f.history.len(), 0);
    assert_eq!(f.rejections().linearization_region_exceeded, 1);
    assert_eq!(f.rejections().uninformative_gradient, 1);
    assert!(
        f.innovation_stats(MeasurementKind::MagneticAnomaly)
            .is_none()
    );
}

#[test]
fn invalid_magnetic_values_are_screened_at_admission() {
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    let good = magnetic(1, 0, MAP_T, [SURVEY_GRADIENT_T_PER_M, 0.0, 0.0], origin());
    let ObservationValue::MagneticAnomaly {
        measured_t,
        map_t_at_lin,
        gradient_ned_t_per_m,
        linearization_point,
        valid_radius_m,
        variance_t2,
    } = good.value
    else {
        panic!("magnetic value");
    };
    let with = |value| Observation::new(good.stamp, value, good.composition);
    let cases = [
        (
            ObservationValue::MagneticAnomaly {
                measured_t: f64::NAN,
                map_t_at_lin,
                gradient_ned_t_per_m,
                linearization_point,
                valid_radius_m,
                variance_t2,
            },
            RejectionReason::NonFiniteValue,
        ),
        (
            ObservationValue::MagneticAnomaly {
                measured_t,
                map_t_at_lin,
                gradient_ned_t_per_m: [0.0, f64::INFINITY, 0.0],
                linearization_point,
                valid_radius_m,
                variance_t2,
            },
            RejectionReason::NonFiniteValue,
        ),
        (
            ObservationValue::MagneticAnomaly {
                measured_t,
                map_t_at_lin,
                gradient_ned_t_per_m,
                linearization_point: GeodeticPosition::new(f64::NAN, 0.0, 0.0),
                valid_radius_m,
                variance_t2,
            },
            RejectionReason::NonFiniteValue,
        ),
        (
            ObservationValue::MagneticAnomaly {
                measured_t,
                map_t_at_lin,
                gradient_ned_t_per_m,
                linearization_point,
                valid_radius_m: 0.0,
                variance_t2,
            },
            RejectionReason::NonFiniteValue,
        ),
        (
            ObservationValue::MagneticAnomaly {
                measured_t,
                map_t_at_lin,
                gradient_ned_t_per_m,
                linearization_point,
                valid_radius_m,
                variance_t2: 0.0,
            },
            RejectionReason::ImplausibleCovariance,
        ),
    ];
    for (value, expected) in cases {
        assert_eq!(
            f.ingest(&with(value), t(0)),
            IngestOutcome::Rejected(expected)
        );
    }
    // A tesla-scale variance has no lower limit: 9e-18 T² is valid.
    assert_accepted(f.ingest(&good, t(0)));
}

#[test]
fn flat_cell_magnetics_never_refresh_silence_or_add_redundancy() {
    // Small process noise keeps the sigma Good, so only silence can
    // degrade the quality.
    let config = FusionConfig {
        process_noise_accel_psd: 1e-6,
        initial_velocity_variance_m2_per_s2: 1e-6,
        ..FusionConfig::default()
    };
    let mut f = NavigationFilter::new(config, CLOCK);
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    let flat = [1e-13, 0.0, 0.0];
    let mut offered = 0_u64;
    for step in 1..=12_u32 {
        let at_ms = u64::from(step) * 500;
        let obs = magnetic(step, at_ms, MAP_T, flat, origin());
        assert!(matches!(
            rejection(f.ingest(&obs, t(at_ms))),
            RejectionReason::UninformativeGradient { .. }
        ));
        offered = offered.wrapping_add(1);
        assert_eq!(f.last_admitted_at, Some(t(0)));
        let solution = f.tick(t(at_ms)).expect("initialized");
        assert_ne!(
            solution.integrity.redundancy,
            Redundancy::IndependentClasses
        );
        assert!(
            !solution
                .integrity
                .contributing
                .contains(SensorClass::MagneticAnomaly)
        );
        assert!(!solution.composition.contains(SensorClass::MagneticAnomaly));
        let expected = if at_ms >= 5_000 {
            SolutionQuality::Unusable
        } else if at_ms >= 2_000 {
            SolutionQuality::Degraded
        } else {
            SolutionQuality::Good
        };
        assert_eq!(solution.integrity.quality, expected, "at {at_ms} ms");
    }
    assert_eq!(f.rejections().uninformative_gradient, offered);

    // Control: an informative gradient is admitted and does add an
    // independent class.
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    let survey = [SURVEY_GRADIENT_T_PER_M, 0.0, 0.0];
    assert_accepted(f.ingest(&magnetic(1, 100, MAP_T, survey, origin()), t(100)));
    let solution = f.tick(t(100)).expect("initialized");
    assert_eq!(
        solution.integrity.redundancy,
        Redundancy::IndependentClasses
    );
}
