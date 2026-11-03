//! Ellipsoidal altitude updates and innovation statistics.

use navigate_contract::SolutionQuality;

use super::*;
use crate::observation::MeasurementKind;

fn altitude(sequence: u32, at_ms: u64, altitude_m: f64) -> Observation {
    Observation::new(
        stamp(40, 1, sequence, at_ms),
        ObservationValue::Altitude {
            altitude_m,
            variance_m2: 4.0,
        },
        SourceComposition::of(SensorClass::Barometric),
    )
}

/// A GNSS fix with a 5 m horizontal and a 30 m vertical 1-sigma.
fn tall_fix(
    sequence: u32,
    at_ms: u64,
    position: GeodeticPosition,
    horizontal_m2: f64,
) -> Observation {
    Observation::new(
        stamp(1, 1, sequence, at_ms),
        ObservationValue::PositionFix {
            position,
            covariance: SymmetricCov3::from_diagonal(horizontal_m2, horizontal_m2, 900.0),
        },
        SourceComposition::of(SensorClass::Gnss),
    )
}

#[test]
fn an_altitude_at_the_origin_restores_vertical_quality() {
    let mut f = filter();
    assert_accepted(f.ingest(&tall_fix(1, 0, origin(), 25.0), t(0)));
    let before = f.tick(t(0)).expect("initialized");
    assert_eq!(before.integrity.quality, SolutionQuality::Degraded);
    assert_accepted(f.ingest(&altitude(1, 0, 510.0), t(0)));
    let after = f.tick(t(0)).expect("initialized");
    // Scalar Kalman update: 500 + 10·900/904, σ² = 900·4/904.
    let expected_m = 500.0 + 10.0 * 900.0 / 904.0;
    assert!((after.position.altitude_m - expected_m).abs() < 1e-3);
    let expected_sigma_m = (900.0_f64 * 4.0 / 904.0).sqrt();
    assert!((after.integrity.vertical_1sigma_m - expected_sigma_m).abs() < 1e-9);
    assert_eq!(after.integrity.quality, SolutionQuality::Good);
}

#[test]
fn an_altitude_50_km_from_the_origin_has_no_curvature_bias() {
    let config = FusionConfig {
        reanchor_distance_m: 0.0,
        ..FusionConfig::default()
    };
    let plane = LocalTangentPlane::new(origin()).expect("plausible origin");
    let on_plane = plane.from_ned(&NedOffset::new(48_000.0, 0.0, 0.0));
    let truth = GeodeticPosition::new(on_plane.latitude_rad, on_plane.longitude_rad, 500.0);
    // A diffuse first fix, then a fix at the far position.
    let mut f = NavigationFilter::new(config, CLOCK);
    let diffuse = Observation::new(
        stamp(1, 1, 1, 0),
        ObservationValue::PositionFix {
            position: origin(),
            covariance: SymmetricCov3::from_diagonal(1e10, 1e10, 1e10),
        },
        SourceComposition::of(SensorClass::Gnss),
    );
    assert_accepted(f.ingest(&diffuse, t(0)));
    assert_accepted(f.ingest(&tall_fix(2, 0, truth, 25.0), t(0)));
    let before = f.tick(t(0)).expect("initialized");
    assert!((before.position.altitude_m - 500.0).abs() < 0.01);
    assert_eq!(before.integrity.quality, SolutionQuality::Degraded);
    // A flat-plane model would see a 180 m innovation and gate this
    // reading. The ellipsoidal model sees 10 m.
    assert_accepted(f.ingest(&altitude(1, 0, 510.0), t(0)));
    let after = f.tick(t(0)).expect("initialized");
    let expected_m = 500.0 + 10.0 * 900.0 / 904.0;
    assert!(
        (after.position.altitude_m - expected_m).abs() < 0.01,
        "{}",
        after.position.altitude_m
    );
    assert!(after.integrity.vertical_1sigma_m < 2.1);
    assert_eq!(after.integrity.quality, SolutionQuality::Good);
    let stats = f
        .innovation_stats(MeasurementKind::Altitude)
        .expect("one update");
    assert!((stats.last_nis - 100.0 / 904.0).abs() < 1e-3, "{stats:?}");
}

#[test]
fn altitude_admission_screens_values_and_needs_initialization() {
    let mut f = filter();
    assert_eq!(
        f.ingest(&altitude(1, 0, 500.0), t(0)),
        IngestOutcome::Rejected(RejectionReason::NotInitialized)
    );
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    assert_eq!(
        f.ingest(&altitude(1, 0, f64::INFINITY), t(0)),
        IngestOutcome::Rejected(RejectionReason::NonFiniteValue)
    );
    let zero_variance = Observation::new(
        stamp(40, 1, 1, 0),
        ObservationValue::Altitude {
            altitude_m: 500.0,
            variance_m2: 0.0,
        },
        SourceComposition::of(SensorClass::Barometric),
    );
    assert_eq!(
        f.ingest(&zero_variance, t(0)),
        IngestOutcome::Rejected(RejectionReason::ImplausibleCovariance)
    );
}

#[test]
fn innovation_stats_track_gate_streaks_and_reset_with_the_epoch() {
    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    assert!(f.innovation_stats(MeasurementKind::PositionFix).is_none());
    // Two outliers 200 m above the fix.
    for sequence in 1..=2 {
        assert!(matches!(
            rejection(f.ingest(&altitude(sequence, 0, 700.0), t(0))),
            RejectionReason::InnovationGate { .. }
        ));
    }
    let streak = f
        .innovation_stats(MeasurementKind::Altitude)
        .expect("gate refusals");
    assert_eq!(streak.admitted_count, 0);
    assert_eq!(streak.consecutive_gate_rejections, 2);
    assert_accepted(f.ingest(&altitude(1, 10, 501.0), t(10)));
    assert_accepted(f.ingest(&altitude(2, 20, 499.0), t(20)));
    let stats = f
        .innovation_stats(MeasurementKind::Altitude)
        .expect("updates");
    assert_eq!(stats.admitted_count, 2);
    assert_eq!(stats.consecutive_gate_rejections, 0);
    assert!(stats.last_nis > 0.0);
    assert!(stats.mean_nis > 0.0);
    // A second fix is an update with an innovation, so it counts.
    assert_accepted(f.ingest(&gnss_fix(1, 2, 30, origin()), t(30)));
    assert_eq!(
        f.innovation_stats(MeasurementKind::PositionFix)
            .map(|s| s.admitted_count),
        Some(1)
    );
    f.reset();
    assert!(f.innovation_stats(MeasurementKind::Altitude).is_none());
    assert!(f.innovation_stats(MeasurementKind::PositionFix).is_none());
    assert_eq!(f.history.len(), 0);
}
