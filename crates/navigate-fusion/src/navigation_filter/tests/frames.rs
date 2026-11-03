//! Frame discipline and filtering on lunar and Martian ellipsoids
//! (ADR-0013).

use navigate_contract::{Ellipsoid, FrameId, SolutionQuality};
use navigate_geodesy::BodyEllipsoid;

use super::*;

/// The lunar mean radius in meters.
const MOON_RADIUS_M: f64 = 1_737_400.0;

fn lunar_origin() -> GeodeticPosition {
    GeodeticPosition::new(-0.15, 0.4, 0.0)
}

fn lunar_filter(config: FusionConfig) -> NavigationFilter {
    NavigationFilter::new_for_ellipsoid(config, CLOCK, BodyEllipsoid::MOON_ME)
}

fn fix_in(sequence: u32, at_ms: u64, position: GeodeticPosition, frame: FrameId) -> Observation {
    gnss_fix(1, sequence, at_ms, position).with_frame(frame)
}

#[test]
fn a_wgs84_filter_admits_earth_frames_and_undeclared_frames() {
    let mut f = filter();
    assert_eq!(f.ellipsoid(), BodyEllipsoid::WGS84);
    assert!(!f.config().require_explicit_frame);
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
    assert_accepted(f.ingest(&fix_in(2, 100, origin(), FrameId::EARTH_ITRF93), t(100)));
    let solution = f.tick(t(100)).expect("initialized");
    assert_eq!(solution.ellipsoid, Ellipsoid::WGS84);
}

#[test]
fn a_wgs84_filter_refuses_a_lunar_fix_without_any_change() {
    let mut f = filter();
    let reason = rejection(f.ingest(&fix_in(1, 0, origin(), FrameId::MOON_ME), t(0)));
    assert_eq!(
        reason,
        RejectionReason::FrameMismatch {
            expected: FrameId::EARTH_ITRF93,
            got: FrameId::MOON_ME,
        }
    );
    assert_eq!(f.rejections().frame_mismatch, 1);
    assert!(!f.is_initialized());
    // The refusal left the sequence position open.
    assert_accepted(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0)));
}

#[test]
fn an_explicit_frame_filter_refuses_an_undeclared_frame() {
    let mut f = lunar_filter(FusionConfig::default());
    assert!(f.config().require_explicit_frame);
    let reason = rejection(f.ingest(&gnss_fix(1, 1, 0, lunar_origin()), t(0)));
    assert_eq!(
        reason,
        RejectionReason::MissingFrame {
            expected: FrameId::MOON_ME,
        }
    );
    assert_eq!(f.rejections().missing_frame, 1);
    assert!(!f.is_initialized());
}

#[test]
fn a_wgs84_filter_with_explicit_frames_refuses_an_undeclared_frame() {
    let config = FusionConfig {
        require_explicit_frame: true,
        ..FusionConfig::default()
    };
    let mut f = NavigationFilter::new(config, CLOCK);
    assert!(matches!(
        rejection(f.ingest(&gnss_fix(1, 1, 0, origin()), t(0))),
        RejectionReason::MissingFrame { .. }
    ));
}

#[test]
fn a_mean_earth_filter_refuses_a_principal_axis_fix() {
    let mut f = lunar_filter(FusionConfig::default());
    let principal = fix_in(1, 0, lunar_origin(), FrameId::MOON_PA);
    assert_eq!(
        rejection(f.ingest(&principal, t(0))),
        RejectionReason::FrameMismatch {
            expected: FrameId::MOON_ME,
            got: FrameId::MOON_PA,
        }
    );
    assert_eq!(f.rejections().frame_mismatch, 1);
    assert!(!f.is_initialized());
    // The same fix in the mean Earth frame is admitted.
    assert_accepted(f.ingest(&fix_in(1, 0, lunar_origin(), FrameId::MOON_ME), t(0)));
    let solution = f.tick(t(0)).expect("initialized");
    assert_eq!(solution.ellipsoid, Ellipsoid::MOON_ME);
    assert!((solution.position.latitude_rad - lunar_origin().latitude_rad).abs() < 1e-12);
}

#[test]
fn the_frame_gate_runs_after_the_clock_gate_and_before_composition() {
    let mut f = lunar_filter(FusionConfig::default());
    let foreign_clock = Observation::in_frame(
        ObservationStamp::new(
            SourceId::new(1),
            SourceEpoch::new(1),
            WrappingSequence::new(1),
            t(0),
            ClockDomainId::new(99),
        ),
        ObservationValue::PositionFix {
            position: lunar_origin(),
            covariance: SymmetricCov3::from_diagonal(25.0, 25.0, 25.0),
        },
        SourceComposition::of(SensorClass::Gnss),
        FrameId::MOON_PA,
    );
    assert!(matches!(
        rejection(f.ingest(&foreign_clock, t(0))),
        RejectionReason::ClockDomainMismatch { .. }
    ));
    let empty = position_fix(
        stamp(1, 1, 1, 0),
        lunar_origin(),
        SourceComposition::empty(),
    )
    .with_frame(FrameId::MOON_PA);
    assert!(matches!(
        rejection(f.ingest(&empty, t(0))),
        RejectionReason::FrameMismatch { .. }
    ));
    assert_eq!(f.rejections().empty_composition, 0);
}

/// Flies north at 50 m/s on the lunar sphere for 60 s with position and
/// velocity fixes every 500 ms. The re-anchor distance is 1 km, so the
/// flight crosses several planes. The velocity prior is wide enough to
/// admit the first 50 m/s velocity fix.
#[test]
fn a_lunar_flight_tracks_the_truth_across_re_anchors() {
    let config = FusionConfig {
        reanchor_distance_m: 1_000.0,
        initial_velocity_variance_m2_per_s2: 10_000.0,
        ..FusionConfig::default()
    };
    let mut f = lunar_filter(config);
    let speed_mps = 50.0;
    let velocity = NedVelocity::new(speed_mps, 0.0, 0.0);
    let truth_at = |at_ms: u64| {
        let travelled_m = speed_mps * at_ms as f64 * 1e-3;
        let o = lunar_origin();
        GeodeticPosition::new(
            o.latitude_rad + travelled_m / MOON_RADIUS_M,
            o.longitude_rad,
            0.0,
        )
    };
    for step in 0..=120_u32 {
        let at_ms = u64::from(step) * 500;
        let fix = fix_in(step + 1, at_ms, truth_at(at_ms), FrameId::MOON_ME);
        assert_accepted(f.ingest(&fix, t(at_ms)));
        let vel = velocity_fix(2, step + 1, at_ms, velocity).with_frame(FrameId::MOON_ME);
        assert_accepted(f.ingest(&vel, t(at_ms)));
    }
    assert_eq!(f.rejections().innovation_gate, 0);
    let solution = f.tick(t(60_000)).expect("initialized");
    assert_eq!(solution.ellipsoid, Ellipsoid::MOON_ME);
    let truth = truth_at(60_000);
    let north_error_m = (solution.position.latitude_rad - truth.latitude_rad) * MOON_RADIUS_M;
    assert!(north_error_m.abs() < 1.0, "north error {north_error_m} m");
    assert!(solution.position.altitude_m.abs() < 1.0);
    assert!((solution.velocity.north_mps - speed_mps).abs() < 0.5);
    assert!(solution.velocity.east_mps.abs() < 0.5);
}

/// The Martian form of the ellipsoidal altitude test: a fix 48 km from
/// the plane origin, then an altitude reading. A flat-plane model would
/// see the curvature drop of the plane as an innovation.
#[test]
fn a_martian_altitude_50_km_from_the_origin_has_no_curvature_bias() {
    let config = FusionConfig {
        reanchor_distance_m: 0.0,
        ..FusionConfig::default()
    };
    let mars = BodyEllipsoid::MARS_IAU;
    let origin = GeodeticPosition::new(-0.08, 2.4, -2_500.0);
    let plane = LocalTangentPlane::with_ellipsoid(origin, mars).expect("plausible origin");
    let on_plane = plane.from_ned(&NedOffset::new(48_000.0, 0.0, 0.0));
    // The plane drops below the surface: the curvature drop on Mars is
    // larger than on the Earth.
    assert!(on_plane.altitude_m - origin.altitude_m > 300.0);
    let truth = GeodeticPosition::new(on_plane.latitude_rad, on_plane.longitude_rad, -2_500.0);
    let mut f = NavigationFilter::new_for_ellipsoid(config, CLOCK, mars);
    let diffuse = Observation::in_frame(
        stamp(1, 1, 1, 0),
        ObservationValue::PositionFix {
            position: origin,
            covariance: SymmetricCov3::from_diagonal(1e10, 1e10, 1e10),
        },
        SourceComposition::of(SensorClass::Gnss),
        FrameId::MARS_IAU,
    );
    assert_accepted(f.ingest(&diffuse, t(0)));
    let fix = Observation::in_frame(
        stamp(1, 1, 2, 0),
        ObservationValue::PositionFix {
            position: truth,
            covariance: SymmetricCov3::from_diagonal(25.0, 25.0, 900.0),
        },
        SourceComposition::of(SensorClass::Gnss),
        FrameId::MARS_IAU,
    );
    assert_accepted(f.ingest(&fix, t(0)));
    let altitude = Observation::in_frame(
        stamp(40, 1, 1, 0),
        ObservationValue::Altitude {
            altitude_m: -2_490.0,
            variance_m2: 4.0,
        },
        SourceComposition::of(SensorClass::Barometric),
        FrameId::MARS_IAU,
    );
    assert_accepted(f.ingest(&altitude, t(0)));
    let after = f.tick(t(0)).expect("initialized");
    assert_eq!(after.ellipsoid, Ellipsoid::MARS_IAU);
    let expected_m = -2_500.0 + 10.0 * 900.0 / 904.0;
    assert!(
        (after.position.altitude_m - expected_m).abs() < 0.01,
        "{}",
        after.position.altitude_m
    );
    assert_eq!(after.integrity.quality, SolutionQuality::Good);
}

#[test]
fn reset_keeps_the_ellipsoid_and_the_frame_discipline() {
    let mut f = lunar_filter(FusionConfig::default());
    assert_accepted(f.ingest(&fix_in(1, 0, lunar_origin(), FrameId::MOON_ME), t(0)));
    f.reset();
    assert_eq!(f.ellipsoid(), BodyEllipsoid::MOON_ME);
    assert!(matches!(
        rejection(f.ingest(&gnss_fix(1, 2, 100, lunar_origin()), t(100))),
        RejectionReason::MissingFrame { .. }
    ));
}
