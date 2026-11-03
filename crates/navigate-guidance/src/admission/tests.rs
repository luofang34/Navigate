//! Admission and leg geometry on the solution ellipsoid (ADR-0013).

use navigate_contract::{
    DurationNanos, Ellipsoid, FrameId, GeodeticPosition, GuidanceSetpoint, LateralReference,
    MonotonicNanos, SolutionQuality, Waypoint,
};

use crate::config::{GuidanceConfig, VelocityGuidanceConfig};
use crate::derive::guide;
use crate::refusal::GuidanceRefusal;
use crate::scenario::{CLOCK, deg, equator_leg, id, now, solution};
use crate::velocity::guide_velocity;

/// The lunar mean radius in meters.
const MOON_RADIUS_M: f64 = 1_737_400.0;

fn lateral_m(setpoint: GuidanceSetpoint) -> f64 {
    match setpoint {
        GuidanceSetpoint::DeviationTracking { lateral_m, .. } => lateral_m,
        other => panic!("expected DeviationTracking, got {other:?}"),
    }
}

fn east_mps(setpoint: GuidanceSetpoint) -> f64 {
    match setpoint {
        GuidanceSetpoint::Velocity { velocity } => velocity.east_mps,
        other => panic!("expected Velocity, got {other:?}"),
    }
}

#[test]
fn an_implausible_ellipsoid_draws_one_refusal_from_both_derivations() {
    let (from, to) = equator_leg();
    let refused = [
        Ellipsoid::new(FrameId::MOON_ME, -1.0, 0.0),
        Ellipsoid::new(FrameId::MARS_IAU, 3_396_190.0, f64::NAN),
        Ellipsoid::sphere(FrameId::MOON_J2000, MOON_RADIUS_M),
    ];
    for ellipsoid in refused {
        let own =
            solution(SolutionQuality::Good, deg(0.0, 0.5, 0.0), now()).with_ellipsoid(ellipsoid);
        let reference = LateralReference::track(from);
        let deviation = guide(
            &own,
            reference,
            &to,
            now(),
            CLOCK,
            &GuidanceConfig::default(),
        )
        .expect_err("refuses");
        let velocity = guide_velocity(
            &own,
            reference,
            &to,
            now(),
            CLOCK,
            &VelocityGuidanceConfig::default(),
        )
        .expect_err("refuses");
        assert!(
            matches!(
                deviation,
                GuidanceRefusal::ImplausibleEllipsoid { semi_major_axis_m, .. }
                    if semi_major_axis_m.to_bits() == ellipsoid.semi_major_axis_m.to_bits()
            ),
            "{deviation:?}"
        );
        assert!(matches!(
            velocity,
            GuidanceRefusal::ImplausibleEllipsoid { .. }
        ));
    }
}

#[test]
fn the_ellipsoid_is_judged_after_age_and_before_the_target() {
    let bad = Ellipsoid::new(FrameId::MOON_ME, 0.0, 0.0);
    let config = GuidanceConfig::default();
    let stale_at = MonotonicNanos::from_nanos(0);
    let stale = solution(SolutionQuality::Good, deg(0.0, 0.5, 0.0), stale_at).with_ellipsoid(bad);
    let late = MonotonicNanos::from_nanos(
        config
            .max_solution_age
            .as_nanos()
            .saturating_add(DurationNanos::from_millis(1).as_nanos()),
    );
    assert!(matches!(
        guide(
            &stale,
            LateralReference::PresentPosition,
            &equator_leg().1,
            late,
            CLOCK,
            &config
        ),
        Err(GuidanceRefusal::SolutionStale { .. })
    ));
    let fresh = solution(SolutionQuality::Good, deg(0.0, 0.5, 0.0), now()).with_ellipsoid(bad);
    let nowhere = Waypoint::new(id("NOWHERE"), GeodeticPosition::new(f64::NAN, 0.0, 0.0));
    assert!(matches!(
        guide(
            &fresh,
            LateralReference::PresentPosition,
            &nowhere,
            now(),
            CLOCK,
            &config
        ),
        Err(GuidanceRefusal::ImplausibleEllipsoid { .. })
    ));
}

#[test]
fn ellipsoid_mismatch_is_refused_when_expected_ellipsoid_is_configured() {
    let (from, to) = equator_leg();
    let earth_solution = solution(SolutionQuality::Good, deg(0.0, 0.5, 0.0), now());
    let lunar_config = GuidanceConfig::default().with_expected_ellipsoid(Ellipsoid::MOON_ME);
    let refusal = guide(
        &earth_solution,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &lunar_config,
    )
    .expect_err("Earth solution on Moon plan must be refused");
    assert_eq!(
        refusal,
        GuidanceRefusal::EllipsoidMismatch {
            expected: Ellipsoid::MOON_ME,
            got: Ellipsoid::WGS84,
        }
    );
    let vel_config = VelocityGuidanceConfig {
        admission: lunar_config,
        ..VelocityGuidanceConfig::default()
    };
    let vel_refusal = guide_velocity(
        &earth_solution,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &vel_config,
    )
    .expect_err("velocity guidance must refuse identically");
    assert_eq!(vel_refusal, refusal);
}

#[test]
fn lunar_cross_track_uses_the_lunar_radius() {
    let (from, to) = equator_leg();
    // 500 m right of an eastbound equatorial track is 500 m south.
    let offset_rad = 500.0 / MOON_RADIUS_M;
    let position = GeodeticPosition::new(-offset_rad, 0.5_f64.to_radians(), 0.0);
    let config = GuidanceConfig::default();
    for ellipsoid in [Ellipsoid::MOON_ME, Ellipsoid::MOON_PA] {
        let own = solution(SolutionQuality::Good, position, now()).with_ellipsoid(ellipsoid);
        let track = guide(
            &own,
            LateralReference::track(from),
            &to,
            now(),
            CLOCK,
            &config,
        )
        .expect("guides");
        assert!((lateral_m(track.setpoint) - 500.0).abs() < 1e-3);
        let eastbound = core::f64::consts::FRAC_PI_2;
        let course = guide(
            &own,
            LateralReference::course(eastbound),
            &to,
            now(),
            CLOCK,
            &config,
        )
        .expect("guides");
        assert!((lateral_m(course.setpoint) - 500.0).abs() < 1e-3);
    }
    // The same coordinates on the Earth are more than 1.8 km off track.
    let earth = solution(SolutionQuality::Good, position, now());
    let on_earth = guide(
        &earth,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &config,
    )
    .expect("guides");
    assert!(lateral_m(on_earth.setpoint) > 1_800.0);
}

#[test]
fn martian_cross_track_uses_the_martian_mean_radius() {
    let (from, to) = equator_leg();
    let mars_mean_radius_m = (2.0 * 3_396_190.0 + 3_376_200.0) / 3.0;
    let position = GeodeticPosition::new(250.0 / mars_mean_radius_m, 0.5_f64.to_radians(), 0.0);
    let own = solution(SolutionQuality::Good, position, now()).with_ellipsoid(Ellipsoid::MARS_IAU);
    let command = guide(
        &own,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &GuidanceConfig::default(),
    )
    .expect("guides");
    assert!((lateral_m(command.setpoint) + 250.0).abs() < 1e-3);
}

#[test]
fn the_lunar_arrival_slowdown_reads_the_lunar_distance() {
    let (_, to) = equator_leg();
    // 15 m short of the waypoint on the Moon, 55 m short on the Earth.
    let short_rad = 15.0 / MOON_RADIUS_M;
    let position = GeodeticPosition::new(0.0, 1.0_f64.to_radians() - short_rad, 0.0);
    let config = VelocityGuidanceConfig::default();
    let moon = solution(SolutionQuality::Good, position, now()).with_ellipsoid(Ellipsoid::MOON_ME);
    let lunar = guide_velocity(
        &moon,
        LateralReference::PresentPosition,
        &to,
        now(),
        CLOCK,
        &config,
    )
    .expect("guides");
    // Inside the 30 m slowdown radius: half of the 2 m/s cruise.
    assert!((east_mps(lunar.setpoint) - 1.0).abs() < 1e-3);
    let earth = solution(SolutionQuality::Good, position, now());
    let terrestrial = guide_velocity(
        &earth,
        LateralReference::PresentPosition,
        &to,
        now(),
        CLOCK,
        &config,
    )
    .expect("guides");
    assert!((east_mps(terrestrial.setpoint) - 2.0).abs() < 1e-3);
}
