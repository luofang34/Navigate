//! Procedure-constraint derivation tests (NAV-VC-001/002/003).

#![allow(clippy::expect_used, clippy::panic)]

use navigate_contract::{AltitudeConstraint, GuidanceSetpoint, LateralReference, SolutionQuality};

use crate::config::{GuidanceConfig, VelocityGuidanceConfig};
use crate::derive::guide;
use crate::scenario::{CLOCK, deg, equator_leg, now, solution};
use crate::velocity::guide_velocity;

#[test]
fn nav_vc_003_a_waypoint_speed_constraint_bounds_the_leg_toward_it() {
    let (from, to) = equator_leg();
    let constrained = to.clone().with_max_speed(0.8);
    let own = solution(SolutionQuality::Good, deg(0.0, 0.3, 0.0), now());
    let config = VelocityGuidanceConfig::default();
    let command = guide_velocity(
        &own,
        LateralReference::track(from),
        &constrained,
        now(),
        CLOCK,
        &config,
    )
    .expect("constrained leg guides");
    let GuidanceSetpoint::Velocity { velocity } = command.setpoint else {
        panic!("velocity derivation emits velocity setpoints");
    };
    let speed = velocity.north_mps.hypot(velocity.east_mps);
    assert!(
        speed <= 0.8 + 1e-9,
        "cruise must honor the 0.8 m/s constraint, got {speed}"
    );
    assert!(speed > 0.5, "the constraint bounds, it does not stall");
}

#[test]
fn nav_vc_002_a_tighter_gradient_binds_the_vertical_rate() {
    let (from, to) = equator_leg();
    // 200 m above an At profile: the gain demands far more descent than
    // either limit allows.
    let profiled = to
        .clone()
        .with_altitude(AltitudeConstraint::At(0.0))
        .with_gradient(0.05);
    let own = solution(SolutionQuality::Good, deg(0.0, 0.3, 200.0), now());
    let config = VelocityGuidanceConfig::default();
    let command = guide_velocity(
        &own,
        LateralReference::track(from),
        &profiled,
        now(),
        CLOCK,
        &config,
    )
    .expect("profiled leg guides");
    let GuidanceSetpoint::Velocity { velocity } = command.setpoint else {
        panic!("velocity derivation emits velocity setpoints");
    };
    // gradient 0.05 at cruise 2 m/s bounds descent to 0.1 m/s — tighter
    // than the 1.0 m/s ceiling.
    assert!(
        (velocity.down_mps - 0.1).abs() < 1e-9,
        "gradient limit binds, got {}",
        velocity.down_mps
    );

    // A loose gradient leaves the configured ceiling in charge.
    let loose = to
        .clone()
        .with_altitude(AltitudeConstraint::At(0.0))
        .with_gradient(10.0);
    let command = guide_velocity(
        &own,
        LateralReference::track(from),
        &loose,
        now(),
        CLOCK,
        &config,
    )
    .expect("loosely profiled leg guides");
    let GuidanceSetpoint::Velocity { velocity } = command.setpoint else {
        panic!("velocity derivation emits velocity setpoints");
    };
    assert!(
        (velocity.down_mps - config.max_vertical_mps).abs() < 1e-9,
        "ceiling binds when the gradient is looser, got {}",
        velocity.down_mps
    );
}

#[test]
fn nav_vc_001_a_window_flows_through_the_deviation_derivation_too() {
    let (from, to) = equator_leg();
    let windowed = to.clone().with_altitude(AltitudeConstraint::Window {
        lower_m: 100.0,
        upper_m: 200.0,
    });
    let own = solution(SolutionQuality::Good, deg(0.0, 0.3, 80.0), now());
    let command = guide(
        &own,
        LateralReference::track(from),
        &windowed,
        now(),
        CLOCK,
        &GuidanceConfig::default(),
    )
    .expect("windowed leg guides");
    let GuidanceSetpoint::DeviationTracking { vertical_m, .. } = command.setpoint else {
        panic!("deviation derivation emits deviation setpoints");
    };
    assert!(
        (vertical_m + 20.0).abs() < 1e-9,
        "20 m below the band is a -20 m deviation, got {vertical_m}"
    );
}

#[test]
fn nav_tt_004_a_fly_over_rejoin_reports_honest_deviation_and_corrects_toward_the_new_leg() {
    // Ownship has just overflown a fix 300 m LEFT of the onward track
    // (the fly-over rejoin geometry). Guidance against the new leg must
    // report the real deviation and command a velocity whose lateral
    // component reduces it — no synthetic intercept, no zeroed needle.
    let (from, to) = equator_leg();
    // The new leg runs west→east along the equator; 300 m left of track
    // is 300 m NORTH of it.
    let own = solution(
        SolutionQuality::Good,
        deg(300.0 / 111_194.926, 0.1, 0.0),
        now(),
    );
    let config = VelocityGuidanceConfig::default();
    let command = guide_velocity(
        &own,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &config,
    )
    .expect("the rejoin guides");
    let GuidanceSetpoint::Velocity { velocity } = command.setpoint else {
        panic!("velocity derivation emits velocity setpoints");
    };
    assert!(
        velocity.north_mps < -0.1,
        "north of an eastbound track corrects southward, got {}",
        velocity.north_mps
    );
    assert!(
        velocity.east_mps > 0.5,
        "progress along the new leg continues, got {}",
        velocity.east_mps
    );
}
