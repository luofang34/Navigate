#![allow(clippy::expect_used, clippy::panic)]
//! Velocity-derivation tests. Every sign is pinned against the geodesy
//! convention it derives from, and every cap is asserted as a magnitude
//! so a per-axis regression cannot pass.

use navigate_contract::{
    AltitudeConstraint, GeodeticPosition, GuidanceSetpoint, LateralReference, NedVelocity,
    SolutionQuality, Waypoint,
};
use navigate_geodesy::{cross_track_m, initial_bearing_rad, wgs84};

use crate::config::VelocityGuidanceConfig;
use crate::scenario::{CLOCK, deg, equator_leg, now, solution};

use super::guide_velocity;

mod refusal;

fn velocity_of(setpoint: GuidanceSetpoint) -> NedVelocity {
    match setpoint {
        GuidanceSetpoint::Velocity { velocity } => velocity,
        other => panic!("expected Velocity, got {other:?}"),
    }
}

fn speed_mps(velocity: NedVelocity) -> f64 {
    velocity.north_mps.hypot(velocity.east_mps)
}

/// A position `offset_m` south of the equator at `longitude_deg` —
/// right of an eastbound equatorial track.
fn right_of_equator(offset_m: f64, longitude_deg: f64) -> GeodeticPosition {
    GeodeticPosition::new(
        -offset_m / wgs84::MEAN_RADIUS_M,
        longitude_deg.to_radians(),
        0.0,
    )
}

#[test]
fn on_track_eastbound_commands_cruise_due_east() {
    let (from, to) = equator_leg();
    let own = solution(SolutionQuality::Good, deg(0.0, 0.5, 0.0), now());
    let config = VelocityGuidanceConfig::default();
    let command = guide_velocity(
        &own,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &config,
    )
    .expect("guides");
    let velocity = velocity_of(command.setpoint);
    assert!(velocity.north_mps.abs() < 1e-6, "on track: {velocity:?}");
    assert!(
        (velocity.east_mps - config.cruise_mps).abs() < 1e-9,
        "cruise eastbound: {velocity:?}"
    );
    assert!(
        velocity.down_mps.abs() < f64::EPSILON,
        "no constraint commands no vertical rate: {velocity:?}"
    );
    assert!(speed_mps(velocity) <= config.max_horizontal_mps);
    assert_eq!(command.issued_at, now());
    assert_eq!(command.basis, own.stamp);
}

#[test]
fn right_of_track_corrects_northward_at_the_configured_gain() {
    let (from, to) = equator_leg();
    // Small enough that the correction stays inside the ceiling, so the
    // gain itself is what the assertion pins.
    let own_pos = right_of_equator(4.0, 0.5);
    let own = solution(SolutionQuality::Good, own_pos, now());
    let config = VelocityGuidanceConfig::default();
    let command = guide_velocity(
        &own,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &config,
    )
    .expect("guides");
    let velocity = velocity_of(command.setpoint);
    let cross_m = cross_track_m(&own_pos, &from, &to.position).expect("valid track");
    assert!(
        (cross_m - 4.0).abs() < 0.01,
        "south of an eastbound track is 4 m right: {cross_m}"
    );
    // 0.3 per second against 4 m of deviation: 1.2 m/s of correction.
    let expected_north_mps = config.cross_track_gain_per_s * cross_m;
    assert!(
        velocity.north_mps > 0.0,
        "right of track must correct northward, toward the track: {velocity:?}"
    );
    assert!(
        (velocity.north_mps - expected_north_mps).abs() < 1e-9,
        "north {} vs gain × cross-track {expected_north_mps}",
        velocity.north_mps
    );
    assert!((velocity.east_mps - config.cruise_mps).abs() < 1e-9);
    assert!(speed_mps(velocity) <= config.max_horizontal_mps);
}

#[test]
fn left_of_track_corrects_southward() {
    let (from, to) = equator_leg();
    let own_pos = right_of_equator(-4.0, 0.5);
    let own = solution(SolutionQuality::Good, own_pos, now());
    let command = guide_velocity(
        &own,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &VelocityGuidanceConfig::default(),
    )
    .expect("guides");
    let velocity = velocity_of(command.setpoint);
    assert!(
        velocity.north_mps < 0.0,
        "left of track must correct southward, toward the track: {velocity:?}"
    );
}

#[test]
fn a_hundred_meters_right_of_track_saturates_the_cap_without_rotating_the_command() {
    let (from, to) = equator_leg();
    let own = solution(SolutionQuality::Good, right_of_equator(100.0, 0.5), now());
    let config = VelocityGuidanceConfig::default();
    let command = guide_velocity(
        &own,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &config,
    )
    .expect("guides");
    let velocity = velocity_of(command.setpoint);
    assert!(
        velocity.north_mps > 0.0,
        "correction stays northward under saturation: {velocity:?}"
    );
    assert!(
        (speed_mps(velocity) - config.max_horizontal_mps).abs() < 1e-9,
        "composed speed rides the cap: {velocity:?}"
    );
    // The correction saturates at the ceiling before composition, so the
    // uncapped vector is (2.5 north, 2.0 east); scaling preserves that
    // ratio where per-axis clipping would not.
    let expected_ratio = config.max_horizontal_mps / config.cruise_mps;
    assert!(
        (velocity.north_mps / velocity.east_mps - expected_ratio).abs() < 1e-9,
        "capping scaled the vector rather than clipping an axis: {velocity:?}"
    );
}

#[test]
fn a_huge_cross_track_stays_within_the_horizontal_cap() {
    let (from, to) = equator_leg();
    let own = solution(
        SolutionQuality::Good,
        right_of_equator(10_000.0, 0.5),
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
    .expect("guides");
    let velocity = velocity_of(command.setpoint);
    assert!(velocity.north_mps > 0.0, "still corrects toward the track");
    assert!(
        speed_mps(velocity) <= config.max_horizontal_mps + 1e-9,
        "10 km off track must not exceed the cap: {}",
        speed_mps(velocity)
    );
}

#[test]
fn direct_to_is_pure_bearing_aligned_velocity() {
    let own_pos = deg(10.0, 20.0, 0.0);
    let to = Waypoint::new("DCT".into(), deg(11.0, 21.0, 0.0));
    let own = solution(SolutionQuality::Good, own_pos, now());
    let config = VelocityGuidanceConfig::default();
    let command = guide_velocity(
        &own,
        LateralReference::PresentPosition,
        &to,
        now(),
        CLOCK,
        &config,
    )
    .expect("guides");
    let velocity = velocity_of(command.setpoint);
    let bearing_rad = initial_bearing_rad(&own_pos, &to.position);
    assert!(
        (velocity.north_mps - config.cruise_mps * bearing_rad.cos()).abs() < 1e-12,
        "north vs bearing-aligned cruise: {velocity:?}"
    );
    assert!(
        (velocity.east_mps - config.cruise_mps * bearing_rad.sin()).abs() < 1e-12,
        "east vs bearing-aligned cruise: {velocity:?}"
    );
    assert!(
        (speed_mps(velocity) - config.cruise_mps).abs() < 1e-12,
        "no cross-track deviation means no correction: {velocity:?}"
    );
}

#[test]
fn inside_the_slowdown_radius_speed_scales_with_distance_remaining() {
    let (from, to) = equator_leg();
    let config = VelocityGuidanceConfig::default();
    // Half the radius out, on track: half the cruise speed.
    let half_radius_m = config.arrival_slowdown_radius_m / 2.0;
    let own_pos = GeodeticPosition::new(
        0.0,
        to.position.longitude_rad - half_radius_m / wgs84::MEAN_RADIUS_M,
        0.0,
    );
    let own = solution(SolutionQuality::Good, own_pos, now());
    let command = guide_velocity(
        &own,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &config,
    )
    .expect("guides");
    let velocity = velocity_of(command.setpoint);
    assert!(
        (speed_mps(velocity) - config.cruise_mps / 2.0).abs() < 1e-6,
        "half the radius out commands half of cruise: {velocity:?}"
    );
}

#[test]
fn the_approach_speed_floor_holds_at_the_waypoint() {
    let (from, to) = equator_leg();
    let config = VelocityGuidanceConfig::default();
    let own_pos = GeodeticPosition::new(
        0.0,
        to.position.longitude_rad - 1.0 / wgs84::MEAN_RADIUS_M,
        0.0,
    );
    let own = solution(SolutionQuality::Good, own_pos, now());
    let command = guide_velocity(
        &own,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &config,
    )
    .expect("guides");
    let velocity = velocity_of(command.setpoint);
    // Linear scaling alone would command 1/30 of cruise here; the floor
    // keeps the leg flyable, and terminating it stays navigate-fpl's call.
    assert!(
        (speed_mps(velocity) - VelocityGuidanceConfig::MIN_APPROACH_SPEED_MPS).abs() < 1e-6,
        "one meter out holds the approach floor: {velocity:?}"
    );
}

#[test]
fn above_the_profile_descends_and_below_it_climbs() {
    let (from, _) = equator_leg();
    let to = Waypoint::new("END".into(), deg(0.0, 1.0, 0.0))
        .with_altitude(AltitudeConstraint::At(100.0));
    let config = VelocityGuidanceConfig::default();
    let above = solution(SolutionQuality::Good, deg(0.0, 0.5, 101.0), now());
    let command = guide_velocity(
        &above,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &config,
    )
    .expect("guides");
    let velocity = velocity_of(command.setpoint);
    assert!(
        velocity.down_mps > 0.0,
        "above the profile must descend: {velocity:?}"
    );
    assert!(
        (velocity.down_mps - config.vertical_gain_per_s).abs() < 1e-9,
        "one meter above at the configured gain: {velocity:?}"
    );
    let below = solution(SolutionQuality::Good, deg(0.0, 0.5, 99.0), now());
    let command = guide_velocity(
        &below,
        LateralReference::track(from),
        &to,
        now(),
        CLOCK,
        &config,
    )
    .expect("guides");
    let velocity = velocity_of(command.setpoint);
    assert!(
        velocity.down_mps < 0.0,
        "below the profile must climb: {velocity:?}"
    );
    assert!((velocity.down_mps + config.vertical_gain_per_s).abs() < 1e-9);
}

#[test]
fn the_vertical_rate_is_capped_both_ways() {
    let (from, _) = equator_leg();
    let to = Waypoint::new("END".into(), deg(0.0, 1.0, 0.0))
        .with_altitude(AltitudeConstraint::At(100.0));
    let config = VelocityGuidanceConfig::default();
    for (altitude_m, expected_mps) in [
        (1_000.0, config.max_vertical_mps),
        (0.0, -config.max_vertical_mps),
    ] {
        let own = solution(SolutionQuality::Good, deg(0.0, 0.5, altitude_m), now());
        let command = guide_velocity(
            &own,
            LateralReference::track(from),
            &to,
            now(),
            CLOCK,
            &config,
        )
        .expect("guides");
        let velocity = velocity_of(command.setpoint);
        assert!(
            (velocity.down_mps - expected_mps).abs() < 1e-9,
            "at {altitude_m} m the rate must ride the cap: {velocity:?}"
        );
    }
}
