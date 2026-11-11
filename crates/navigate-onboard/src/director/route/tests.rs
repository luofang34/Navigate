use super::*;
use crate::Navigator;
use crate::SequenceCause;
use crate::reference::tests::{CLOCK, aligned};
use navigate_contract::{AltitudeConstraint, GeodeticPosition, GuidanceSetpoint, TurnType};
use navigate_guidance::{GuidanceConfig, VelocityGuidanceConfig};

const LAT: f64 = 41.7;
const HEIGHT: f64 = 1_000.0;

#[cfg(test)]
fn waypoints() -> Vec<Waypoint> {
    let point = |ident: &str, lat: f64, lon: f64, speed: Option<f64>| {
        let w = Waypoint::new(
            Ident::try_new(ident).unwrap(),
            GeodeticPosition::new(lat.to_radians(), lon.to_radians(), HEIGHT),
        )
        .with_altitude(AltitudeConstraint::At(HEIGHT))
        .with_turn(TurnType::FlyOver);
        match speed {
            Some(s) => w.with_max_speed(s),
            None => w,
        }
    };
    vec![
        point("A", LAT, -0.99, None),
        point("B", LAT, -0.96, Some(30.0)),
        point("C", LAT + 0.02, -0.96, None),
    ]
}

#[cfg(test)]
fn director(waypoints: &[Waypoint], law: GuidanceLaw) -> RouteDirector {
    let config = ExecutionConfig::new(100.0).with_surface_gravity_mps2(9.806_65);
    RouteDirector::new("TRIP", waypoints.to_vec(), config, law, 40.0, CLOCK).unwrap()
}

#[cfg(test)]
fn tracking() -> GuidanceLaw {
    GuidanceLaw::DeviationTracking(
        GuidanceConfig::default().with_expected_ellipsoid(Ellipsoid::WGS84),
    )
}

#[test]
fn the_director_sequences_each_waypoint_on_the_solution_and_completes() {
    let waypoints = waypoints();
    let mut director = director(&waypoints, tracking());
    let mut navigator = aligned(&[]);
    let now = MonotonicNanos::from_nanos(1_000_000_000);
    let mut solution = navigator.solution(now).unwrap();
    assert_eq!(director.step(None, now).setpoint, None);
    let mut events = Vec::new();
    for w in &waypoints {
        let output = director.step(Some(&solution), now);
        assert!(output.setpoint.is_some(), "{output:?}");
        solution.position = w.position;
        let output = director.step(Some(&solution), now);
        events.extend(output.event);
    }
    let indices: Vec<_> = events
        .iter()
        .map(|e| (e.waypoint_index, e.plan_complete))
        .collect();
    assert_eq!(indices, [(0, false), (1, false), (2, true)]);
    assert_eq!(director.active_index(), None);
    assert_eq!(director.step(Some(&solution), now).active_index, None);
}

#[test]
fn the_commanded_speed_follows_the_waypoint_limit() {
    let waypoints = waypoints();
    let director = director(&waypoints, tracking());
    assert_eq!(director.leg_speed(0), 40.0);
    assert_eq!(director.leg_speed(1), 30.0);
    assert_eq!(director.leg_speed(9), 40.0);
}

#[test]
fn the_velocity_law_gives_a_velocity_setpoint() {
    let waypoints = waypoints();
    let law = GuidanceLaw::Velocity(VelocityGuidanceConfig::default());
    let mut director = director(&waypoints, law);
    let mut navigator = aligned(&[]);
    let now = MonotonicNanos::from_nanos(1_000_000_000);
    let solution = navigator.solution(now).unwrap();
    let output = director.step(Some(&solution), now);
    assert!(
        matches!(output.setpoint, Some(GuidanceSetpoint::Velocity { .. })),
        "{output:?}"
    );
}

#[test]
fn unusable_routes_are_refused() {
    let waypoints = waypoints();
    let config = ExecutionConfig::new(100.0);
    assert!(RouteDirector::new("TRIP", waypoints.clone(), config, tracking(), 0.0, CLOCK).is_err());
    assert!(RouteDirector::new("", waypoints.clone(), config, tracking(), 40.0, CLOCK).is_err());
    assert!(RouteDirector::new("TRIP", Vec::new(), config, tracking(), 40.0, CLOCK).is_err());
}

#[test]
fn a_director_snapshot_restores_the_sequencing_state() {
    let waypoints = waypoints();
    let mut director = director(&waypoints, tracking());
    let mut navigator = aligned(&[]);
    let now = MonotonicNanos::from_nanos(1_000_000_000);
    let mut solution = navigator.solution(now).unwrap();
    let saved = director.snapshot();
    solution.position = waypoints[0].position;
    director.step(Some(&solution), now);
    assert_eq!(director.active_index(), Some(1));
    director.restore(&*saved).unwrap();
    assert_eq!(director.active_index(), Some(0));
}

#[test]
fn a_fix_passed_outside_the_capture_radius_is_a_passed_event() {
    let waypoints = waypoints();
    let mut director = director(&waypoints, tracking());
    let mut navigator = aligned(&[]);
    let now = MonotonicNanos::from_nanos(1_000_000_000);
    let mut solution = navigator.solution(now).unwrap();
    solution.position = waypoints[0].position;
    director.step(Some(&solution), now);
    // Past B (lon -0.96) along the A→B course, 300 m south of it: outside
    // the 100 m capture radius and beyond the perpendicular through B.
    let b = waypoints[1].position;
    solution.position = GeodeticPosition::new(
        b.latitude_rad - 300.0 / 6_364_000.0,
        b.longitude_rad + 0.0005_f64.to_radians(),
        HEIGHT,
    );
    let output = director.step(Some(&solution), now);
    let event = output.event.unwrap();
    assert_eq!(event.waypoint_index, 1);
    assert!(
        matches!(event.cause, SequenceCause::Passed { cross_track_m } if (cross_track_m - 300.0).abs() < 20.0),
        "{event:?}"
    );
}
