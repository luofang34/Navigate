//! E-guidance: the sign of the law, the hover, the throttle clamp, the
//! capture, the glide slope, and a closed-loop descent on a flat body.

use nalgebra::Vector3;

use super::{GlideSlope, PoweredDescentTarget, evaluate_powered_descent};
use crate::error::TrajectoryError;
use crate::state::StateVector;

const RADIUS_M: f64 = 1_737_400.0;

fn surface_target() -> PoweredDescentTarget {
    PoweredDescentTarget::new(
        Vector3::new(0.0, 0.0, RADIUS_M),
        Vector3::zeros(),
        Vector3::zeros(),
        0.0,
        10.0,
    )
}

fn lunar_gravity() -> Vector3<f64> {
    Vector3::new(0.0, 0.0, -1.62)
}

#[test]
fn a_vehicle_short_of_the_target_accelerates_toward_it() {
    let target = surface_target();
    let state = StateVector::new(Vector3::new(-100.0, 0.0, RADIUS_M), Vector3::zeros());
    let command = evaluate_powered_descent(&target, &state, 20.0, &Vector3::zeros()).unwrap();
    assert!((command.direction_unit - Vector3::x()).norm() < 1e-12);
    assert!((command.specific_force_mps2 - 12.0 * 100.0 / 400.0).abs() < 1e-9);
}

#[test]
fn at_the_target_the_thrust_holds_a_hover() {
    let target = surface_target().with_capture(0.0, 0.0);
    let state = StateVector::new(
        target.position_m + Vector3::new(1e-3, 0.0, 0.0),
        Vector3::zeros(),
    );
    let command = evaluate_powered_descent(&target, &state, 10.0, &lunar_gravity()).unwrap();
    assert!(!command.cutoff);
    assert!((command.direction_unit - Vector3::z()).norm() < 1e-3);
    assert!((command.specific_force_mps2 - 1.62).abs() < 1e-6);
}

#[test]
fn the_specific_force_stays_in_the_throttle_range() {
    let target = surface_target();
    let far = StateVector::new(Vector3::new(-50e3, 0.0, RADIUS_M + 5e3), Vector3::zeros());
    let command = evaluate_powered_descent(&target, &far, 10.0, &lunar_gravity()).unwrap();
    assert_eq!(command.specific_force_mps2, 10.0);
    let mut low = surface_target();
    low.min_specific_force_mps2 = 2.0;
    let state = StateVector::new(low.position_m, Vector3::zeros());
    let command = evaluate_powered_descent(&low, &state, 10.0, &Vector3::zeros()).unwrap();
    assert!(command.cutoff);
    let state = StateVector::new(
        low.position_m + Vector3::new(0.0, 0.0, 10.0),
        Vector3::zeros(),
    );
    let command = evaluate_powered_descent(&low, &state, 10.0, &Vector3::zeros()).unwrap();
    assert_eq!(command.specific_force_mps2, 2.0);
}

#[test]
fn the_engine_stops_inside_the_capture_box() {
    let target = surface_target();
    let state = StateVector::new(
        target.position_m + Vector3::new(0.1, 0.0, 0.3),
        Vector3::new(0.0, 0.0, -0.2),
    );
    let command = evaluate_powered_descent(&target, &state, 0.0, &lunar_gravity()).unwrap();
    assert!(command.cutoff);
    assert_eq!(command.specific_force_mps2, 0.0);
    assert!((command.direction_unit - Vector3::z()).norm() < 1e-12);
}

#[test]
fn a_vehicle_below_the_glide_slope_is_refused() {
    let cone = GlideSlope::new(10f64.to_radians(), 5_000.0);
    let target = surface_target().with_glide_slope(cone);
    let low = StateVector::new(
        Vector3::new(-2_000.0, 0.0, RADIUS_M + 100.0),
        Vector3::zeros(),
    );
    assert!(matches!(
        evaluate_powered_descent(&target, &low, 60.0, &lunar_gravity()),
        Err(TrajectoryError::GlideSlopeViolated { elevation_rad, .. })
            if (elevation_rad - (100f64 / 2_000.0).atan()).abs() < 1e-6
    ));
    let high = StateVector::new(
        Vector3::new(-2_000.0, 0.0, RADIUS_M + 500.0),
        Vector3::zeros(),
    );
    assert!(evaluate_powered_descent(&target, &high, 60.0, &lunar_gravity()).is_ok());
    let far = StateVector::new(Vector3::new(-20e3, 0.0, RADIUS_M + 100.0), Vector3::zeros());
    assert!(evaluate_powered_descent(&target, &far, 60.0, &lunar_gravity()).is_ok());
}

#[test]
fn a_closed_loop_descent_captures_the_target() {
    let target = surface_target();
    let mut state = StateVector::new(
        Vector3::new(-1_500.0, 0.0, RADIUS_M + 2_000.0),
        Vector3::new(40.0, 0.0, -20.0),
    );
    let dt = 0.05;
    let mut captured = None;
    for step in 0..2_400 {
        let time_to_go_s = 100.0 - f64::from(step) * dt;
        let command =
            evaluate_powered_descent(&target, &state, time_to_go_s, &lunar_gravity()).unwrap();
        if command.cutoff {
            captured = Some(state);
            break;
        }
        let a = command.direction_unit * command.specific_force_mps2 + lunar_gravity();
        state.position_m += state.velocity_mps * dt + a * (0.5 * dt * dt);
        state.velocity_mps += a * dt;
    }
    let captured = captured.expect("the descent must reach the capture box");
    assert!((captured.position_m - target.position_m).norm() < 1.0);
    assert!(captured.velocity_mps.norm() < 0.5);
}

#[test]
fn invalid_values_are_refused_by_name() {
    let mut target = surface_target();
    target.max_specific_force_mps2 = -1.0;
    let state = StateVector::new(Vector3::new(0.0, 0.0, RADIUS_M + 10.0), Vector3::zeros());
    assert_eq!(
        evaluate_powered_descent(&target, &state, 10.0, &lunar_gravity()),
        Err(TrajectoryError::InvalidParameter {
            name: "max_specific_force_mps2"
        })
    );
    assert_eq!(
        evaluate_powered_descent(&surface_target(), &state, f64::NAN, &lunar_gravity()),
        Err(TrajectoryError::InvalidParameter {
            name: "time_to_go_s"
        })
    );
}
