//! Burn steering: the velocity to go, the cutoff, and a closed-loop
//! deorbit burn on a two-body orbit.

use nalgebra::Vector3;
use navigate_contract::{FrameId, MonotonicNanos};

use super::{BurnArc, BurnTarget, RadialBranch, evaluate_burn};
use crate::error::TrajectoryError;
use crate::state::StateVector;

const EARTH_GM: f64 = 3.986_004_418e14;
const EARTH_RADIUS_M: f64 = 6_378_137.0;

fn arc(target: BurnTarget) -> BurnArc {
    BurnArc::new(
        FrameId::EARTH_J2000,
        target,
        MonotonicNanos::from_nanos(0),
        MonotonicNanos::from_nanos(10),
        0.2,
        0.6,
    )
}

fn circular(radius_m: f64) -> StateVector {
    let speed = (EARTH_GM / radius_m).sqrt();
    StateVector::new(
        Vector3::new(radius_m, 0.0, 0.0),
        Vector3::new(0.0, speed, 0.0),
    )
}

/// The periapsis radius of the osculating conic of `state`.
fn periapsis_radius_m(state: &StateVector) -> f64 {
    let r = state.position_m.norm();
    let v = state.velocity_mps.norm();
    let energy = 0.5 * v * v - EARTH_GM / r;
    let a = -EARTH_GM / (2.0 * energy);
    let h = state.position_m.cross(&state.velocity_mps).norm();
    let e = (1.0 - h * h / (EARTH_GM * a)).max(0.0).sqrt();
    a * (1.0 - e)
}

fn acceleration(position: &Vector3<f64>, thrust: &Vector3<f64>) -> Vector3<f64> {
    let r = position.norm();
    -position * (EARTH_GM / (r * r * r)) + thrust
}

/// One fourth-order Runge–Kutta step with a constant thrust
/// acceleration.
fn rk4(state: &StateVector, thrust: &Vector3<f64>, dt: f64) -> StateVector {
    let (p, v) = (state.position_m, state.velocity_mps);
    let k1v = acceleration(&p, thrust);
    let k1p = v;
    let k2v = acceleration(&(p + k1p * (dt / 2.0)), thrust);
    let k2p = v + k1v * (dt / 2.0);
    let k3v = acceleration(&(p + k2p * (dt / 2.0)), thrust);
    let k3p = v + k2v * (dt / 2.0);
    let k4v = acceleration(&(p + k3p * dt), thrust);
    let k4p = v + k3v * dt;
    StateVector::new(
        p + (k1p + k2p * 2.0 + k3p * 2.0 + k4p) * (dt / 6.0),
        v + (k1v + k2v * 2.0 + k3v * 2.0 + k4v) * (dt / 6.0),
    )
}

#[test]
fn a_fixed_target_thrusts_along_the_velocity_to_go() {
    let state = circular(EARTH_RADIUS_M + 400e3);
    let required = state.velocity_mps + Vector3::new(0.0, -30.0, 40.0);
    let command = evaluate_burn(
        &arc(BurnTarget::Velocity {
            velocity_mps: required,
        }),
        &state,
    )
    .unwrap();
    assert!(!command.cutoff);
    assert!((command.specific_force_mps2 - 0.6).abs() < 1e-12);
    assert!((command.velocity_to_go_mps.norm() - 50.0).abs() < 1e-9);
    assert!((command.direction_unit - Vector3::new(0.0, -0.6, 0.8)).norm() < 1e-12);
}

#[test]
fn the_engine_stops_at_the_cutoff_threshold() {
    let state = circular(EARTH_RADIUS_M + 400e3);
    let command = evaluate_burn(
        &arc(BurnTarget::DeltaV {
            ignition_velocity_mps: state.velocity_mps,
            delta_v_mps: Vector3::new(0.1, 0.0, 0.0),
        }),
        &state,
    )
    .unwrap();
    assert!(command.cutoff);
    assert_eq!(command.specific_force_mps2, 0.0);
    assert!((command.direction_unit - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-12);
}

#[test]
fn invalid_values_are_refused_by_name() {
    let state = circular(EARTH_RADIUS_M + 400e3);
    let mut bad = arc(BurnTarget::Velocity {
        velocity_mps: Vector3::zeros(),
    });
    bad.max_specific_force_mps2 = f64::NAN;
    assert_eq!(
        evaluate_burn(&bad, &state),
        Err(TrajectoryError::InvalidParameter {
            name: "max_specific_force_mps2"
        })
    );
    let mut late = bad;
    late.max_specific_force_mps2 = 0.6;
    late.ignition_close = MonotonicNanos::from_nanos(0);
    late.ignition_open = MonotonicNanos::from_nanos(1);
    assert_eq!(
        late.validate(),
        Err(TrajectoryError::InvalidParameter {
            name: "ignition_close"
        })
    );
    let nan_state = StateVector::new(Vector3::new(f64::NAN, 0.0, 0.0), Vector3::zeros());
    assert_eq!(
        evaluate_burn(&late, &nan_state),
        Err(TrajectoryError::InvalidParameter {
            name: "ignition_close"
        })
    );
}

#[test]
fn a_radius_below_the_target_periapsis_is_unreachable() {
    let state = circular(EARTH_RADIUS_M + 100e3);
    let target = BurnTarget::Apsides {
        periapsis_radius_m: EARTH_RADIUS_M + 200e3,
        apoapsis_radius_m: EARTH_RADIUS_M + 400e3,
        gm_m3_per_s2: EARTH_GM,
        branch: RadialBranch::Descending,
    };
    assert!(matches!(
        evaluate_burn(&arc(target), &state),
        Err(TrajectoryError::ApsidesUnreachable { .. })
    ));
}

#[test]
fn a_closed_loop_deorbit_burn_reaches_the_target_periapsis() {
    let apoapsis = EARTH_RADIUS_M + 400e3;
    let periapsis = EARTH_RADIUS_M + 30e3;
    let target = BurnTarget::Apsides {
        periapsis_radius_m: periapsis,
        apoapsis_radius_m: apoapsis,
        gm_m3_per_s2: EARTH_GM,
        branch: RadialBranch::Descending,
    };
    let burn = arc(target);
    let mut state = circular(apoapsis);
    let dt = 0.5;
    let mut delta_v = 0.0;
    let mut cutoff = false;
    for _ in 0..2_000 {
        let command = evaluate_burn(&burn, &state).unwrap();
        if command.cutoff {
            cutoff = true;
            break;
        }
        let thrust = command.direction_unit * command.specific_force_mps2;
        state = rk4(&state, &thrust, dt);
        delta_v += command.specific_force_mps2 * dt;
    }
    assert!(cutoff, "the burn must reach cutoff");
    let ideal = (EARTH_GM / apoapsis).sqrt()
        - (EARTH_GM * 2.0 * periapsis / (apoapsis * (apoapsis + periapsis))).sqrt();
    let achieved = periapsis_radius_m(&state);
    assert!(
        (achieved - periapsis).abs() < 2_000.0,
        "periapsis {achieved} m"
    );
    assert!(
        delta_v < ideal * 1.02,
        "delta-v {delta_v} m/s, ideal {ideal} m/s"
    );
    assert!(state.position_m.dot(&state.velocity_mps) < 0.0);
}
