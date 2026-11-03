//! The LVLH frame, the Clohessy–Wiltshire transfer, the corridor, and a
//! closed-loop approach on two-body orbits.

use nalgebra::Vector3;

use super::{
    ApproachCorridor, LvlhFrame, RendezvousTarget, cw_transfer_velocity, evaluate_rendezvous,
};
use crate::error::TrajectoryError;
use crate::state::StateVector;

const EARTH_GM: f64 = 3.986_004_418e14;
const ORBIT_RADIUS_M: f64 = 6_378_137.0 + 400e3;

fn mean_motion() -> f64 {
    (EARTH_GM / ORBIT_RADIUS_M.powi(3)).sqrt()
}

fn target_state() -> StateVector {
    StateVector::new(
        Vector3::new(ORBIT_RADIUS_M, 0.0, 0.0),
        Vector3::new(0.0, (EARTH_GM / ORBIT_RADIUS_M).sqrt(), 0.0),
    )
}

fn v_bar_corridor() -> ApproachCorridor {
    ApproachCorridor::new(Vector3::new(0.0, -1.0, 0.0), 15f64.to_radians(), 200.0, 3.0)
}

/// The Clohessy–Wiltshire acceleration in LVLH coordinates.
fn cw_acceleration(rho: &Vector3<f64>, rho_dot: &Vector3<f64>, n: f64) -> Vector3<f64> {
    Vector3::new(
        2.0 * n * rho_dot.y + 3.0 * n * n * rho.x,
        -2.0 * n * rho_dot.x,
        -n * n * rho.z,
    )
}

fn two_body(p: &Vector3<f64>) -> Vector3<f64> {
    -p * (EARTH_GM / p.norm().powi(3))
}

/// One fourth-order Runge–Kutta step of `acc`.
fn rk4(
    state: &StateVector,
    dt: f64,
    acc: impl Fn(&Vector3<f64>, &Vector3<f64>) -> Vector3<f64>,
) -> StateVector {
    let (p, v) = (state.position_m, state.velocity_mps);
    let k1 = (v, acc(&p, &v));
    let k2 = (
        v + k1.1 * (dt / 2.0),
        acc(&(p + k1.0 * (dt / 2.0)), &(v + k1.1 * (dt / 2.0))),
    );
    let k3 = (
        v + k2.1 * (dt / 2.0),
        acc(&(p + k2.0 * (dt / 2.0)), &(v + k2.1 * (dt / 2.0))),
    );
    let k4 = (v + k3.1 * dt, acc(&(p + k3.0 * dt), &(v + k3.1 * dt)));
    StateVector::new(
        p + (k1.0 + k2.0 * 2.0 + k3.0 * 2.0 + k4.0) * (dt / 6.0),
        v + (k1.1 + k2.1 * 2.0 + k3.1 * 2.0 + k4.1) * (dt / 6.0),
    )
}

#[test]
fn the_transfer_velocity_reaches_the_hold_point_under_the_cw_equations() {
    let n = mean_motion();
    let rho0 = Vector3::new(120.0, -1_000.0, 40.0);
    let rho_f = Vector3::new(0.0, -30.0, 0.0);
    let rho_dot = cw_transfer_velocity(&rho0, &rho_f, n, 900.0).unwrap();
    let mut state = StateVector::new(rho0, rho_dot);
    for _ in 0..1_800 {
        state = rk4(&state, 0.5, |p, v| cw_acceleration(p, v, n));
    }
    assert!(
        (state.position_m - rho_f).norm() < 1e-3,
        "{:?}",
        state.position_m
    );
}

#[test]
fn a_transfer_of_a_whole_number_of_half_orbits_is_singular() {
    let n = mean_motion();
    let rho0 = Vector3::new(0.0, -1_000.0, 0.0);
    for half_orbits in [1.0, 2.0] {
        let time_of_flight_s = half_orbits * core::f64::consts::PI / n;
        assert_eq!(
            cw_transfer_velocity(&rho0, &Vector3::zeros(), n, time_of_flight_s),
            Err(TrajectoryError::SingularTransfer { time_of_flight_s })
        );
    }
    assert!(matches!(
        cw_transfer_velocity(&rho0, &Vector3::zeros(), n, -1.0),
        Err(TrajectoryError::InvalidParameter { .. })
    ));
}

#[test]
fn the_lvlh_frame_gives_back_the_inertial_velocity() {
    let target = target_state();
    let frame = LvlhFrame::from_target(&target).unwrap();
    let [radial, along, normal] = frame.axes();
    assert!((radial - Vector3::x()).norm() < 1e-12);
    assert!((along - Vector3::y()).norm() < 1e-12);
    assert!((normal - Vector3::z()).norm() < 1e-12);
    assert!((frame.angular_rate_rad_per_s() - mean_motion()).abs() < 1e-15);
    let chaser = StateVector::new(
        target.position_m + Vector3::new(10.0, -100.0, 5.0),
        target.velocity_mps + Vector3::new(0.1, 0.2, -0.3),
    );
    let relative = frame.relative_state(&chaser);
    assert!((relative.position_m - Vector3::new(10.0, -100.0, 5.0)).norm() < 1e-6);
    let back = frame.inertial_velocity(&relative.position_m, &relative.velocity_mps);
    assert!((back - chaser.velocity_mps).norm() < 1e-9);
}

#[test]
fn the_corridor_refuses_outside_the_cone_and_for_a_wide_uncertainty() {
    let corridor = v_bar_corridor();
    assert_eq!(
        corridor.check(&Vector3::new(0.0, -500.0, 300.0), 1.0),
        Ok(None)
    );
    let margin = corridor
        .check(&Vector3::new(0.0, -100.0, 0.0), 1.0)
        .unwrap()
        .unwrap();
    assert!((margin - 100.0 * 15f64.to_radians().sin()).abs() < 1e-9);
    assert!(matches!(
        corridor.check(&Vector3::new(60.0, -100.0, 0.0), 1.0),
        Err(TrajectoryError::OutsideApproachCorridor { margin_m }) if margin_m < 0.0
    ));
    assert!(matches!(
        corridor.check(&Vector3::new(0.0, -100.0, 0.0), 10.0),
        Err(TrajectoryError::UncertaintyExceedsCorridorMargin { scaled_1sigma_m, .. })
            if (scaled_1sigma_m - 30.0).abs() < 1e-12
    ));
    assert!(matches!(
        corridor.check(&Vector3::new(0.0, -100.0, 0.0), f64::NAN),
        Err(TrajectoryError::InvalidParameter {
            name: "position_1sigma_m"
        })
    ));
}

/// Flies the chaser for `duration_s` seconds toward `plan`, arriving
/// `arrival_s` seconds after the start, with an ideal velocity tracker.
/// The result is the final states and the smallest corridor margin.
fn fly(
    mut target: StateVector,
    mut chaser: StateVector,
    plan: &RendezvousTarget,
    arrival_s: f64,
    duration_s: u32,
) -> Result<(StateVector, StateVector, f64), TrajectoryError> {
    let mut min_margin = f64::INFINITY;
    for step in 0..duration_s {
        let time_to_go_s = arrival_s - f64::from(step);
        let command = evaluate_rendezvous(&target, &chaser, plan, time_to_go_s, 0.5)?;
        if let Some(margin) = command.corridor_margin_m {
            min_margin = min_margin.min(margin);
        }
        chaser.velocity_mps = command.velocity_mps;
        chaser = rk4(&chaser, 1.0, |p, _| two_body(p));
        target = rk4(&target, 1.0, |p, _| two_body(p));
    }
    Ok((target, chaser, min_margin))
}

/// A chaser 1 km behind the target on the V-bar, at rest in LVLH.
fn chaser_behind(target: &StateVector) -> StateVector {
    let frame = LvlhFrame::from_target(target).unwrap();
    let rho0 = Vector3::new(0.0, -1_000.0, 0.0);
    StateVector::new(
        target.position_m + frame.axes()[1] * -1_000.0,
        frame.inertial_velocity(&rho0, &Vector3::zeros()),
    )
}

#[test]
fn a_direct_transfer_through_the_keep_out_sphere_is_refused() {
    let target = target_state();
    let plan = RendezvousTarget::new(Vector3::new(0.0, -30.0, 0.0), v_bar_corridor(), 60.0);
    let refused = fly(target, chaser_behind(&target), &plan, 1_200.0, 1_500);
    assert!(matches!(
        refused,
        Err(TrajectoryError::OutsideApproachCorridor { margin_m }) if margin_m < 0.0
    ));
}

#[test]
fn a_two_stage_approach_stays_in_the_corridor_and_holds_on_the_v_bar() {
    let target = target_state();
    let entry = RendezvousTarget::new(Vector3::new(0.0, -250.0, 0.0), v_bar_corridor(), 60.0);
    let (target, chaser, _) = fly(target, chaser_behind(&target), &entry, 1_000.0, 1_100).unwrap();
    let hold = RendezvousTarget::new(Vector3::new(0.0, -30.0, 0.0), v_bar_corridor(), 60.0);
    let (target, chaser, min_margin) = fly(target, chaser, &hold, 300.0, 600).unwrap();
    let relative = LvlhFrame::from_target(&target)
        .unwrap()
        .relative_state(&chaser);
    assert!(
        (relative.position_m - hold.hold_point_lvlh_m).norm() < 0.5,
        "{:?}",
        relative.position_m
    );
    assert!(relative.velocity_mps.norm() < 0.01);
    assert!(min_margin.is_finite() && min_margin > 1.5, "{min_margin}");
}
