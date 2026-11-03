//! Rendezvous and proximity operations in the local-vertical
//! local-horizontal (LVLH) frame of a target.
//!
//! The Clohessy–Wiltshire equations give the relative motion near a
//! target on a circular orbit. In LVLH coordinates `(x, y, z)` along
//! `(R, T, N)` with the mean motion `n`:
//! `ẍ − 2nẏ − 3n²x = 0`, `ÿ + 2nẋ = 0`, and `z̈ + n²z = 0`.
//! [`cw_transfer_velocity`] solves the two-impulse transfer: the LVLH
//! velocity at the start that moves the chaser to a hold point in a
//! given time. [`evaluate_rendezvous`] solves it again in each cycle and
//! commands the matching inertial velocity, after the
//! [`ApproachCorridor`] check.

use nalgebra::{Matrix3, Vector3};

use crate::error::TrajectoryError;
use crate::float_math::{cos, round, sin};
use crate::state::StateVector;

mod corridor;
mod lvlh;

#[cfg(test)]
mod tests;

pub use corridor::ApproachCorridor;
pub use lvlh::LvlhFrame;

/// A time of flight closer than this to a whole number of half orbits,
/// in radians of orbit angle, has no stable transfer solution.
const SINGULAR_MARGIN_RAD: f64 = 1e-2;

/// The hold point, the corridor, and the floor on the time to go of one
/// approach.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct RendezvousTarget {
    /// The hold point in LVLH coordinates in meters.
    pub hold_point_lvlh_m: Vector3<f64>,
    /// The keep-out sphere and the approach cone.
    pub corridor: ApproachCorridor,
    /// The time to go in seconds never goes below this value. At the
    /// floor, the transfer becomes a station-keeping law around the
    /// hold point.
    pub min_time_to_go_s: f64,
}

impl RendezvousTarget {
    /// Builds a target from its parts.
    #[must_use]
    pub const fn new(
        hold_point_lvlh_m: Vector3<f64>,
        corridor: ApproachCorridor,
        min_time_to_go_s: f64,
    ) -> Self {
        Self {
            hold_point_lvlh_m,
            corridor,
            min_time_to_go_s,
        }
    }
}

/// The output of one evaluation of [`evaluate_rendezvous`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct RendezvousCommand {
    /// The commanded inertial velocity of the chaser in m/s.
    pub velocity_mps: Vector3<f64>,
    /// The commanded velocity minus the chaser velocity in m/s, in the
    /// inertial axes.
    pub delta_v_mps: Vector3<f64>,
    /// The chaser position relative to the target in LVLH coordinates
    /// in meters.
    pub relative_position_lvlh_m: Vector3<f64>,
    /// The commanded chaser velocity relative to the target in LVLH
    /// coordinates in m/s.
    pub relative_velocity_lvlh_mps: Vector3<f64>,
    /// The margin to the approach cone in meters, or `None` outside the
    /// keep-out sphere.
    pub corridor_margin_m: Option<f64>,
}

/// The LVLH velocity at `rho0` that moves the chaser to `rho_f` in
/// `time_of_flight_s` seconds under the Clohessy–Wiltshire equations
/// with the mean motion `mean_motion_rad_per_s`.
///
/// # Errors
///
/// [`TrajectoryError::InvalidParameter`] when the mean motion or the time
/// of flight is not finite and positive, and
/// [`TrajectoryError::SingularTransfer`] when the orbit angle of the
/// transfer is near a whole number of half orbits.
pub fn cw_transfer_velocity(
    rho0: &Vector3<f64>,
    rho_f: &Vector3<f64>,
    mean_motion_rad_per_s: f64,
    time_of_flight_s: f64,
) -> Result<Vector3<f64>, TrajectoryError> {
    let n = mean_motion_rad_per_s;
    if !(n.is_finite() && n > 0.0) {
        return Err(TrajectoryError::InvalidParameter {
            name: "mean_motion_rad_per_s",
        });
    }
    if !(time_of_flight_s.is_finite() && time_of_flight_s > 0.0) {
        return Err(TrajectoryError::InvalidParameter {
            name: "time_of_flight_s",
        });
    }
    let theta = n * time_of_flight_s;
    let half_orbits = round(theta / core::f64::consts::PI);
    let singular = TrajectoryError::SingularTransfer { time_of_flight_s };
    if half_orbits >= 1.0
        && (theta - half_orbits * core::f64::consts::PI).abs() < SINGULAR_MARGIN_RAD
    {
        return Err(singular);
    }
    let (s, c) = (sin(theta), cos(theta));
    #[rustfmt::skip]
    let phi_rr = Matrix3::new(
        4.0 - 3.0 * c, 0.0, 0.0,
        6.0 * (s - theta), 1.0, 0.0,
        0.0, 0.0, c,
    );
    #[rustfmt::skip]
    let phi_rv = Matrix3::new(
        s / n, 2.0 * (1.0 - c) / n, 0.0,
        2.0 * (c - 1.0) / n, (4.0 * s - 3.0 * theta) / n, 0.0,
        0.0, 0.0, s / n,
    );
    let inverse = phi_rv.try_inverse().ok_or(singular)?;
    Ok(inverse * (rho_f - phi_rr * rho0))
}

/// Evaluates the rendezvous law for a chaser at `chaser` near a target
/// at `target`, both in one inertial frame.
///
/// The law checks the corridor with the relative one-sigma position
/// uncertainty `position_1sigma_m`, then solves the transfer to the hold
/// point in `max(time_to_go_s, min_time_to_go_s)` seconds.
///
/// # Errors
///
/// The errors of [`LvlhFrame::from_target`], of
/// [`ApproachCorridor::check`], and of [`cw_transfer_velocity`], and
/// [`TrajectoryError::InvalidParameter`] when the hold point is outside
/// the corridor or the floor on the time to go is not positive.
pub fn evaluate_rendezvous(
    target: &StateVector,
    chaser: &StateVector,
    plan: &RendezvousTarget,
    time_to_go_s: f64,
    position_1sigma_m: f64,
) -> Result<RendezvousCommand, TrajectoryError> {
    if !(plan.min_time_to_go_s.is_finite() && plan.min_time_to_go_s > 0.0) {
        return Err(TrajectoryError::InvalidParameter {
            name: "min_time_to_go_s",
        });
    }
    if plan.corridor.check(&plan.hold_point_lvlh_m, 0.0).is_err() {
        return Err(TrajectoryError::InvalidParameter {
            name: "hold_point_lvlh_m",
        });
    }
    let frame = LvlhFrame::from_target(target)?;
    let chaser = chaser.checked()?;
    let relative = frame.relative_state(&chaser);
    let corridor_margin_m = plan
        .corridor
        .check(&relative.position_m, position_1sigma_m)?;
    let time_of_flight_s = time_to_go_s.max(plan.min_time_to_go_s);
    let rho_dot = cw_transfer_velocity(
        &relative.position_m,
        &plan.hold_point_lvlh_m,
        frame.angular_rate_rad_per_s(),
        time_of_flight_s,
    )?;
    let velocity_mps = frame.inertial_velocity(&relative.position_m, &rho_dot);
    Ok(RendezvousCommand {
        velocity_mps,
        delta_v_mps: velocity_mps - chaser.velocity_mps,
        relative_position_lvlh_m: relative.position_m,
        relative_velocity_lvlh_mps: rho_dot,
        corridor_margin_m,
    })
}
