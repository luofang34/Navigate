//! The navigation solution as a visual search prior (ADR-0009).

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use navigate_contract::{ClockDomainId, MonotonicNanos, NavigationSolution, SymmetricCov3};
use navigate_visual::{CameraPose, LocalFrame, SearchPrior};

use crate::VisualFusionError;

/// Capture time and camera attitude of the frame that the prior is for.
#[derive(Clone, Copy, Debug)]
pub struct FrameCapture {
    /// Capture time of the frame.
    pub at: MonotonicNanos,
    /// Clock domain of `at`. It must be the clock domain of the solution.
    pub clock: ClockDomainId,
    /// Camera attitude in the reference frame, from the flight controller
    /// attitude and the camera mounting.
    pub orientation: UnitQuaternion<f64>,
    /// One-sigma uncertainty of `orientation`, in radians.
    pub attitude_sigma_rad: f64,
}

/// Project a navigation solution to a frame's capture time as a search prior.
///
/// Position moves with the solution velocity. Position covariance grows with
/// the velocity covariance over the time step. The prior only narrows the
/// search. It is never an admission bound.
///
/// # Errors
/// Refuses a foreign clock domain, a position outside the frame, and an
/// unusable covariance.
pub fn search_prior(
    solution: &NavigationSolution,
    frame: LocalFrame,
    capture: FrameCapture,
) -> Result<SearchPrior, VisualFusionError> {
    if capture.clock != solution.stamp.clock {
        return Err(VisualFusionError::ClockDomainMismatch);
    }
    let dt = signed_seconds(capture.at, solution.stamp.solved_at);
    let p = solution.position;
    let origin = frame
        .local([
            p.latitude_rad.to_degrees(),
            p.longitude_rad.to_degrees(),
            p.altitude_m,
        ])
        .map_err(|_| VisualFusionError::ImplausiblePosition)?;
    let v = solution.velocity;
    let velocity_enu = Vector3::new(v.east_mps, v.north_mps, -v.down_mps);
    let covariance =
        ned_to_enu(&solution.position_cov) + ned_to_enu(&solution.velocity_cov) * dt * dt;
    let prior = SearchPrior {
        center: CameraPose {
            position: origin + velocity_enu * dt,
            orientation: capture.orientation,
        },
        position_covariance_m2: covariance,
        attitude_sigma_rad: capture.attitude_sigma_rad,
    };
    prior
        .validate()
        .map_err(|_| VisualFusionError::InvalidCovariance)?;
    Ok(prior)
}

fn signed_seconds(later: MonotonicNanos, earlier: MonotonicNanos) -> f64 {
    let (a, b) = (later.as_nanos(), earlier.as_nanos());
    if a >= b {
        (a - b) as f64 * 1e-9
    } else {
        -((b - a) as f64 * 1e-9)
    }
}

fn ned_to_enu(covariance: &SymmetricCov3) -> Matrix3<f64> {
    let [xx, xy, xz, yy, yz, zz] = covariance.upper_triangle();
    let ned = Matrix3::new(xx, xy, xz, xy, yy, yz, xz, yz, zz);
    let axes = Matrix3::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0);
    axes * ned * axes.transpose()
}

#[cfg(test)]
mod tests;
