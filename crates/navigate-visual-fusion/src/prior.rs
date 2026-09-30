//! The navigation solution as a visual search prior (ADR-0009).

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use navigate_contract::{
    ClockDomainId, DurationNanos, MonotonicNanos, NavigationSolution, SymmetricCov3,
};
use navigate_visual::{CameraPose, LocalFrame, SearchPrior};

use crate::VisualFusionError;

/// Capture time and camera attitude of the frame that the prior is for.
#[derive(Clone, Copy, Debug)]
pub struct FrameCapture {
    /// Capture time of the frame.
    pub at: MonotonicNanos,
    /// Clock domain of `at`. It must be the clock domain of the solution.
    pub clock: ClockDomainId,
    /// Known absolute error bound of the capture timestamp.
    pub time_error_bound: DurationNanos,
    /// Camera attitude in the reference frame, from the flight controller
    /// attitude and the camera mounting.
    pub orientation: UnitQuaternion<f64>,
    /// Conservative angular RMS allowance of `orientation`, in radians.
    /// This search allowance is not a calibrated probability or integrity bound.
    pub attitude_sigma_rad: f64,
}

/// Bounds on a constant-velocity search projection.
#[derive(Clone, Copy, Debug)]
pub struct PropagationBudget {
    /// Maximum time distance, including both timestamp error bounds.
    pub maximum_age: DurationNanos,
    /// Known absolute error bound of the navigation solution timestamp.
    pub solution_time_error_bound: DurationNanos,
    /// Upper RMS bound on unmodeled acceleration in any direction, in m/s².
    /// None means unknown and cannot produce a bounded search projection.
    pub acceleration_rms_bound_mps2: Option<f64>,
}

/// Project a navigation solution to capture time for candidate search.
///
/// Position follows constant velocity. The error envelope includes unknown
/// position/velocity correlation, timestamp error, and unmodeled acceleration.
/// The host supplies valid error budgets. This is not a fusion prediction or
/// an admission bound. The input position must refer to the camera centre, or
/// its error budget must cover the camera lever arm and reference datum.
///
/// # Errors
/// Refuses foreign clocks, excessive age, absent process error, and invalid
/// positions, velocities, or covariance matrices.
pub fn search_prior(
    solution: &NavigationSolution,
    frame: LocalFrame,
    capture: FrameCapture,
    budget: PropagationBudget,
) -> Result<SearchPrior, VisualFusionError> {
    if capture.clock != solution.stamp.clock {
        return Err(VisualFusionError::ClockDomainMismatch);
    }
    let (age, timing, acceleration) = projection_limits(solution, capture, budget)?;
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
    let position_cov = ned_to_enu(&solution.position_cov);
    let velocity_cov = ned_to_enu(&solution.velocity_cov);
    if !v.is_finite() || !valid_covariance(&position_cov) || !valid_covariance(&velocity_cov) {
        return Err(VisualFusionError::InvalidCovariance);
    }
    let mut covariance = correlated_sum(position_cov, velocity_cov * age * age);
    covariance = correlated_sum(
        covariance,
        velocity_enu * velocity_enu.transpose() * timing * timing,
    );
    let process_rms = 0.5 * acceleration * age * age;
    covariance = correlated_sum(covariance, Matrix3::identity() * process_rms * process_rms);
    if covariance.iter().any(|v| !v.is_finite()) {
        return Err(VisualFusionError::InvalidCovariance);
    }
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

fn projection_limits(
    solution: &NavigationSolution,
    capture: FrameCapture,
    budget: PropagationBudget,
) -> Result<(f64, f64, f64), VisualFusionError> {
    let acceleration = budget
        .acceleration_rms_bound_mps2
        .filter(|v| v.is_finite() && *v > 0.0)
        .ok_or(VisualFusionError::InvalidBudget {
            field: "projection acceleration RMS bound",
        })?;
    let timing_ns = capture
        .time_error_bound
        .as_nanos()
        .checked_add(budget.solution_time_error_bound.as_nanos())
        .ok_or(VisualFusionError::InvalidBudget {
            field: "projection timestamp errors",
        })?;
    let offset = capture
        .at
        .as_nanos()
        .abs_diff(solution.stamp.solved_at.as_nanos());
    let age_ns = offset
        .checked_add(timing_ns)
        .ok_or(VisualFusionError::InvalidBudget {
            field: "projection time distance",
        })?;
    if age_ns > budget.maximum_age.as_nanos() || budget.maximum_age.as_nanos() == 0 {
        return Err(VisualFusionError::ProjectionWindow {
            age_ns,
            maximum_ns: budget.maximum_age.as_nanos(),
        });
    }
    Ok((age_ns as f64 * 1e-9, timing_ns as f64 * 1e-9, acceleration))
}

fn valid_covariance(value: &Matrix3<f64>) -> bool {
    value.iter().all(|v| v.is_finite())
        && value
            .symmetric_eigen()
            .eigenvalues
            .iter()
            .all(|v| *v >= 0.0)
}

fn correlated_sum(a: Matrix3<f64>, b: Matrix3<f64>) -> Matrix3<f64> {
    let (sa, sb) = (a.trace().sqrt(), b.trace().sqrt());
    if sa == 0.0 {
        return b;
    }
    if sb == 0.0 {
        return a;
    }
    // Weighted Cauchy-Schwarz bounds the sum for every valid cross covariance.
    a * (1.0 + sb / sa) + b * (1.0 + sa / sb)
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
