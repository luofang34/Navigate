//! Polynomial E-guidance of a planetary powered descent.
//!
//! The law fits a quadratic acceleration profile in time that meets the
//! target position `p_f`, velocity `v_f`, and acceleration `a_f` at the
//! time to go `t`. From the state `(p, v)`, the commanded acceleration at
//! the start of the profile is
//!
//! `a_cmd = 12·(p_f − p − v·t)/t² − 6·(v_f − v)/t + a_f`.
//!
//! The thrust acceleration is `a_cmd − g(p)`, with its magnitude clamped
//! to the throttle range. The law solves the fit again in each cycle.
//! Below a floor on `t`, the floor value is used, so that the gains stay
//! bounded and the law becomes a damped regulator about the target.

use nalgebra::Vector3;

use crate::error::TrajectoryError;
use crate::float_math::asin;
use crate::state::{StateVector, unit};

#[cfg(test)]
mod tests;

/// The glide-slope cone of a landing target.
///
/// Inside the slant range [`Self::max_range_m`], the elevation of the
/// vehicle above the horizon of the target must be
/// [`Self::min_elevation_rad`] or more. Farther away, the curvature of
/// the body puts the vehicle below that horizon, and the cone does not
/// apply.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct GlideSlope {
    /// The minimum elevation in radians.
    pub min_elevation_rad: f64,
    /// The slant range in meters inside which the cone applies.
    pub max_range_m: f64,
}

impl GlideSlope {
    /// A cone that never applies.
    pub const NONE: Self = Self::new(0.0, 0.0);

    /// Builds a cone from its parts.
    #[must_use]
    pub const fn new(min_elevation_rad: f64, max_range_m: f64) -> Self {
        Self {
            min_elevation_rad,
            max_range_m,
        }
    }
}

/// The touchdown target, the throttle range, and the capture box of one
/// powered descent, in one body-fixed frame.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PoweredDescentTarget {
    /// The target position `p_f` in meters.
    pub position_m: Vector3<f64>,
    /// The target velocity `v_f` in m/s.
    pub velocity_mps: Vector3<f64>,
    /// The target total acceleration `a_f` in m/s². Zero holds the
    /// vehicle in a hover at the target.
    pub acceleration_mps2: Vector3<f64>,
    /// The minimum specific force of the engine in m/s².
    pub min_specific_force_mps2: f64,
    /// The maximum specific force of the engine in m/s².
    pub max_specific_force_mps2: f64,
    /// The glide-slope cone.
    pub glide_slope: GlideSlope,
    /// The floor on the time to go in seconds.
    pub min_time_to_go_s: f64,
    /// The capture radius in meters. The engine stops when the distance
    /// to the target is this value or less and the velocity error is
    /// [`Self::capture_speed_mps`] or less. The glide slope does not
    /// apply inside it.
    pub capture_radius_m: f64,
    /// The capture speed in m/s. See [`Self::capture_radius_m`].
    pub capture_speed_mps: f64,
}

impl PoweredDescentTarget {
    /// The default floor on the time to go in seconds.
    pub const DEFAULT_MIN_TIME_TO_GO_S: f64 = 3.0;
    /// The default capture radius in meters.
    pub const DEFAULT_CAPTURE_RADIUS_M: f64 = 0.5;
    /// The default capture speed in m/s.
    pub const DEFAULT_CAPTURE_SPEED_MPS: f64 = 0.3;

    /// A target at `position_m` with `velocity_mps` and
    /// `acceleration_mps2`, for an engine with the given specific force
    /// range in m/s². The glide slope is [`GlideSlope::NONE`], and the
    /// floor and the capture box have their default values.
    #[must_use]
    pub const fn new(
        position_m: Vector3<f64>,
        velocity_mps: Vector3<f64>,
        acceleration_mps2: Vector3<f64>,
        min_specific_force_mps2: f64,
        max_specific_force_mps2: f64,
    ) -> Self {
        Self {
            position_m,
            velocity_mps,
            acceleration_mps2,
            min_specific_force_mps2,
            max_specific_force_mps2,
            glide_slope: GlideSlope::NONE,
            min_time_to_go_s: Self::DEFAULT_MIN_TIME_TO_GO_S,
            capture_radius_m: Self::DEFAULT_CAPTURE_RADIUS_M,
            capture_speed_mps: Self::DEFAULT_CAPTURE_SPEED_MPS,
        }
    }

    /// Sets the glide-slope cone.
    #[must_use]
    pub const fn with_glide_slope(mut self, glide_slope: GlideSlope) -> Self {
        self.glide_slope = glide_slope;
        self
    }

    /// Sets the floor on the time to go in seconds.
    #[must_use]
    pub const fn with_min_time_to_go_s(mut self, min_time_to_go_s: f64) -> Self {
        self.min_time_to_go_s = min_time_to_go_s;
        self
    }

    /// Sets the capture radius in meters and the capture speed in m/s.
    #[must_use]
    pub const fn with_capture(mut self, capture_radius_m: f64, capture_speed_mps: f64) -> Self {
        self.capture_radius_m = capture_radius_m;
        self.capture_speed_mps = capture_speed_mps;
        self
    }

    /// Checks that every value is finite and in its range.
    ///
    /// # Errors
    ///
    /// [`TrajectoryError::InvalidParameter`] with the name of the first
    /// value that is not.
    pub fn validate(&self) -> Result<(), TrajectoryError> {
        let finite = |v: &Vector3<f64>| v.iter().all(|c| c.is_finite());
        let in_range = |value: f64, low: f64| value.is_finite() && value >= low;
        let checks = [
            (finite(&self.position_m), "position_m"),
            (finite(&self.velocity_mps), "velocity_mps"),
            (finite(&self.acceleration_mps2), "acceleration_mps2"),
            (
                in_range(self.min_specific_force_mps2, 0.0),
                "min_specific_force_mps2",
            ),
            (
                in_range(self.max_specific_force_mps2, self.min_specific_force_mps2)
                    && self.max_specific_force_mps2 > 0.0,
                "max_specific_force_mps2",
            ),
            (
                in_range(self.glide_slope.min_elevation_rad, 0.0),
                "min_elevation_rad",
            ),
            (in_range(self.glide_slope.max_range_m, 0.0), "max_range_m"),
            (
                in_range(self.min_time_to_go_s, 0.0) && self.min_time_to_go_s > 0.0,
                "min_time_to_go_s",
            ),
            (in_range(self.capture_radius_m, 0.0), "capture_radius_m"),
            (in_range(self.capture_speed_mps, 0.0), "capture_speed_mps"),
        ];
        match checks.into_iter().find(|(ok, _)| !ok) {
            Some((_, name)) => Err(TrajectoryError::InvalidParameter { name }),
            None => Ok(()),
        }
    }
}

/// The output of one evaluation of [`evaluate_powered_descent`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct DescentCommand {
    /// The thrust direction, a unit vector in the body-fixed axes. At
    /// cutoff it is the local vertical of the target.
    pub direction_unit: Vector3<f64>,
    /// The commanded specific force in m/s². It is `0.0` at cutoff.
    pub specific_force_mps2: f64,
    /// The magnitude of `v_f − v` in m/s.
    pub velocity_to_go_mps: f64,
    /// The distance to the target in meters.
    pub range_m: f64,
    /// `true` inside the capture box.
    pub cutoff: bool,
}

/// Evaluates the E-guidance law at `state` with the time to go
/// `time_to_go_s` and the gravity `gravity_mps2` at the vehicle, in the
/// body-fixed axes of the target.
///
/// # Errors
///
/// The errors of [`PoweredDescentTarget::validate`],
/// [`TrajectoryError::NonFiniteState`],
/// [`TrajectoryError::InvalidParameter`] for a time to go or a gravity
/// that is not finite, [`TrajectoryError::DegenerateOrbit`] for a target
/// at the center of the body, and
/// [`TrajectoryError::GlideSlopeViolated`] below the glide-slope cone.
pub fn evaluate_powered_descent(
    target: &PoweredDescentTarget,
    state: &StateVector,
    time_to_go_s: f64,
    gravity_mps2: &Vector3<f64>,
) -> Result<DescentCommand, TrajectoryError> {
    target.validate()?;
    let state = state.checked()?;
    if !time_to_go_s.is_finite() {
        return Err(TrajectoryError::InvalidParameter {
            name: "time_to_go_s",
        });
    }
    if !gravity_mps2.iter().all(|c| c.is_finite()) {
        return Err(TrajectoryError::InvalidParameter {
            name: "gravity_mps2",
        });
    }
    let target_position = target.position_m;
    let up = unit(&target_position).ok_or(TrajectoryError::DegenerateOrbit)?;
    let to_target = target_position - state.position_m;
    let velocity_error = target.velocity_mps - state.velocity_mps;
    let range_m = to_target.norm();
    let velocity_to_go_mps = velocity_error.norm();
    if range_m <= target.capture_radius_m && velocity_to_go_mps <= target.capture_speed_mps {
        return Ok(DescentCommand {
            direction_unit: up,
            specific_force_mps2: 0.0,
            velocity_to_go_mps,
            range_m,
            cutoff: true,
        });
    }
    if range_m > target.capture_radius_m {
        check_glide_slope(&target.glide_slope, &to_target, &up, range_m)?;
    }
    let t = time_to_go_s.max(target.min_time_to_go_s);
    let acceleration = (to_target - state.velocity_mps * t) * (12.0 / (t * t))
        - velocity_error * (6.0 / t)
        + target.acceleration_mps2;
    let thrust = acceleration - gravity_mps2;
    Ok(DescentCommand {
        direction_unit: unit(&thrust).unwrap_or(up),
        specific_force_mps2: thrust.norm().clamp(
            target.min_specific_force_mps2,
            target.max_specific_force_mps2,
        ),
        velocity_to_go_mps,
        range_m,
        cutoff: false,
    })
}

fn check_glide_slope(
    cone: &GlideSlope,
    to_target: &Vector3<f64>,
    up: &Vector3<f64>,
    range_m: f64,
) -> Result<(), TrajectoryError> {
    if range_m <= 0.0 || range_m > cone.max_range_m {
        return Ok(());
    }
    let elevation_rad = asin((-to_target.dot(up) / range_m).clamp(-1.0, 1.0));
    if elevation_rad < cone.min_elevation_rad {
        return Err(TrajectoryError::GlideSlopeViolated {
            elevation_rad,
            minimum_rad: cone.min_elevation_rad,
        });
    }
    Ok(())
}
