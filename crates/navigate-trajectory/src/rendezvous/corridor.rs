//! The keep-out sphere and the approach cone around a target.

use nalgebra::Vector3;

use crate::error::TrajectoryError;
use crate::float_math::{cos, sin};
use crate::state::unit;

/// A keep-out sphere around the target, with one approach cone through
/// it.
///
/// Inside the sphere, the chaser must stay in the cone. The margin is
/// the distance from the chaser to the cone surface. The check refuses
/// when the margin is negative, or when the scaled position uncertainty
/// is larger than the margin.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ApproachCorridor {
    /// The cone axis from the target toward the chaser, in LVLH
    /// coordinates. For an approach along the negative along-track
    /// axis, it is `(0, −1, 0)`.
    pub axis_lvlh: Vector3<f64>,
    /// The half angle of the cone in radians, in `(0, π/2)`.
    pub half_angle_rad: f64,
    /// The radius of the keep-out sphere in meters.
    pub keep_out_radius_m: f64,
    /// The number of one-sigma position uncertainties that must fit in
    /// the margin.
    pub sigma_multiplier: f64,
}

impl ApproachCorridor {
    /// Builds a corridor from its parts.
    #[must_use]
    pub const fn new(
        axis_lvlh: Vector3<f64>,
        half_angle_rad: f64,
        keep_out_radius_m: f64,
        sigma_multiplier: f64,
    ) -> Self {
        Self {
            axis_lvlh,
            half_angle_rad,
            keep_out_radius_m,
            sigma_multiplier,
        }
    }

    /// Checks the chaser at the LVLH position `rho` with the one-sigma
    /// position uncertainty `position_1sigma_m`. The result is the
    /// margin to the cone surface in meters, or `None` when the chaser
    /// and its scaled uncertainty are outside the keep-out sphere.
    ///
    /// # Errors
    ///
    /// [`TrajectoryError::InvalidParameter`] for a corridor value or an
    /// uncertainty that is not finite or out of range,
    /// [`TrajectoryError::OutsideApproachCorridor`] when the chaser is
    /// outside the cone, and
    /// [`TrajectoryError::UncertaintyExceedsCorridorMargin`] when the
    /// scaled uncertainty is larger than the margin.
    pub fn check(
        &self,
        rho: &Vector3<f64>,
        position_1sigma_m: f64,
    ) -> Result<Option<f64>, TrajectoryError> {
        let axis = self.validated_axis()?;
        if !(position_1sigma_m.is_finite() && position_1sigma_m >= 0.0) {
            return Err(TrajectoryError::InvalidParameter {
                name: "position_1sigma_m",
            });
        }
        let scaled_1sigma_m = self.sigma_multiplier * position_1sigma_m;
        if rho.norm() > self.keep_out_radius_m + scaled_1sigma_m {
            return Ok(None);
        }
        let axial = rho.dot(&axis);
        let lateral = (rho - axis * axial).norm();
        let margin_m = axial * sin(self.half_angle_rad) - lateral * cos(self.half_angle_rad);
        if margin_m < 0.0 {
            return Err(TrajectoryError::OutsideApproachCorridor { margin_m });
        }
        if scaled_1sigma_m > margin_m {
            return Err(TrajectoryError::UncertaintyExceedsCorridorMargin {
                scaled_1sigma_m,
                margin_m,
            });
        }
        Ok(Some(margin_m))
    }

    fn validated_axis(&self) -> Result<Vector3<f64>, TrajectoryError> {
        let invalid = |name| TrajectoryError::InvalidParameter { name };
        if !(self.half_angle_rad.is_finite()
            && self.half_angle_rad > 0.0
            && self.half_angle_rad < core::f64::consts::FRAC_PI_2)
        {
            return Err(invalid("half_angle_rad"));
        }
        if !(self.keep_out_radius_m.is_finite() && self.keep_out_radius_m >= 0.0) {
            return Err(invalid("keep_out_radius_m"));
        }
        if !(self.sigma_multiplier.is_finite() && self.sigma_multiplier >= 0.0) {
            return Err(invalid("sigma_multiplier"));
        }
        unit(&self.axis_lvlh).ok_or(invalid("axis_lvlh"))
    }
}
