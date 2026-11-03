//! Atmospheric drag with an exponential density and a co-rotating
//! atmosphere (ADR-0018).
//!
//! The height is `h = |r| − R` above a sphere of the equatorial radius
//! `R`. The density is `ρ = ρ₀·exp(−(h − h₀)/H)`. The atmosphere turns
//! with the rate `ω` relative to the frame, so the air-relative velocity
//! is `w = v − ω × r`. With the ballistic factor `B = C_D·A/m`, the
//! acceleration is `a = −½·ρ·B·|w|·w` and its Jacobians are
//!
//! - `∂a/∂v = −½·ρ·B·(|w|·I + w·wᵀ/|w|)`
//! - `∂a/∂r = ρ·B·|w|/(2H)·w·r̂ᵀ − (∂a/∂v)·[ω×]`

use nalgebra::{Matrix3, Vector3};

use crate::float_math::exp;
use crate::force_model::ForceEvaluation;

/// Below this air-relative speed in m/s, the drag and its Jacobians are
/// zero. The direction of the drag is not defined at zero speed.
const MIN_AIR_SPEED_MPS: f64 = 1e-9;

/// Below this distance from the frame origin in meters, the height has
/// no gradient.
const MIN_RADIUS_M: f64 = 1.0;

/// Drag in an atmosphere with an exponential density.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExponentialDrag {
    /// Drag coefficient `C_D`. It has no unit.
    pub cd: f64,
    /// Area-to-mass ratio `A/m` in m²/kg.
    pub area_to_mass_m2_per_kg: f64,
    /// Density `ρ₀` at the reference height in kg/m³.
    pub reference_density_kg_m3: f64,
    /// Reference height `h₀` above the sphere in meters.
    pub reference_altitude_m: f64,
    /// Scale height `H` in meters.
    pub scale_height_m: f64,
    /// Radius `R` of the sphere of the heights in meters.
    pub equatorial_radius_m: f64,
    /// Angular rate `ω` of the atmosphere relative to the frame, in
    /// rad/s in the frame axes. In an inertial frame it is the rotation
    /// rate of the body. In the body-fixed frame of the body it is zero.
    pub atmosphere_rotation_rad_s: [f64; 3],
}

impl ExponentialDrag {
    /// Whether every value is finite, `C_D`, `A/m`, and `ρ₀` are not
    /// negative, and `H` and `R` are positive.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        let finite = [
            self.cd,
            self.area_to_mass_m2_per_kg,
            self.reference_density_kg_m3,
            self.reference_altitude_m,
            self.scale_height_m,
            self.equatorial_radius_m,
        ]
        .iter()
        .chain(&self.atmosphere_rotation_rad_s)
        .all(|v| v.is_finite());
        finite
            && self.cd >= 0.0
            && self.area_to_mass_m2_per_kg >= 0.0
            && self.reference_density_kg_m3 >= 0.0
            && self.scale_height_m > 0.0
            && self.equatorial_radius_m > 0.0
    }

    /// The density in kg/m³ at `r`.
    #[must_use]
    pub fn density(&self, r: &Vector3<f64>) -> f64 {
        let height = r.norm() - self.equatorial_radius_m;
        self.reference_density_kg_m3
            * exp(-(height - self.reference_altitude_m) / self.scale_height_m)
    }

    /// The velocity of the air in m/s at `r`, relative to the frame.
    #[must_use]
    pub fn air_velocity(&self, r: &Vector3<f64>) -> Vector3<f64> {
        Vector3::from(self.atmosphere_rotation_rad_s).cross(r)
    }

    /// The acceleration in m/s² at `r` with the velocity `v` relative to
    /// the frame, and its Jacobians.
    #[must_use]
    pub fn evaluate(&self, r: &Vector3<f64>, v: &Vector3<f64>) -> ForceEvaluation {
        let radius = r.norm();
        let w = v - self.air_velocity(r);
        let speed = w.norm();
        if speed < MIN_AIR_SPEED_MPS || radius < MIN_RADIUS_M {
            return ForceEvaluation::ZERO;
        }
        let rho = self.density(r);
        let ballistic = self.cd * self.area_to_mass_m2_per_kg;
        let half = 0.5 * rho * ballistic;
        let velocity_jacobian = -half * (speed * Matrix3::identity() + w * w.transpose() / speed);
        let omega = Vector3::from(self.atmosphere_rotation_rad_s);
        let density_slope = half * speed / self.scale_height_m * w * (r / radius).transpose();
        ForceEvaluation {
            acceleration_mps2: -half * speed * w,
            position_jacobian: density_slope - velocity_jacobian * omega.cross_matrix(),
            velocity_jacobian,
        }
    }
}
