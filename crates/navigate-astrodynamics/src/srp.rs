//! Cannonball solar radiation pressure with a cylindrical shadow
//! (ADR-0018).
//!
//! The pressure at one astronomical unit is
//! [`SOLAR_PRESSURE_AT_1_AU_N_PER_M2`]. It decreases with the square of
//! the distance from the Sun. With `d = r − r_☉`, the acceleration is
//! `a = P₀·C_R·(A/m)·AU²·d/|d|³`, away from the Sun. Its gradient is
//! `∂a/∂r = P₀·C_R·(A/m)·AU²·(I − 3·d̂·d̂ᵀ)/|d|³`. The central body at the
//! frame origin casts a cylindrical shadow along the Sun direction. In
//! the shadow the acceleration and the gradient are zero.

use nalgebra::{Matrix3, Vector3};

use crate::float_math::powi;
use crate::force_model::ForceEvaluation;

/// Solar radiation pressure at one astronomical unit in N/m².
pub const SOLAR_PRESSURE_AT_1_AU_N_PER_M2: f64 = 4.56e-6;

/// One astronomical unit in meters.
pub const ASTRONOMICAL_UNIT_M: f64 = 1.495_978_707e11;

/// Below this distance from the Sun in meters, the pressure model has no
/// direction.
const MIN_SUN_DISTANCE_M: f64 = 1.0;

/// A spherical spacecraft in the light of the Sun.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CannonballSrp {
    /// Reflectivity coefficient `C_R`, from 1 for full absorption to 2
    /// for full specular reflection. It has no unit.
    pub reflectivity_cr: f64,
    /// Area-to-mass ratio `A/m` in m²/kg.
    pub area_to_mass_m2_per_kg: f64,
    /// Position of the Sun in meters, in the frame.
    pub sun_position_in_frame_m: [f64; 3],
    /// Radius of the central body at the frame origin in meters. It sets
    /// the radius of the shadow cylinder. Zero gives no shadow.
    pub central_body_radius_m: f64,
}

impl CannonballSrp {
    /// Whether every value is finite, `C_R` and `A/m` are not negative,
    /// the radius is not negative, and the Sun is not at the origin.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        let sun = Vector3::from(self.sun_position_in_frame_m);
        sun.iter().all(|c| c.is_finite())
            && sun.norm() >= MIN_SUN_DISTANCE_M
            && self.reflectivity_cr.is_finite()
            && self.reflectivity_cr >= 0.0
            && self.area_to_mass_m2_per_kg.is_finite()
            && self.area_to_mass_m2_per_kg >= 0.0
            && self.central_body_radius_m.is_finite()
            && self.central_body_radius_m >= 0.0
    }

    /// Whether `r` is in the cylindrical shadow of the central body:
    /// behind the body as seen from the Sun, and closer to the shadow
    /// axis than the body radius.
    #[must_use]
    pub fn in_shadow(&self, r: &Vector3<f64>) -> bool {
        let sun = Vector3::from(self.sun_position_in_frame_m);
        let sun_norm = sun.norm();
        if sun_norm < MIN_SUN_DISTANCE_M {
            return false;
        }
        let toward_sun = sun / sun_norm;
        let along = r.dot(&toward_sun);
        along < 0.0 && (r - along * toward_sun).norm() < self.central_body_radius_m
    }

    /// The acceleration in m/s² at `r` and its gradient in 1/s². The
    /// pressure does not depend on the velocity.
    #[must_use]
    pub fn evaluate(&self, r: &Vector3<f64>) -> ForceEvaluation {
        let d = r - Vector3::from(self.sun_position_in_frame_m);
        let distance = d.norm();
        if distance < MIN_SUN_DISTANCE_M || self.in_shadow(r) {
            return ForceEvaluation::ZERO;
        }
        let k = SOLAR_PRESSURE_AT_1_AU_N_PER_M2
            * self.reflectivity_cr
            * self.area_to_mass_m2_per_kg
            * ASTRONOMICAL_UNIT_M
            * ASTRONOMICAL_UNIT_M;
        let unit = d / distance;
        let inverse_cube = 1.0 / powi(distance, 3);
        ForceEvaluation {
            acceleration_mps2: k * inverse_cube * d,
            position_jacobian: k
                * inverse_cube
                * (Matrix3::identity() - 3.0 * unit * unit.transpose()),
            velocity_jacobian: Matrix3::zeros(),
        }
    }
}
