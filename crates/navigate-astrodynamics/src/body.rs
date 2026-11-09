//! Rotating planetary bodies and their apparent gravity (ADR-0014,
//! ADR-0016).
//!
//! The apparent gravity of a rotating body is the sum of the point
//! mass, the `J2` zonal harmonic, and the centrifugal term. A navigation
//! filter evaluates it at the body-centered position of the vehicle.

use nalgebra::{Matrix3, Vector3};
use navigate_geodesy::{BodyEllipsoid, geodetic_to_body_fixed};

use crate::float_math::{sin_cos, sqrt};
use crate::gravity::{CentralBodyGravity, MAX_ABS_J2, centrifugal, centrifugal_gradient};

/// A rotating body with a reference ellipsoid, a gravitational
/// parameter, a `J2` zonal harmonic, and a rotation rate about the
/// body-fixed `z` axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanetaryBody {
    /// The reference ellipsoid. Its frame is the body-fixed frame.
    pub ellipsoid: BodyEllipsoid,
    /// Gravitational parameter `μ` in m³/s².
    pub gm_m3_per_s2: f64,
    /// Unnormalized `J2` zonal harmonic. It has no unit.
    pub j2: f64,
    /// Rotation rate about the body-fixed `z` axis in rad/s.
    pub rotation_rate_rad_per_s: f64,
}

impl PlanetaryBody {
    /// The Earth on WGS84 with the WGS84 gravity constants.
    pub const EARTH_WGS84: Self = Self {
        ellipsoid: BodyEllipsoid::WGS84,
        gm_m3_per_s2: 3.986_004_418e14,
        j2: 1.082_626_68e-3,
        rotation_rate_rad_per_s: 7.292_115e-5,
    };

    /// The Moon in the principal axis frame, with the GRAIL GRGM1200A
    /// field: `μ` from the model, and `J2 = −√5·C̄₂₀` taken from the model's
    /// 1738 km reference radius to the 1737.4 km radius of the ellipsoid,
    /// as `J2·R²` is invariant.
    pub const MOON_PA: Self = Self {
        ellipsoid: BodyEllipsoid::MOON_PA,
        gm_m3_per_s2: 4.902_800_122_445_3e12,
        j2: 2.033_639_490e-4,
        rotation_rate_rad_per_s: 2.661_699_5e-6,
    };

    /// The Moon in the mean Earth frame, with the constants of
    /// [`Self::MOON_PA`].
    pub const MOON_ME: Self = Self {
        ellipsoid: BodyEllipsoid::MOON_ME,
        ..Self::MOON_PA
    };

    /// Mars in the IAU body-fixed frame.
    pub const MARS_IAU: Self = Self {
        ellipsoid: BodyEllipsoid::MARS_IAU,
        gm_m3_per_s2: 4.282_837_2e13,
        j2: 1.960_45e-3,
        rotation_rate_rad_per_s: 7.088_218_1e-5,
    };

    /// Whether `μ` is finite and positive, `J2` is finite and small, and
    /// the rotation rate is finite.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        self.gm_m3_per_s2.is_finite()
            && self.gm_m3_per_s2 > 0.0
            && self.j2.is_finite()
            && self.j2.abs() < MAX_ABS_J2
            && self.rotation_rate_rad_per_s.is_finite()
    }

    /// The point mass and the `J2` term about the body-fixed `z` axis.
    #[must_use]
    pub const fn central_gravity(&self) -> CentralBodyGravity {
        CentralBodyGravity {
            gm_m3_per_s2: self.gm_m3_per_s2,
            j2: self.j2,
            equatorial_radius_m: self.ellipsoid.semi_major_axis_m(),
            pole_unit_in_frame: [0.0, 0.0, 1.0],
        }
    }

    /// The rotation vector of the body in its body-fixed axes, in rad/s.
    #[must_use]
    pub const fn rotation_body_fixed(&self) -> Vector3<f64> {
        Vector3::new(0.0, 0.0, self.rotation_rate_rad_per_s)
    }

    /// The apparent gravity at the body-fixed position `r` in meters and
    /// its gradient `∂g/∂r`, both in the body-fixed axes.
    #[must_use]
    pub fn apparent_gravity_body_fixed(&self, r: &Vector3<f64>) -> (Vector3<f64>, Matrix3<f64>) {
        let central = self.central_gravity();
        let omega = self.rotation_body_fixed();
        (
            central.acceleration(r) + centrifugal(&omega, r),
            central.gradient(r) + centrifugal_gradient(&omega),
        )
    }

    /// The magnitude in m/s² of the apparent gravity of the `J2` model on
    /// the reference ellipsoid at the geodetic latitude `latitude_rad`.
    /// The model is symmetric about the `z` axis, so the longitude has no
    /// effect.
    #[must_use]
    pub fn surface_gravity_mps2(&self, latitude_rad: f64) -> f64 {
        let position = navigate_contract::GeodeticPosition::new(latitude_rad, 0.0, 0.0);
        let r = geodetic_to_body_fixed(&position, self.ellipsoid);
        let (g, _) = self.apparent_gravity_body_fixed(&Vector3::new(r.x_m, r.y_m, r.z_m));
        g.norm()
    }

    /// The normal gravity in m/s² of the Somigliana closed form at the
    /// geodetic latitude `latitude_rad`:
    /// `γ = (a·γ_e·cos²φ + b·γ_p·sin²φ) / √(a²·cos²φ + b²·sin²φ)`.
    ///
    /// The equatorial value `γ_e` and the polar value `γ_p` come from
    /// [`Self::surface_gravity_mps2`]. The closed form is smooth in the
    /// latitude, and it is equal to the `J2` model at the equator and at
    /// the poles.
    #[must_use]
    pub fn somigliana_gravity_mps2(&self, latitude_rad: f64) -> f64 {
        let a = self.ellipsoid.semi_major_axis_m();
        let b = self.ellipsoid.semi_minor_axis_m();
        let gamma_e = self.surface_gravity_mps2(0.0);
        let gamma_p = self.surface_gravity_mps2(core::f64::consts::FRAC_PI_2);
        let (sin_lat, cos_lat) = sin_cos(latitude_rad);
        let (sin2, cos2) = (sin_lat * sin_lat, cos_lat * cos_lat);
        (a * gamma_e * cos2 + b * gamma_p * sin2) / sqrt(a * a * cos2 + b * b * sin2)
    }
}

#[cfg(test)]
mod tests;
