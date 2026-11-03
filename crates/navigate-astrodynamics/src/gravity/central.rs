//! The point mass and the `J2` zonal harmonic of a central body.

use nalgebra::{Matrix3, Vector3};

use super::{MAX_ABS_J2, MIN_CENTRAL_DISTANCE_M};
use crate::float_math::powi;

/// The point mass and the `J2` zonal harmonic of the central body about
/// a pole in the frame axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CentralBodyGravity {
    /// Gravitational parameter `μ` in m³/s².
    pub gm_m3_per_s2: f64,
    /// Unnormalized `J2` zonal harmonic. It has no unit.
    pub j2: f64,
    /// Reference radius `a` of `J2` in meters.
    pub equatorial_radius_m: f64,
    /// Unit pole of the body in the frame axes.
    pub pole_unit_in_frame: [f64; 3],
}

impl CentralBodyGravity {
    /// Whether every constant is finite, `μ` and `a` are positive, `J2`
    /// is small, and the pole has unit norm.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        let pole_norm = Vector3::from(self.pole_unit_in_frame).norm();
        self.gm_m3_per_s2.is_finite()
            && self.gm_m3_per_s2 > 0.0
            && self.j2.is_finite()
            && self.j2.abs() < MAX_ABS_J2
            && self.equatorial_radius_m.is_finite()
            && self.equatorial_radius_m > 0.0
            && (pole_norm - 1.0).abs() <= 1e-9
    }

    /// The gravity in m/s² at `r` in meters. The form is free of
    /// coordinates, with `k` the pole, `s = r·k`, and `u = s/|r|`:
    /// `g = −μ/|r|³·(r + 1.5·J2·(a/|r|)²·((1 − 5u²)·r + 2·s·k))`.
    #[must_use]
    pub fn acceleration(&self, r: &Vector3<f64>) -> Vector3<f64> {
        let r_norm = r.norm();
        if r_norm < MIN_CENTRAL_DISTANCE_M {
            return Vector3::zeros();
        }
        let k = Vector3::from(self.pole_unit_in_frame);
        let s = r.dot(&k);
        let u = s / r_norm;
        let ratio = self.equatorial_radius_m / r_norm;
        let zonal = 1.5 * self.j2 * ratio * ratio * ((1.0 - 5.0 * u * u) * r + 2.0 * s * k);
        -self.gm_m3_per_s2 / powi(r_norm, 3) * (r + zonal)
    }

    /// The exact gradient `∂g/∂r` of [`Self::acceleration`] in 1/s².
    #[must_use]
    pub fn gradient(&self, r: &Vector3<f64>) -> Matrix3<f64> {
        let r_norm = r.norm();
        if r_norm < MIN_CENTRAL_DISTANCE_M {
            return Matrix3::zeros();
        }
        let identity = Matrix3::identity();
        let mu = self.gm_m3_per_s2;
        let r2 = r_norm * r_norm;
        let point_mass = -mu / powi(r_norm, 3) * (identity - 3.0 * r * r.transpose() / r2);
        let k = Vector3::from(self.pole_unit_in_frame);
        let s = r.dot(&k);
        let u2 = s * s / r2;
        // g₂ = −c·w/|r|⁵ with c = 1.5·μ·J2·a² and
        // w = (1 − 5u²)·r + 2·s·k.
        let c = 1.5 * mu * self.j2 * self.equatorial_radius_m * self.equatorial_radius_m;
        let w = (1.0 - 5.0 * u2) * r + 2.0 * s * k;
        let dw = (1.0 - 5.0 * u2) * identity - 10.0 * s / r2 * r * k.transpose()
            + 10.0 * s * s / (r2 * r2) * r * r.transpose()
            + 2.0 * k * k.transpose();
        let r5 = r2 * r2 * r_norm;
        let zonal = -c * (dw / r5 - 5.0 * w * r.transpose() / (r5 * r2));
        point_mass + zonal
    }
}
