//! Spherical harmonic gravity from borrowed Stokes coefficients
//! (ADR-0018).
//!
//! The field uses the Cartesian solid harmonics `V_nm` and `W_nm` of
//! Montenbruck and Gill in the body-fixed axes. The recursion uses `x`,
//! `y`, and `z` and no angle, so it has no limit at a pole and needs no
//! trigonometric function. The potential is
//! `U = μ/R·Σ (C_nm·V_nm + S_nm·W_nm)` for `0 ≤ m ≤ n ≤ N`. The field
//! gives the acceleration `∇U` and its exact gradient `∇∇U`. Each
//! derivative is a sum of solid harmonics of a higher degree, so the
//! work table holds the degrees up to `N + 2`. The table is on the
//! stack and has a fixed size, so the evaluation does not allocate.

use nalgebra::{Matrix3, Vector3};

use super::MIN_CENTRAL_DISTANCE_M;
use crate::float_math::sqrt;

mod solid;

#[cfg(test)]
mod tests;

use solid::SolidHarmonics;

/// Largest degree of a [`SphericalHarmonicField`]. The work table of an
/// evaluation holds the degrees up to this value plus two.
pub const MAX_INLINE_HARMONIC_DEGREE: usize = 36;

/// Largest deviation of `RᵀR` from the identity that an orientation of a
/// field can have.
const ORTHONORMAL_TOLERANCE: f64 = 1e-9;

/// The normalization of the Stokes coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarmonicNormalization {
    /// Unnormalized coefficients: `J2 = −C₂₀`.
    Unnormalized,
    /// Fully normalized coefficients with the factor
    /// `√((2 − δ_m0)·(2n + 1)·(n − m)!/(n + m)!)`: `J2 = −√5·C̄₂₀`.
    FullyNormalized,
}

/// A refusal of the constants of a [`SphericalHarmonicField`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum HarmonicFieldError {
    /// The degree is larger than [`MAX_INLINE_HARMONIC_DEGREE`].
    #[error("degree {degree} exceeds the inline limit {max}")]
    DegreeTooHigh {
        /// The requested degree.
        degree: usize,
        /// The inline limit.
        max: usize,
    },
    /// A coefficient slice is shorter than the triangle of the degree.
    #[error("degree needs {needed} coefficients; C has {c_len} and S has {s_len}")]
    TooFewCoefficients {
        /// The number of coefficients of the triangle.
        needed: usize,
        /// The length of the `C` slice.
        c_len: usize,
        /// The length of the `S` slice.
        s_len: usize,
    },
    /// A constant or a coefficient is not finite, `μ` or the reference
    /// radius is not positive, or the orientation is not orthonormal.
    #[error("a constant or a coefficient of the field is not plausible")]
    NotPlausible,
}

/// The index of the coefficient of degree `n` and order `m` in a
/// coefficient slice: `n·(n + 1)/2 + m`.
#[must_use]
pub const fn harmonic_index(n: usize, m: usize) -> usize {
    n * (n + 1) / 2 + m
}

/// The number of coefficients of the triangle `0 ≤ m ≤ n ≤ degree`.
#[must_use]
pub const fn harmonic_coefficient_count(degree: usize) -> usize {
    harmonic_index(degree + 1, 0)
}

/// A spherical harmonic gravity field that borrows its Stokes
/// coefficients.
///
/// The slices `C` and `S` hold the coefficients in the order of
/// [`harmonic_index`], from degree zero. `C₀₀ = 1` gives the point mass.
/// The field is `Copy`. A field that borrows `'static` coefficients is
/// `'static`, so a [`super::SpaceGravityModel`] can hold it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalHarmonicField<'a> {
    gm_m3_per_s2: f64,
    reference_radius_m: f64,
    degree: usize,
    normalization: HarmonicNormalization,
    c: &'a [f64],
    s: &'a [f64],
    /// Rotation from the frame axes to the body-fixed axes of the
    /// coefficients.
    frame_to_body: Matrix3<f64>,
}

impl<'a> SphericalHarmonicField<'a> {
    /// A field of the degree `degree` in the body-fixed axes of the
    /// coefficients.
    ///
    /// # Errors
    ///
    /// [`HarmonicFieldError::DegreeTooHigh`],
    /// [`HarmonicFieldError::TooFewCoefficients`], and
    /// [`HarmonicFieldError::NotPlausible`].
    pub fn new(
        gm_m3_per_s2: f64,
        reference_radius_m: f64,
        degree: usize,
        normalization: HarmonicNormalization,
        c: &'a [f64],
        s: &'a [f64],
    ) -> Result<Self, HarmonicFieldError> {
        if degree > MAX_INLINE_HARMONIC_DEGREE {
            return Err(HarmonicFieldError::DegreeTooHigh {
                degree,
                max: MAX_INLINE_HARMONIC_DEGREE,
            });
        }
        let needed = harmonic_coefficient_count(degree);
        if c.len() < needed || s.len() < needed {
            return Err(HarmonicFieldError::TooFewCoefficients {
                needed,
                c_len: c.len(),
                s_len: s.len(),
            });
        }
        let field = Self {
            gm_m3_per_s2,
            reference_radius_m,
            degree,
            normalization,
            c,
            s,
            frame_to_body: Matrix3::identity(),
        };
        if field.is_plausible() {
            Ok(field)
        } else {
            Err(HarmonicFieldError::NotPlausible)
        }
    }

    /// This field in a frame whose axes are not the body-fixed axes.
    /// `frame_to_body` rotates the frame axes into the body-fixed axes.
    /// A rotating body turns relative to an inertial frame, so the caller
    /// sets a new orientation as the time goes on.
    ///
    /// # Errors
    ///
    /// [`HarmonicFieldError::NotPlausible`] for a matrix that is not
    /// finite and orthonormal.
    pub fn with_orientation(self, frame_to_body: Matrix3<f64>) -> Result<Self, HarmonicFieldError> {
        let field = Self {
            frame_to_body,
            ..self
        };
        if field.is_plausible() {
            Ok(field)
        } else {
            Err(HarmonicFieldError::NotPlausible)
        }
    }

    /// The degree of the field.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
    }

    /// Gravitational parameter `μ` in m³/s².
    #[must_use]
    pub const fn gm_m3_per_s2(&self) -> f64 {
        self.gm_m3_per_s2
    }

    /// Reference radius `R` of the coefficients in meters.
    #[must_use]
    pub const fn reference_radius_m(&self) -> f64 {
        self.reference_radius_m
    }

    /// Whether `μ` and `R` are finite and positive, each coefficient of
    /// the triangle is finite, and the orientation is orthonormal.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        let needed = harmonic_coefficient_count(self.degree);
        let coefficients = match (self.c.get(..needed), self.s.get(..needed)) {
            (Some(c), Some(s)) => c.iter().chain(s).all(|value| value.is_finite()),
            _ => false,
        };
        let r = &self.frame_to_body;
        let orthonormal = r.iter().all(|value| value.is_finite())
            && (r.transpose() * r - Matrix3::identity()).abs().max() <= ORTHONORMAL_TOLERANCE;
        self.degree <= MAX_INLINE_HARMONIC_DEGREE
            && self.gm_m3_per_s2.is_finite()
            && self.gm_m3_per_s2 > 0.0
            && self.reference_radius_m.is_finite()
            && self.reference_radius_m > 0.0
            && coefficients
            && orthonormal
    }

    /// The unnormalized coefficients `(C_nm, S_nm)`. `scale` is the
    /// normalization factor of the coefficient.
    fn coefficients(&self, n: usize, m: usize, scale: f64) -> (f64, f64) {
        let index = harmonic_index(n, m);
        let c = self.c.get(index).copied().unwrap_or(0.0);
        let s = self.s.get(index).copied().unwrap_or(0.0);
        (c * scale, s * scale)
    }

    /// Calls `term` with each degree, order, and unnormalized coefficient
    /// pair of the field.
    fn for_each_term(&self, mut term: impl FnMut(usize, usize, f64, f64)) {
        for n in 0..=self.degree {
            // (n − m)!/(n + m)!, updated for each order.
            let mut ratio = 1.0;
            for m in 0..=n {
                if m > 0 {
                    ratio /= ((n - m + 1) * (n + m)) as f64;
                }
                let scale = match self.normalization {
                    HarmonicNormalization::Unnormalized => 1.0,
                    HarmonicNormalization::FullyNormalized => {
                        let kronecker = if m == 0 { 1.0 } else { 2.0 };
                        sqrt(kronecker * (2 * n + 1) as f64 * ratio)
                    }
                };
                let (c, s) = self.coefficients(n, m, scale);
                term(n, m, c, s);
            }
        }
    }

    /// The potential `U` in m²/s² at `r` in meters in the frame axes.
    /// Zero closer than one meter to the center.
    #[must_use]
    pub fn potential(&self, r: &Vector3<f64>) -> f64 {
        let body = self.frame_to_body * r;
        if body.norm() < MIN_CENTRAL_DISTANCE_M {
            return 0.0;
        }
        let table = SolidHarmonics::at(&body, self.reference_radius_m, self.degree);
        let mut sum = 0.0;
        self.for_each_term(|n, m, c, s| sum += table.potential_term(n, m, c, s));
        self.gm_m3_per_s2 / self.reference_radius_m * sum
    }

    /// The acceleration `∇U` in m/s² at `r` in meters, and its gradient
    /// `∇∇U` in 1/s², both in the frame axes. Both are zero closer than
    /// one meter to the center.
    #[must_use]
    pub fn evaluate(&self, r: &Vector3<f64>) -> (Vector3<f64>, Matrix3<f64>) {
        let body = self.frame_to_body * r;
        if body.norm() < MIN_CENTRAL_DISTANCE_M {
            return (Vector3::zeros(), Matrix3::zeros());
        }
        let radius = self.reference_radius_m;
        let table = SolidHarmonics::at(&body, radius, self.degree + 2);
        let mut acceleration = Vector3::zeros();
        let mut hessian = Matrix3::zeros();
        self.for_each_term(|n, m, c, s| {
            acceleration += table.gradient_term(n, m, c, s);
            hessian += table.hessian_term(n, m, c, s);
        });
        let mu = self.gm_m3_per_s2;
        let acceleration = acceleration * (mu / (radius * radius));
        let hessian = hessian * (mu / (radius * radius * radius));
        let to_frame = self.frame_to_body.transpose();
        (
            to_frame * acceleration,
            to_frame * hessian * self.frame_to_body,
        )
    }
}
