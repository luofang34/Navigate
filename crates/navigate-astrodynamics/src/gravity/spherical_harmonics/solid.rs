//! The table of Cartesian solid harmonics `V_nm` and `W_nm` and their
//! first and second derivatives (ADR-0018).
//!
//! With `Φ_nm = V_nm + i·W_nm` and the operators `∂₊ = ∂x + i·∂y` and
//! `∂₋ = ∂x − i·∂y`, the derivatives of the solid harmonics are again
//! solid harmonics of the next degree:
//!
//! - `R·∂₊Φ_nm = −Φ_(n+1),(m+1)`
//! - `R·∂₋Φ_nm = (n−m+1)·(n−m+2)·Φ_(n+1),(m−1)`
//! - `R·∂zΦ_nm = −(n−m+1)·Φ_(n+1),m`
//!
//! A negative order is `Φ_n,−k = (−1)^k·(n−k)!/(n+k)!·conj(Φ_nk)`, so the
//! three rules hold for each order. The second derivatives apply the
//! rules two times.

use nalgebra::{Matrix3, Vector3};

use super::MAX_INLINE_HARMONIC_DEGREE;
use crate::float_math::sqrt;

/// Size of each table axis: the degree limit, two degrees for the second
/// derivatives, and the degree zero.
pub(super) const TABLE: usize = MAX_INLINE_HARMONIC_DEGREE + 3;

/// A complex value `(re, im)`.
type Complex = (f64, f64);

/// The solid harmonics of one position up to one degree.
pub(super) struct SolidHarmonics {
    v: [[f64; TABLE]; TABLE],
    w: [[f64; TABLE]; TABLE],
}

impl SolidHarmonics {
    /// The harmonics up to the degree `top` at `r` in the body-fixed axes,
    /// with the reference radius `radius_m`. The recursion of Montenbruck
    /// and Gill uses `x`, `y`, and `z` only, so it has no limit at a pole.
    pub(super) fn at(r: &Vector3<f64>, radius_m: f64, top: usize) -> Self {
        let mut table = Self {
            v: [[0.0; TABLE]; TABLE],
            w: [[0.0; TABLE]; TABLE],
        };
        let top = top.min(TABLE - 1);
        let r2 = r.norm_squared();
        let scale = radius_m / r2;
        let (x0, y0, z0) = (r.x * scale, r.y * scale, r.z * scale);
        let rho = radius_m * scale;
        for m in 0..=top {
            let (v_mm, w_mm) = if m == 0 {
                (radius_m / sqrt(r2), 0.0)
            } else {
                let (v, w) = table.get(m - 1, m - 1);
                let k = (2 * m - 1) as f64;
                (k * (x0 * v - y0 * w), k * (x0 * w + y0 * v))
            };
            table.set(m, m, v_mm, w_mm);
            for n in (m + 1)..=top {
                let (v1, w1) = table.get(n - 1, m);
                let (v2, w2) = if n >= m + 2 {
                    table.get(n - 2, m)
                } else {
                    (0.0, 0.0)
                };
                let a = (2 * n - 1) as f64 * z0;
                let b = (n + m - 1) as f64 * rho;
                let d = (n - m) as f64;
                table.set(n, m, (a * v1 - b * v2) / d, (a * w1 - b * w2) / d);
            }
        }
        table
    }

    fn get(&self, n: usize, m: usize) -> Complex {
        let v = self.v.get(n).and_then(|row| row.get(m)).copied();
        let w = self.w.get(n).and_then(|row| row.get(m)).copied();
        (v.unwrap_or(0.0), w.unwrap_or(0.0))
    }

    fn set(&mut self, n: usize, m: usize, v: f64, w: f64) {
        if let Some(slot) = self.v.get_mut(n).and_then(|row| row.get_mut(m)) {
            *slot = v;
        }
        if let Some(slot) = self.w.get_mut(n).and_then(|row| row.get_mut(m)) {
            *slot = w;
        }
    }

    /// `Φ_nm` for an order that can be negative.
    fn phi(&self, n: usize, m: i64) -> Complex {
        let k = m.unsigned_abs() as usize;
        if k > n {
            return (0.0, 0.0);
        }
        let (v, w) = self.get(n, k);
        if m >= 0 {
            return (v, w);
        }
        // (n−k)!/(n+k)! = 1/((n−k+1)·…·(n+k)).
        let product: f64 = ((n - k + 1)..=(n + k)).map(|j| j as f64).product();
        let sign = if k.is_multiple_of(2) { 1.0 } else { -1.0 };
        let factor = sign / product;
        (factor * v, -factor * w)
    }

    /// `U_nm·R/μ = V_nm·C + W_nm·S`.
    pub(super) fn potential_term(&self, n: usize, m: usize, c: f64, s: f64) -> f64 {
        let (v, w) = self.get(n, m);
        c * v + s * w
    }

    /// `R²/μ` times the gradient of the potential term of degree `n` and
    /// order `m` with the unnormalized coefficients `c` and `s`.
    pub(super) fn gradient_term(&self, n: usize, m: usize, c: f64, s: f64) -> Vector3<f64> {
        let (n1, mi) = (n + 1, m as i64);
        let k1 = (n - m + 1) as f64;
        let plus = scaled(-1.0, self.phi(n1, mi + 1));
        let minus = scaled(k1 * (k1 + 1.0), self.phi(n1, mi - 1));
        let z = scaled(-k1, self.phi(n1, mi));
        let x = scaled(0.5, add(plus, minus));
        let y = times_minus_i(scaled(0.5, sub(plus, minus)));
        Vector3::new(real(c, s, x), real(c, s, y), real(c, s, z))
    }

    /// `R³/μ` times the Hessian of the potential term of degree `n` and
    /// order `m` with the unnormalized coefficients `c` and `s`.
    pub(super) fn hessian_term(&self, n: usize, m: usize, c: f64, s: f64) -> Matrix3<f64> {
        let (n2, mi) = (n + 2, m as i64);
        let k1 = (n - m + 1) as f64;
        let k2 = k1 * (k1 + 1.0);
        let pp = self.phi(n2, mi + 2);
        let pm = scaled(-k2, self.phi(n2, mi));
        let mm = scaled(k2 * (k1 + 2.0) * (k1 + 3.0), self.phi(n2, mi - 2));
        let zz = scaled(k2, self.phi(n2, mi));
        let zp = scaled(k1, self.phi(n2, mi + 1));
        let zm = scaled(-k2 * (k1 + 2.0), self.phi(n2, mi - 1));
        let xx = scaled(0.25, add(add(pp, scaled(2.0, pm)), mm));
        let yy = scaled(-0.25, add(sub(pp, scaled(2.0, pm)), mm));
        let xy = times_minus_i(scaled(0.25, sub(pp, mm)));
        let xz = scaled(0.5, add(zp, zm));
        let yz = times_minus_i(scaled(0.5, sub(zp, zm)));
        let (xx, yy, zz) = (real(c, s, xx), real(c, s, yy), real(c, s, zz));
        let (xy, xz, yz) = (real(c, s, xy), real(c, s, xz), real(c, s, yz));
        Matrix3::new(xx, xy, xz, xy, yy, yz, xz, yz, zz)
    }
}

/// `Re((C − i·S)·d)`.
fn real(c: f64, s: f64, d: Complex) -> f64 {
    c * d.0 + s * d.1
}

fn scaled(k: f64, a: Complex) -> Complex {
    (k * a.0, k * a.1)
}

fn add(a: Complex, b: Complex) -> Complex {
    (a.0 + b.0, a.1 + b.1)
}

fn sub(a: Complex, b: Complex) -> Complex {
    (a.0 - b.0, a.1 - b.1)
}

/// `−i·a`.
fn times_minus_i(a: Complex) -> Complex {
    (a.1, -a.0)
}
