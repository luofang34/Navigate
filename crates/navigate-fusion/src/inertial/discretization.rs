//! Matrix exponential and the Van Loan discretization of the error-state
//! dynamics (ADR-0014).

use nalgebra::{Const, DimMin, SMatrix};

use super::nominal::{Mat17, N};
use crate::float_math::powi;

/// Degree of the diagonal Padé approximant.
const PADE_DEGREE: usize = 6;

/// The scaled matrix must have an infinity norm at or below this value.
/// Then the `[6/6]` approximant is accurate to the `f64` rounding level.
const SCALED_NORM_BOUND: f64 = 0.5;

/// Upper limit of the squaring count. A larger count means a norm that
/// is not physical for one propagation step.
const MAX_SQUARINGS: u32 = 60;

/// Dimension of the Van Loan block matrix.
const VAN_LOAN_DIM: usize = 2 * N;

/// The matrix exponential `exp(a)` with the `[6/6]` Padé approximant
/// and scaling and squaring. `None` when `a` is not finite, its norm
/// needs more than [`MAX_SQUARINGS`] squarings, or the denominator is
/// singular.
pub(crate) fn expm<const D: usize>(a: &SMatrix<f64, D, D>) -> Option<SMatrix<f64, D, D>>
where
    Const<D>: DimMin<Const<D>, Output = Const<D>>,
{
    if !a.iter().all(|v| v.is_finite()) {
        return None;
    }
    let mut norm = infinity_norm(a);
    let mut squarings = 0_u32;
    while norm > SCALED_NORM_BOUND {
        norm *= 0.5;
        squarings = squarings.wrapping_add(1);
        if squarings > MAX_SQUARINGS {
            return None;
        }
    }
    let x = a * powi(0.5, i32::try_from(squarings).ok()?);
    let identity = SMatrix::<f64, D, D>::identity();
    let mut power = identity;
    let mut numerator = identity;
    let mut denominator = identity;
    let mut coefficient = 1.0;
    let mut sign = 1.0;
    for k in 1..=PADE_DEGREE {
        // c_k = c_(k−1)·(q − k + 1) / (k·(2q − k + 1)) for q = 6.
        coefficient *= (PADE_DEGREE - k + 1) as f64 / (k * (2 * PADE_DEGREE - k + 1)) as f64;
        sign = -sign;
        power = x * power;
        numerator += power * coefficient;
        denominator += power * (sign * coefficient);
    }
    let mut result = denominator.lu().solve(&numerator)?;
    for _ in 0..squarings {
        result = result * result;
    }
    Some(result)
}

fn infinity_norm<const D: usize>(a: &SMatrix<f64, D, D>) -> f64 {
    a.row_iter()
        .map(|row| row.iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// The Van Loan discretization of `ẋ = F·x + w` with `E[w·wᵀ] = Q_c·δ`
/// over `dt_s` seconds. With `M = [[−F, Q_c], [0, Fᵀ]]·dt` and
/// `E = exp(M)`, the result is `Φ = E₂₂ᵀ` and the symmetric
/// `Q_d = Φ·E₁₂`. `None` when [`expm`] fails.
pub(crate) fn van_loan(f: &Mat17, q_c: &Mat17, dt_s: f64) -> Option<(Mat17, Mat17)> {
    let mut m = SMatrix::<f64, VAN_LOAN_DIM, VAN_LOAN_DIM>::zeros();
    m.fixed_view_mut::<N, N>(0, 0).copy_from(&(-f * dt_s));
    m.fixed_view_mut::<N, N>(0, N).copy_from(&(q_c * dt_s));
    m.fixed_view_mut::<N, N>(N, N)
        .copy_from(&(f.transpose() * dt_s));
    let e = expm(&m)?;
    let phi = e.fixed_view::<N, N>(N, N).transpose();
    let q_d = phi * e.fixed_view::<N, N>(0, N);
    Some((phi, (q_d + q_d.transpose()) * 0.5))
}
