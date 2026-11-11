//! Matrix exponential and the Van Loan discretization of the error-state
//! dynamics (ADR-0014).

#[cfg(test)]
use nalgebra::SMatrix;
use nalgebra::{Const, DimMin};

use super::nominal::ErrorMat;
use crate::float_math::powi;

/// Degree of the diagonal Padé approximant.
const PADE_DEGREE: usize = 6;

/// The scaled matrix must have an infinity norm at or below this value.
/// Then the `[6/6]` approximant is accurate to the `f64` rounding level.
const SCALED_NORM_BOUND: f64 = 0.5;

/// Upper limit of the squaring count. A larger count means a norm that
/// is not physical for one propagation step.
const MAX_SQUARINGS: u32 = 60;

/// Dimension of the dense Van Loan block matrix of the tests.
#[cfg(test)]
const VAN_LOAN_DIM: usize = 2 * super::nominal::FULL_STATES;

/// The matrix exponential `exp(a)` of a dense matrix, the reference of
/// the block form in the tests, with the `[6/6]` Padé approximant
/// and scaling and squaring. `None` when `a` is not finite, its norm
/// needs more than [`MAX_SQUARINGS`] squarings, or the denominator is
/// singular.
#[cfg(test)]
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

#[cfg(test)]
fn infinity_norm<const D: usize>(a: &SMatrix<f64, D, D>) -> f64 {
    a.row_iter()
        .map(|row| row.iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// The Van Loan discretization of `ẋ = F·x + w` with `E[w·wᵀ] = Q_c·δ`
/// over `dt_s` seconds. With `M = [[−F, Q_c], [0, Fᵀ]]·dt` and
/// `E = exp(M)`, the result is `Φ = E₂₂ᵀ` and the symmetric
/// `Q_d = Φ·E₁₂`. `None` when the exponential fails, as for [`expm`].
///
/// `M` is block upper-triangular, and so is each matrix of the `[6/6]`
/// Padé approximant with scaling and squaring. Thus the exponential runs
/// on the three `N×N` blocks and needs no `2N×2N` matrix: the powers are
/// `[[A,B],[0,C]]·[[P,Q],[0,R]] = [[A·P, A·Q + B·R], [0, C·R]]`, the
/// solve of `D·X = Nᵤ` is `X₁₁ = D₁₁⁻¹·N₁₁`, `X₂₂ = D₂₂⁻¹·N₂₂`, and
/// `X₁₂ = D₁₁⁻¹·(N₁₂ − D₁₂·X₂₂)`, and a squaring is
/// `[[X₁₁², X₁₁·X₁₂ + X₁₂·X₂₂], [0, X₂₂²]]`.
pub(crate) fn van_loan<const N: usize>(
    f: &ErrorMat<N>,
    q_c: &ErrorMat<N>,
    dt_s: f64,
) -> Option<(ErrorMat<N>, ErrorMat<N>)>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    let m = Upper {
        a: -f * dt_s,
        b: q_c * dt_s,
        c: f.transpose() * dt_s,
    };
    let e = block_expm(&m)?;
    let phi = e.c.transpose();
    let q_d = phi * e.b;
    Some((phi, (q_d + q_d.transpose()) * 0.5))
}

/// A block upper-triangular matrix `[[a, b], [0, c]]` of `N×N` blocks.
#[derive(Clone, Copy)]
struct Upper<const N: usize> {
    a: ErrorMat<N>,
    b: ErrorMat<N>,
    c: ErrorMat<N>,
}

impl<const N: usize> Upper<N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    fn identity() -> Self {
        Self {
            a: ErrorMat::<N>::identity(),
            b: ErrorMat::<N>::zeros(),
            c: ErrorMat::<N>::identity(),
        }
    }

    fn mul(&self, other: &Self) -> Self {
        Self {
            a: self.a * other.a,
            b: self.a * other.b + self.b * other.c,
            c: self.c * other.c,
        }
    }

    fn scaled(&self, k: f64) -> Self {
        Self {
            a: self.a * k,
            b: self.b * k,
            c: self.c * k,
        }
    }

    fn add_scaled(&mut self, other: &Self, k: f64) {
        self.a += other.a * k;
        self.b += other.b * k;
        self.c += other.c * k;
    }

    fn is_finite(&self) -> bool {
        [&self.a, &self.b, &self.c]
            .iter()
            .all(|block| block.iter().all(|v| v.is_finite()))
    }

    /// The infinity norm: the largest absolute row sum of the full
    /// matrix.
    fn infinity_norm(&self) -> f64 {
        let top = self
            .a
            .row_iter()
            .zip(self.b.row_iter())
            .map(|(a, b)| a.iter().chain(b.iter()).map(|v| v.abs()).sum::<f64>());
        let bottom = self
            .c
            .row_iter()
            .map(|row| row.iter().map(|v| v.abs()).sum::<f64>());
        top.chain(bottom).fold(0.0, f64::max)
    }

    /// `X` with `self·X = rhs`, from the two diagonal solves.
    fn solve(&self, rhs: &Self) -> Option<Self> {
        let lu_a = self.a.lu();
        let a = lu_a.solve(&rhs.a)?;
        let c = self.c.lu().solve(&rhs.c)?;
        let b = lu_a.solve(&(rhs.b - self.b * c))?;
        Some(Self { a, b, c })
    }
}

/// [`expm`] of a block upper-triangular matrix, with the same `[6/6]`
/// Padé approximant, scaling, and squaring on the blocks.
fn block_expm<const N: usize>(m: &Upper<N>) -> Option<Upper<N>>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    if !m.is_finite() {
        return None;
    }
    let mut norm = m.infinity_norm();
    let mut squarings = 0_u32;
    while norm > SCALED_NORM_BOUND {
        norm *= 0.5;
        squarings = squarings.wrapping_add(1);
        if squarings > MAX_SQUARINGS {
            return None;
        }
    }
    let x = m.scaled(powi(0.5, i32::try_from(squarings).ok()?));
    let mut power = Upper::identity();
    let mut numerator = Upper::identity();
    let mut denominator = Upper::identity();
    let mut coefficient = 1.0;
    let mut sign = 1.0;
    for k in 1..=PADE_DEGREE {
        // c_k = c_(k−1)·(q − k + 1) / (k·(2q − k + 1)) for q = 6.
        coefficient *= (PADE_DEGREE - k + 1) as f64 / (k * (2 * PADE_DEGREE - k + 1)) as f64;
        sign = -sign;
        power = x.mul(&power);
        numerator.add_scaled(&power, coefficient);
        denominator.add_scaled(&power, sign * coefficient);
    }
    let mut result = denominator.solve(&numerator)?;
    for _ in 0..squarings {
        result = result.mul(&result);
    }
    Some(result)
}

/// The dense `2N×2N` Van Loan discretization, the reference of the
/// block form in the tests.
#[cfg(test)]
pub(crate) fn van_loan_dense(
    f: &ErrorMat,
    q_c: &ErrorMat,
    dt_s: f64,
) -> Option<(ErrorMat, ErrorMat)> {
    const N: usize = super::nominal::FULL_STATES;
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
