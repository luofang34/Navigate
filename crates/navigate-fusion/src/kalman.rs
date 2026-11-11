//! Linear Kalman core over local NED, generic over the state dimension.
//!
//! The six-state layout is `[p_north, p_east, p_down, v_north, v_east,
//! v_down]` in meters and meters per second. The core is generic over the
//! state dimension `N` (ADR-0012). A propagator gives the prediction step
//! and a measurement model gives the update step. Measurement updates use
//! the Joseph-form covariance recursion. The Joseph form keeps the
//! covariance symmetric and positive semidefinite under floating-point
//! rounding. The shorter `(I − KH)P` form does not.
//!
//! [`joseph_covariance_update`] gives the same covariance update to a
//! covariance analysis that has no state mean. [`FogmBias`] models a
//! first-order Gauss–Markov bias.

use nalgebra::{Matrix3, SMatrix, SVector};
use navigate_contract::SymmetricCov3;
use navigate_geodesy::LocalTangentPlane;

use crate::config::FusionConfig;
use crate::rejection::RejectionReason;

pub(crate) type Vec6 = SVector<f64, 6>;
pub(crate) type Mat6 = SMatrix<f64, 6, 6>;

mod models;
mod propagation;

use crate::float_math::sqrt;
pub(crate) use models::{
    AltitudeModel, LinearBlock, MagneticAnomalyModel, MapLinearization, RangeModel,
    check_gradient_information, check_linearization_region,
};
pub use propagation::FogmBias;
pub(crate) use propagation::{ConstantVelocityPropagator, Propagator};

/// A measurement of dimension `M` of the `N`-dimensional state
/// (ADR-0008, ADR-0012).
///
/// The filter core applies every model through one update path. A new
/// navigation means is a new model. The core does not change.
pub(crate) trait MeasurementModel<const M: usize, const N: usize = 6> {
    /// Checks that the model is valid for the state before the gate runs.
    /// The default accepts every state. A map model refuses a state
    /// outside its linearization region or a gradient without
    /// information (ADR-0012).
    fn admissible(
        &self,
        _x: &SVector<f64, N>,
        _p: &SMatrix<f64, N, N>,
        _config: &FusionConfig,
    ) -> Result<(), RejectionReason> {
        Ok(())
    }
    /// The innovation `z − h(x)` at the state mean.
    fn innovation(&self, x: &SVector<f64, N>) -> SVector<f64, M>;
    /// The linearization `∂h/∂x` at the state mean. `None` where the model
    /// has no defined derivative. The core reports `None` as an
    /// implausible-covariance rejection.
    fn jacobian(&self, x: &SVector<f64, N>) -> Option<SMatrix<f64, M, N>>;
    /// The measurement noise covariance.
    fn noise(&self) -> SMatrix<f64, M, M>;
}

/// A gated measurement ready to apply: the innovation, its linearization,
/// the inverted innovation covariance, and the chi-square statistic that
/// the gate judges.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PreparedUpdate<const M: usize, const N: usize> {
    innovation: SVector<f64, M>,
    h: SMatrix<f64, M, N>,
    r: SMatrix<f64, M, M>,
    s_inverse: SMatrix<f64, M, M>,
    pub(crate) chi2: f64,
}

/// Mean and covariance of the `N`-dimensional state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct KalmanState<const N: usize> {
    pub(crate) x: SVector<f64, N>,
    pub(crate) p: SMatrix<f64, N, N>,
}

impl KalmanState<6> {
    /// A zero-mean state anchored at a position fix. The position block
    /// is the fix covariance with zero position–velocity cross terms. The
    /// velocity block is a diagonal diffuse prior. A measurement update
    /// against a diffuse prior that the fix itself centers would show a
    /// certainty that the fix does not have.
    pub(crate) fn from_position_fix(r: &Matrix3<f64>, velocity_variance_m2_per_s2: f64) -> Self {
        let mut p = Mat6::zeros();
        for i in 0..3 {
            for j in 0..3 {
                p[(i, j)] = r[(i, j)];
            }
            p[(i + 3, i + 3)] = velocity_variance_m2_per_s2;
        }
        Self {
            x: Vec6::zeros(),
            p,
        }
    }

    /// Constant-velocity propagation over `dt_s` seconds with
    /// white-acceleration process noise of density `accel_psd` (m²/s³).
    /// [`ConstantVelocityPropagator`] does the work.
    pub(crate) fn propagated(&self, dt_s: f64, accel_psd: f64) -> Self {
        ConstantVelocityPropagator { accel_psd }.propagate(self, dt_s)
    }
}

impl<const N: usize> KalmanState<N> {
    /// Linearizes a model at the mean and computes the innovation and its
    /// gate statistic. The result is `None` when the model has no
    /// derivative at the mean. It is also `None` when the innovation
    /// covariance `S = HPHᵀ + R` has no Cholesky factorization. The
    /// caller reports `None` as an implausible-covariance rejection. The
    /// fixed dimensions `M` and `N` keep the update free of heap
    /// allocation.
    pub(crate) fn prepare_update<const M: usize>(
        &self,
        model: &impl MeasurementModel<M, N>,
    ) -> Option<PreparedUpdate<M, N>> {
        let h = model.jacobian(&self.x)?;
        let r = model.noise();
        let s = h * self.p * h.transpose() + r;
        let s_inverse = s.cholesky()?.inverse();
        let innovation = model.innovation(&self.x);
        let chi2 = innovation.dot(&(s_inverse * innovation));
        Some(PreparedUpdate {
            innovation,
            h,
            r,
            s_inverse,
            chi2,
        })
    }

    /// Applies a prepared measurement with the Joseph-form covariance
    /// update `P' = (I − KH) P (I − KH)ᵀ + K R Kᵀ`.
    pub(crate) fn apply_update<const M: usize>(&self, prepared: &PreparedUpdate<M, N>) -> Self {
        let gain = self.p * prepared.h.transpose() * prepared.s_inverse;
        let x = self.x + gain * prepared.innovation;
        let p = joseph(&self.p, &prepared.h, &prepared.r, &gain);
        Self { x, p }
    }
}

/// The Joseph-form covariance update of `p` with the measurement rows
/// `h` and the noise `r`: `P' = (I − KH)·P·(I − KH)ᵀ + K·R·Kᵀ` with
/// `K = P·Hᵀ·S⁻¹` and `S = H·P·Hᵀ + R`. The result is symmetric. It is
/// `None` when `S` has no Cholesky factorization.
///
/// A covariance analysis without a state mean, such as a Cramér–Rao
/// bound, uses this update. The filters use the same form.
#[must_use]
pub fn joseph_covariance_update<const M: usize, const N: usize>(
    p: &SMatrix<f64, N, N>,
    h: &SMatrix<f64, M, N>,
    r: &SMatrix<f64, M, M>,
) -> Option<SMatrix<f64, N, N>> {
    let s = h * p * h.transpose() + r;
    let gain = p * h.transpose() * s.cholesky()?.inverse();
    let next = joseph(p, h, r, &gain);
    Some((next + next.transpose()) * 0.5)
}

/// `(I − KH)·P·(I − KH)ᵀ + K·R·Kᵀ` for the gain `gain`.
pub(crate) fn joseph<const M: usize, const N: usize>(
    p: &SMatrix<f64, N, N>,
    h: &SMatrix<f64, M, N>,
    r: &SMatrix<f64, M, M>,
    gain: &SMatrix<f64, N, M>,
) -> SMatrix<f64, N, N> {
    let identity_minus_kh = SMatrix::<f64, N, N>::identity() - gain * h;
    identity_minus_kh * p * identity_minus_kh.transpose() + gain * r * gain.transpose()
}

pub(crate) fn cov3_to_matrix(cov: &SymmetricCov3) -> Matrix3<f64> {
    let [xx, xy, xz, yy, yz, zz] = cov.upper_triangle();
    Matrix3::new(xx, xy, xz, xy, yy, yz, xz, yz, zz)
}

/// Collapses a full 3×3 matrix onto the contract's upper-triangle form.
pub(crate) fn matrix_to_cov3(m: &Matrix3<f64>) -> SymmetricCov3 {
    SymmetricCov3::from_upper_triangle([
        m[(0, 0)],
        m[(0, 1)],
        m[(0, 2)],
        m[(1, 1)],
        m[(1, 2)],
        m[(2, 2)],
    ])
}

/// The rotation that maps a NED vector of `from` into the NED axes of
/// `to`.
pub(crate) fn rotation_between(from: &LocalTangentPlane, to: &LocalTangentPlane) -> Matrix3<f64> {
    let rows = from.ned_rotation_to(to);
    Matrix3::new(
        rows[0][0], rows[0][1], rows[0][2], rows[1][0], rows[1][1], rows[1][2], rows[2][0],
        rows[2][1], rows[2][2],
    )
}

/// Whether a symmetric 3×3 matrix is positive definite: a strictly
/// positive diagonal and strictly positive leading principal minors
/// (Sylvester's criterion). Definiteness, not semidefiniteness — a
/// zero-variance direction would let a single observation collapse the
/// published uncertainty to exactly zero. A closed-form test, so
/// admission never depends on iterative convergence.
pub(crate) fn is_positive_definite(m: &Matrix3<f64>) -> bool {
    let a = m[(0, 0)];
    let b = m[(0, 1)];
    let d = m[(1, 1)];
    let f = m[(2, 2)];
    a > 0.0 && d > 0.0 && f > 0.0 && a * d - b * b > 0.0 && m.determinant() > 0.0
}

/// One-sigma horizontal uncertainty: the square root of the largest
/// eigenvalue of the leading 2×2 north/east covariance block, in closed
/// form for a symmetric 2×2 matrix.
pub(crate) fn horizontal_1sigma_m<const N: usize>(p: &SMatrix<f64, N, N>) -> f64 {
    let a = p[(0, 0)];
    let b = p[(0, 1)];
    let c = p[(1, 1)];
    let mean = 0.5 * (a + c);
    let radius = sqrt(0.25 * (a - c) * (a - c) + b * b);
    sqrt((mean + radius).max(0.0))
}

/// One-sigma vertical uncertainty: the square root of the down variance.
pub(crate) fn vertical_1sigma_m<const N: usize>(p: &SMatrix<f64, N, N>) -> f64 {
    sqrt(p[(2, 2)].max(0.0))
}

#[cfg(test)]
mod tests;
