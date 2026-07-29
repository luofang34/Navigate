//! Six-state linear Kalman core over local NED.
//!
//! State ordering is `[p_north, p_east, p_down, v_north, v_east, v_down]`
//! in meters and meters per second. Propagation is constant-velocity with
//! white-acceleration process noise; measurement updates use the
//! Joseph-form covariance recursion, chosen over the shorter `(I − KH)P`
//! form because it preserves symmetry and positive semidefiniteness under
//! floating-point rounding.

use nalgebra::{Matrix3, SMatrix, SVector, Vector3};
use navigate_contract::SymmetricCov3;

pub(crate) type Vec6 = SVector<f64, 6>;
pub(crate) type Mat6 = SMatrix<f64, 6, 6>;
type Mat3x6 = SMatrix<f64, 3, 6>;

/// Which 3-vector block of the state a measurement observes.
#[derive(Debug, Clone, Copy)]
pub(crate) enum MeasurementBlock {
    /// Position rows: `H = [I₃ 0₃]`.
    Position,
    /// Velocity rows: `H = [0₃ I₃]`.
    Velocity,
}

impl MeasurementBlock {
    fn observation_matrix(self) -> Mat3x6 {
        let offset = match self {
            Self::Position => 0,
            Self::Velocity => 3,
        };
        let mut h = Mat3x6::zeros();
        for i in 0..3 {
            h[(i, i + offset)] = 1.0;
        }
        h
    }
}

/// A gated measurement ready to apply: the innovation, the inverted
/// innovation covariance, and the chi-square statistic the gate judges.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PreparedUpdate {
    innovation: Vector3<f64>,
    s_inverse: Matrix3<f64>,
    pub(crate) chi2: f64,
}

/// Mean and covariance of the six-dimensional state.
#[derive(Debug, Clone, Copy)]
pub(crate) struct KalmanState {
    pub(crate) x: Vec6,
    pub(crate) p: Mat6,
}

impl KalmanState {
    /// A zero-mean state anchored at a position fix: the position block
    /// is the fix covariance itself with zero position–velocity cross
    /// terms, and the velocity block is a diagonal diffuse prior. A
    /// measurement update against a diffuse prior the fix itself centers
    /// would fake certainty the fix does not carry.
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
    /// white-acceleration process noise of density `accel_psd` (m²/s³):
    /// `Q_pp = q·dt³/3`, `Q_pv = q·dt²/2`, `Q_vv = q·dt` per axis.
    pub(crate) fn propagated(&self, dt_s: f64, accel_psd: f64) -> Self {
        if dt_s <= 0.0 {
            return *self;
        }
        let q_pp = accel_psd * dt_s * dt_s * dt_s / 3.0;
        let q_pv = accel_psd * dt_s * dt_s / 2.0;
        let q_vv = accel_psd * dt_s;
        let mut f = Mat6::identity();
        let mut q = Mat6::zeros();
        for i in 0..3 {
            f[(i, i + 3)] = dt_s;
            q[(i, i)] = q_pp;
            q[(i, i + 3)] = q_pv;
            q[(i + 3, i)] = q_pv;
            q[(i + 3, i + 3)] = q_vv;
        }
        Self {
            x: f * self.x,
            p: f * self.p * f.transpose() + q,
        }
    }

    /// Computes the innovation and its gate statistic against a 3-vector
    /// measurement. `None` when the innovation covariance `S = HPHᵀ + R`
    /// has no Cholesky factorization — surfaced to the caller as an
    /// implausible-covariance rejection, never a panic.
    pub(crate) fn prepare_update(
        &self,
        block: MeasurementBlock,
        z: &Vector3<f64>,
        r: &Matrix3<f64>,
    ) -> Option<PreparedUpdate> {
        let h = block.observation_matrix();
        let s = h * self.p * h.transpose() + r;
        let s_inverse = s.cholesky()?.inverse();
        let innovation = z - h * self.x;
        let chi2 = innovation.dot(&(s_inverse * innovation));
        Some(PreparedUpdate {
            innovation,
            s_inverse,
            chi2,
        })
    }

    /// Applies a prepared measurement with the Joseph-form covariance
    /// update `P' = (I − KH) P (I − KH)ᵀ + K R Kᵀ`.
    pub(crate) fn apply_update(
        &self,
        block: MeasurementBlock,
        r: &Matrix3<f64>,
        prepared: &PreparedUpdate,
    ) -> Self {
        let h = block.observation_matrix();
        let gain = self.p * h.transpose() * prepared.s_inverse;
        let x = self.x + gain * prepared.innovation;
        let identity_minus_kh = Mat6::identity() - gain * h;
        let p = identity_minus_kh * self.p * identity_minus_kh.transpose()
            + gain * r * gain.transpose();
        Self { x, p }
    }
}

/// Expands a symmetric upper triangle into a full 3×3 matrix.
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
/// eigenvalue of the 2×2 north/east covariance block, in closed form for
/// a symmetric 2×2 matrix.
pub(crate) fn horizontal_1sigma_m(p: &Mat6) -> f64 {
    let a = p[(0, 0)];
    let b = p[(0, 1)];
    let c = p[(1, 1)];
    let mean = 0.5 * (a + c);
    let radius = (0.25 * (a - c) * (a - c) + b * b).sqrt();
    (mean + radius).max(0.0).sqrt()
}

/// One-sigma vertical uncertainty: the square root of the down variance.
pub(crate) fn vertical_1sigma_m(p: &Mat6) -> f64 {
    p[(2, 2)].max(0.0).sqrt()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use nalgebra::{Matrix3, Vector3};

    use super::{
        KalmanState, Mat6, MeasurementBlock, Vec6, horizontal_1sigma_m, is_positive_definite,
        vertical_1sigma_m,
    };

    #[test]
    fn propagation_inflates_position_uncertainty() {
        let state = KalmanState::from_position_fix(&Matrix3::from_diagonal_element(4.0), 1.0);
        let advanced = state.propagated(2.0, 0.5);
        // P_pp' = P_pp + dt²·P_vv + q·dt³/3 = 4 + 4 + 4/3.
        assert!((advanced.p[(0, 0)] - (4.0 + 4.0 + 4.0 / 3.0)).abs() < 1e-9);
        assert!(advanced.p[(0, 0)] > state.p[(0, 0)]);
    }

    #[test]
    fn update_moves_the_state_toward_the_measurement() {
        let state = KalmanState::from_position_fix(&Matrix3::from_diagonal_element(100.0), 10.0);
        let z = Vector3::new(10.0, 0.0, 0.0);
        let r = Matrix3::identity();
        let prepared = state
            .prepare_update(MeasurementBlock::Position, &z, &r)
            .expect("invertible innovation covariance");
        let updated = state.apply_update(MeasurementBlock::Position, &r, &prepared);
        assert!((updated.x[0] - 10.0 * 100.0 / 101.0).abs() < 1e-9);
        assert!(updated.p[(0, 0)] < state.p[(0, 0)]);
        // Joseph form keeps the covariance symmetric.
        assert!((updated.p - updated.p.transpose()).norm() < 1e-12);
    }

    #[test]
    fn definiteness_screen_refuses_impossible_and_singular_covariances() {
        let overcorrelated = Matrix3::new(25.0, 100.0, 0.0, 100.0, 25.0, 0.0, 0.0, 0.0, 25.0);
        assert!(!is_positive_definite(&overcorrelated));
        assert!(is_positive_definite(&Matrix3::identity()));
        assert!(!is_positive_definite(&Matrix3::from_diagonal_element(-1.0)));
        // Zero and rank-deficient covariances claim a direction of exact
        // certainty; the definite screen refuses them.
        assert!(!is_positive_definite(&Matrix3::zeros()));
        assert!(!is_positive_definite(&Matrix3::new(
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
        )));
    }

    #[test]
    fn singular_innovation_covariance_is_reported_not_inverted() {
        let degenerate = KalmanState {
            x: Vec6::zeros(),
            p: Mat6::zeros(),
        };
        let prepared = degenerate.prepare_update(
            MeasurementBlock::Position,
            &Vector3::zeros(),
            &Matrix3::zeros(),
        );
        assert!(prepared.is_none());
    }

    #[test]
    fn sigma_extraction_reads_the_horizontal_block() {
        let mut p = Mat6::zeros();
        p[(0, 0)] = 9.0;
        p[(1, 1)] = 4.0;
        p[(2, 2)] = 16.0;
        assert!((horizontal_1sigma_m(&p) - 3.0).abs() < 1e-12);
        assert!((vertical_1sigma_m(&p) - 4.0).abs() < 1e-12);
        // Correlation rotates the ellipse: the major axis exceeds both
        // diagonal sigmas.
        p[(0, 1)] = 5.0;
        p[(1, 0)] = 5.0;
        assert!(horizontal_1sigma_m(&p) > 3.0);
    }
}
