//! The nominal state, the layout of the error state, and the closed-loop
//! injection with its covariance reset (ADR-0014).
//!
//! The attitude error is a right perturbation in the body axes:
//! `C_true = C_nom·exp([δθ×])` and `q_true = q_nom ⊗ exp(δθ/2)`. The bias,
//! empirical acceleration, and clock errors are additive:
//! `b_true = b_nom + δb`.

use crate::float_math::atan2;
use nalgebra::{Matrix3, Quaternion, SMatrix, SVector, UnitQuaternion, Vector3};
use navigate_contract::AttitudeQuaternion;

use super::layout::{MAX_CORRELATED_BLOCKS, StateLayout};

/// Dimension of the error state of [`StateLayout::FULL`].
pub(crate) const FULL_STATES: usize = 22;
/// First index of the position error in meters, in the frame axes.
pub(crate) const POS: usize = 0;
/// First index of the velocity error in m/s, in the frame axes.
pub(crate) const VEL: usize = 3;
/// First index of the attitude error in radians, in the body axes.
pub(crate) const ATT: usize = 6;
/// First index of the accelerometer bias error in m/s², in the body axes.
pub(crate) const ACCEL_BIAS: usize = 9;
/// First index of the gyroscope bias error in rad/s, in the body axes.
pub(crate) const GYRO_BIAS: usize = 12;
/// Index of the barometric altitude bias error of [`StateLayout::FULL`].
#[cfg(test)]
pub(crate) const ALT_BIAS: usize = 15;
/// Index of the magnetic map bias error of [`StateLayout::FULL`].
#[cfg(test)]
pub(crate) const MAG_BIAS: usize = 16;
/// First index of the empirical acceleration error of
/// [`StateLayout::FULL`].
#[cfg(test)]
pub(crate) const DMC: usize = 17;
/// Index of the receiver clock bias error of [`StateLayout::FULL`].
#[cfg(test)]
pub(crate) const CLOCK_BIAS: usize = 20;
/// Index of the receiver clock drift error of [`StateLayout::FULL`].
#[cfg(test)]
pub(crate) const CLOCK_DRIFT: usize = 21;

/// An error-state vector of dimension `N`.
pub(crate) type ErrorVec<const N: usize = FULL_STATES> = SVector<f64, N>;
/// An error-state matrix of dimension `N`.
pub(crate) type ErrorMat<const N: usize = FULL_STATES> = SMatrix<f64, N, N>;

/// Below this rotation-vector norm, [`rotation_vector`] uses the
/// small-angle form.
const SMALL_ANGLE: f64 = 1e-12;

/// The nominal state of the strapdown mechanization.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct InertialNominal {
    /// Position in meters in the active frame.
    pub(crate) p: Vector3<f64>,
    /// Velocity relative to the active frame in m/s, in the frame axes.
    pub(crate) v: Vector3<f64>,
    /// Attitude from the body axes to the frame axes.
    pub(crate) q: UnitQuaternion<f64>,
    /// Accelerometer bias in m/s², in the body axes.
    pub(crate) b_a: Vector3<f64>,
    /// Gyroscope bias in rad/s, in the body axes.
    pub(crate) b_g: Vector3<f64>,
    /// Barometric altitude bias in meters.
    pub(crate) b_alt: f64,
    /// Scalar magnetic map bias in tesla.
    pub(crate) b_mag: f64,
    /// Empirical acceleration of the dynamic model compensation in m/s²,
    /// in the frame axes.
    pub(crate) a_dmc: Vector3<f64>,
    /// Receiver clock range bias `c·δt` in meters.
    pub(crate) clock_bias_m: f64,
    /// Receiver clock range-rate drift `c·δṫ` in m/s.
    pub(crate) clock_drift_mps: f64,
    /// The mean of each correlated-error block, in the axes of its kind.
    pub(crate) correlated: [Vector3<f64>; MAX_CORRELATED_BLOCKS],
}

impl InertialNominal {
    /// The nominal state at `p`, `v`, and `q` with each bias, the
    /// empirical acceleration, and the clock at zero.
    pub(crate) fn at(p: Vector3<f64>, v: Vector3<f64>, q: UnitQuaternion<f64>) -> Self {
        Self {
            p,
            v,
            q,
            b_a: Vector3::zeros(),
            b_g: Vector3::zeros(),
            b_alt: 0.0,
            b_mag: 0.0,
            a_dmc: Vector3::zeros(),
            clock_bias_m: 0.0,
            clock_drift_mps: 0.0,
            correlated: [Vector3::zeros(); MAX_CORRELATED_BLOCKS],
        }
    }

    /// The nominal state with the error estimate `dx` of `layout`
    /// injected. A block that the layout does not have keeps its value.
    pub(crate) fn inject<const N: usize>(&self, dx: &ErrorVec<N>, layout: &StateLayout) -> Self {
        let scalar = |offset: Option<usize>, value: f64| offset.map_or(value, |i| value + dx[i]);
        Self {
            p: self.p + dx.fixed_rows::<3>(POS),
            v: self.v + dx.fixed_rows::<3>(VEL),
            q: renormalized(self.q * exp_rotation(&dx.fixed_rows::<3>(ATT).into_owned())),
            b_a: self.b_a + dx.fixed_rows::<3>(ACCEL_BIAS),
            b_g: self.b_g + dx.fixed_rows::<3>(GYRO_BIAS),
            b_alt: scalar(layout.alt_bias(), self.b_alt),
            b_mag: scalar(layout.mag_bias(), self.b_mag),
            a_dmc: layout
                .dmc()
                .map_or(self.a_dmc, |i| self.a_dmc + dx.fixed_rows::<3>(i)),
            clock_bias_m: scalar(layout.clock(), self.clock_bias_m),
            clock_drift_mps: scalar(layout.clock().map(|i| i + 1), self.clock_drift_mps),
            correlated: inject_correlated(&self.correlated, dx, layout),
        }
    }

    /// The rotation matrix from the body axes to the frame axes.
    pub(crate) fn c_b_to_f(&self) -> Matrix3<f64> {
        self.q.to_rotation_matrix().into_inner()
    }
}

/// The correlated-error means with the estimate `dx` of each block of
/// `layout` added.
fn inject_correlated<const N: usize>(
    means: &[Vector3<f64>; MAX_CORRELATED_BLOCKS],
    dx: &ErrorVec<N>,
    layout: &StateLayout,
) -> [Vector3<f64>; MAX_CORRELATED_BLOCKS] {
    let mut injected = *means;
    for (mean, offset) in injected.iter_mut().zip(layout.correlated()) {
        if let Some(i) = offset {
            *mean += dx.fixed_rows::<3>(i);
        }
    }
    injected
}

/// The covariance of the error state after the injection of `dx`:
/// `P' = J·P·Jᵀ` with `J = I` except `J_θθ = I − ½·[δθ×]`. The result is
/// symmetric.
pub(crate) fn reset_covariance<const N: usize>(p: &ErrorMat<N>, dx: &ErrorVec<N>) -> ErrorMat<N> {
    let j = reset_jacobian(&dx.fixed_rows::<3>(ATT).into_owned());
    let reset = j * p * j.transpose();
    (reset + reset.transpose()) * 0.5
}

/// The Jacobian of the closed-loop reset for the attitude error
/// `dtheta`.
pub(crate) fn reset_jacobian<const N: usize>(dtheta: &Vector3<f64>) -> ErrorMat<N> {
    let mut j = ErrorMat::<N>::identity();
    j.fixed_view_mut::<3, 3>(ATT, ATT)
        .copy_from(&(Matrix3::identity() - 0.5 * dtheta.cross_matrix()));
    j
}

/// The unit quaternion of the rotation vector `theta`, that is
/// `exp(θ/2)`.
pub(crate) fn exp_rotation(theta: &Vector3<f64>) -> UnitQuaternion<f64> {
    UnitQuaternion::from_scaled_axis(*theta)
}

/// The rotation vector `2·log(q)` of `q` in the hemisphere `w ≥ 0`. The
/// result has a norm in `[0, π]`.
pub(crate) fn rotation_vector(q: &UnitQuaternion<f64>) -> Vector3<f64> {
    let raw = q.quaternion();
    let (w, v) = if raw.w < 0.0 {
        (-raw.w, -raw.imag())
    } else {
        (raw.w, raw.imag())
    };
    let s = v.norm();
    if s < SMALL_ANGLE {
        return 2.0 * v;
    }
    v * (2.0 * atan2(s, w) / s)
}

/// Removes the rounding drift of the quaternion norm.
pub(crate) fn renormalized(q: UnitQuaternion<f64>) -> UnitQuaternion<f64> {
    UnitQuaternion::new_normalize(q.into_inner())
}

/// The nalgebra form of a contract quaternion. The contract and nalgebra
/// both put the scalar first in the constructor.
pub(crate) fn to_unit_quaternion(q: &AttitudeQuaternion) -> UnitQuaternion<f64> {
    UnitQuaternion::new_normalize(Quaternion::new(q.w, q.x, q.y, q.z))
}

/// The contract form of a unit quaternion.
pub(crate) fn to_contract(q: &UnitQuaternion<f64>) -> AttitudeQuaternion {
    let raw = q.quaternion();
    AttitudeQuaternion::new(raw.w, raw.i, raw.j, raw.k)
}
