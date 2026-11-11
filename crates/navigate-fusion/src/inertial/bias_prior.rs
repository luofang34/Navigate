//! A prior on the IMU biases for the initialization, such as the result
//! of a static self-alignment (ADR-0014).
//!
//! Without a prior the filter starts each bias at zero with its
//! stationary variance. A self-alignment measures the biases before the
//! vehicle moves, so it knows them better than that. The prior gives the
//! filter that knowledge at the initialization.

use nalgebra::Vector3;
use navigate_contract::SymmetricCov3;

use super::InertialNavigationFilter;
use super::nominal::{ACCEL_BIAS, ErrorMat, GYRO_BIAS, InertialNominal};
use crate::admission::check_cov3;
use crate::kalman::cov3_to_matrix;
use crate::rejection::RejectionReason;

/// The accelerometer and gyroscope biases at the initialization, with
/// their covariances, in the body axes of the IMU.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ImuBiasPrior {
    /// Accelerometer bias in m/s², in the body axes.
    pub accel_bias_body_mps2: [f64; 3],
    /// Covariance of the accelerometer bias in (m/s²)², in the body axes.
    pub accel_cov_body: SymmetricCov3,
    /// Gyroscope bias in rad/s, in the body axes.
    pub gyro_bias_body_rad_s: [f64; 3],
    /// Covariance of the gyroscope bias in (rad/s)², in the body axes.
    pub gyro_cov_body: SymmetricCov3,
}

impl ImuBiasPrior {
    /// Builds a prior from its parts.
    #[must_use]
    pub const fn new(
        accel_bias_body_mps2: [f64; 3],
        accel_cov_body: SymmetricCov3,
        gyro_bias_body_rad_s: [f64; 3],
        gyro_cov_body: SymmetricCov3,
    ) -> Self {
        Self {
            accel_bias_body_mps2,
            accel_cov_body,
            gyro_bias_body_rad_s,
            gyro_cov_body,
        }
    }

    /// Refuses a bias that is not finite and a covariance that is not
    /// finite and positive definite.
    fn check(&self) -> Result<(), RejectionReason> {
        let finite = self
            .accel_bias_body_mps2
            .iter()
            .chain(&self.gyro_bias_body_rad_s)
            .all(|value| value.is_finite());
        if !finite {
            return Err(RejectionReason::NonFiniteValue);
        }
        check_cov3(&self.accel_cov_body)?;
        check_cov3(&self.gyro_cov_body)
    }

    /// Puts the biases into `nominal` and the covariances into `p`, in
    /// place of the zero biases and their stationary variances.
    pub(super) fn apply(&self, nominal: &mut InertialNominal, p: &mut ErrorMat) {
        nominal.b_a = Vector3::from(self.accel_bias_body_mps2);
        nominal.b_g = Vector3::from(self.gyro_bias_body_rad_s);
        p.fixed_view_mut::<3, 3>(ACCEL_BIAS, ACCEL_BIAS)
            .copy_from(&cov3_to_matrix(&self.accel_cov_body));
        p.fixed_view_mut::<3, 3>(GYRO_BIAS, GYRO_BIAS)
            .copy_from(&cov3_to_matrix(&self.gyro_cov_body));
    }
}

impl InertialNavigationFilter {
    /// Sets the IMU bias prior of the next initialization, or removes it
    /// with `None`. It does not change a state that the filter already
    /// has. The prior stays for the initializations after a
    /// [`Self::reset`] until the next call.
    ///
    /// Each bias is a first-order Gauss–Markov state, so its estimate
    /// decays toward zero with the correlation time of
    /// [`crate::InertialConfig::accel_bias`] and
    /// [`crate::InertialConfig::gyro_bias`], and its variance moves
    /// toward the stationary variance.
    ///
    /// # Errors
    ///
    /// [`RejectionReason::NonFiniteValue`] for a bias that is not finite
    /// and [`RejectionReason::ImplausibleCovariance`] for a covariance
    /// that is not finite and positive definite. A refused prior leaves
    /// the previous one in place.
    pub fn set_imu_bias_prior(
        &mut self,
        prior: Option<ImuBiasPrior>,
    ) -> Result<(), RejectionReason> {
        if let Some(prior) = &prior {
            let checked = prior.check();
            self.record_failure(checked)?;
        }
        self.bias_prior = prior;
        Ok(())
    }
}
