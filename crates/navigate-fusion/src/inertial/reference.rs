//! The error covariance of the inertial filter along a known reference
//! trajectory (ADR-0014, ADR-0016).
//!
//! A covariance analysis, for example a posterior Cramér–Rao bound,
//! propagates the error covariance of the filter along a reference
//! trajectory without a filter state. This module gives the layout of
//! the error states, the initial covariance of the filter,
//! and the transition `Φ` and the discrete process noise `Q_d` at one
//! point of the trajectory. The filter propagates its covariance with the
//! same transition, so the analysis and the filter use one model.

use nalgebra::{SMatrix, Vector3};
use navigate_astrodynamics::GravityField;
use navigate_contract::{AttitudeQuaternion, SymmetricCov3};

use super::config::InertialConfig;
use super::mechanization::{ImuRates, transition};
use super::nominal::{
    ACCEL_BIAS, ALT_BIAS, ATT, CLOCK_BIAS, CLOCK_DRIFT, DMC, GYRO_BIAS, InertialNominal, MAG_BIAS,
    N, POS, VEL, to_unit_quaternion,
};
use super::stochastic;
use crate::admission::{check_cov3, is_unit_quaternion};
use crate::kalman::cov3_to_matrix;
use crate::rejection::RejectionReason;

/// Number of error states.
pub const ERROR_STATES: usize = N;
/// First index of the position error in meters, in the frame axes.
pub const ERROR_POSITION: usize = POS;
/// First index of the velocity error in m/s, in the frame axes.
pub const ERROR_VELOCITY: usize = VEL;
/// First index of the attitude error in radians, in the body axes.
pub const ERROR_ATTITUDE: usize = ATT;
/// First index of the accelerometer bias error in m/s², in the body axes.
pub const ERROR_ACCEL_BIAS: usize = ACCEL_BIAS;
/// First index of the gyroscope bias error in rad/s, in the body axes.
pub const ERROR_GYRO_BIAS: usize = GYRO_BIAS;
/// Index of the barometric altitude bias error in meters.
pub const ERROR_ALT_BIAS: usize = ALT_BIAS;
/// Index of the scalar magnetic map bias error in tesla.
pub const ERROR_MAG_BIAS: usize = MAG_BIAS;
/// First index of the empirical acceleration error of the dynamic model
/// compensation in m/s², in the frame axes.
pub const ERROR_DMC: usize = DMC;
/// Index of the receiver clock range bias error `c·δt` in meters.
pub const ERROR_CLOCK_BIAS: usize = CLOCK_BIAS;
/// Index of the receiver clock range-rate drift error `c·δṫ` in m/s.
pub const ERROR_CLOCK_DRIFT: usize = CLOCK_DRIFT;

/// A square matrix over the error states.
pub type ErrorMatrix = SMatrix<f64, ERROR_STATES, ERROR_STATES>;

/// One point of a reference trajectory in the axes of a navigation
/// frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NominalPoint {
    /// Position in meters in the frame axes.
    pub position_m: [f64; 3],
    /// Velocity relative to the frame in m/s, in the frame axes.
    pub velocity_mps: [f64; 3],
    /// Unit attitude from the body axes to the frame axes.
    pub attitude: AttitudeQuaternion,
    /// Specific force in m/s², in the body axes.
    pub specific_force_body_mps2: [f64; 3],
    /// Inertial angular rate in rad/s, in the body axes.
    pub angular_rate_body_rad_s: [f64; 3],
}

impl NominalPoint {
    /// Whether every value is finite and the attitude is a unit
    /// quaternion.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        let finite = self
            .position_m
            .iter()
            .chain(&self.velocity_mps)
            .chain(&self.specific_force_body_mps2)
            .chain(&self.angular_rate_body_rad_s)
            .all(|v| v.is_finite());
        finite && is_unit_quaternion(&self.attitude)
    }
}

/// The initial error covariance of the filter: the position, velocity,
/// and attitude blocks from the given covariances, each bias and the
/// empirical acceleration at its stationary variance, and the clock at
/// its configured initial variance. A state without a configured model
/// has zero variance.
///
/// # Errors
///
/// [`RejectionReason::ImplausibleCovariance`] for a covariance that is
/// not finite and positive definite.
pub fn initial_error_covariance(
    config: &InertialConfig,
    position_cov: &SymmetricCov3,
    velocity_cov: &SymmetricCov3,
    attitude_cov: &SymmetricCov3,
) -> Result<ErrorMatrix, RejectionReason> {
    for covariance in [position_cov, velocity_cov, attitude_cov] {
        check_cov3(covariance)?;
    }
    let mut p = ErrorMatrix::zeros();
    p.fixed_view_mut::<3, 3>(POS, POS)
        .copy_from(&cov3_to_matrix(position_cov));
    p.fixed_view_mut::<3, 3>(VEL, VEL)
        .copy_from(&cov3_to_matrix(velocity_cov));
    p.fixed_view_mut::<3, 3>(ATT, ATT)
        .copy_from(&cov3_to_matrix(attitude_cov));
    stochastic::set_initial_variances(&mut p, config);
    Ok(p)
}

/// The transition `Φ` and the discrete process noise `Q_d` of the error
/// states over `dt_s` seconds at `point`, in the frame of `field`. The
/// biases of the nominal state are zero.
///
/// # Errors
///
/// [`RejectionReason::NonFiniteValue`] for an implausible point or step,
/// and [`RejectionReason::ImplausibleCovariance`] when the
/// discretization fails.
pub fn error_transition(
    config: &InertialConfig,
    field: &impl GravityField,
    point: &NominalPoint,
    dt_s: f64,
) -> Result<(ErrorMatrix, ErrorMatrix), RejectionReason> {
    if !(point.is_plausible() && dt_s.is_finite() && dt_s > 0.0) {
        return Err(RejectionReason::NonFiniteValue);
    }
    let nominal = InertialNominal::at(
        Vector3::from(point.position_m),
        Vector3::from(point.velocity_mps),
        to_unit_quaternion(&point.attitude),
    );
    let rates = ImuRates {
        specific_force: Vector3::from(point.specific_force_body_mps2),
        angular_rate: Vector3::from(point.angular_rate_body_rad_s),
    };
    let (_, phi, q_d) = transition(&nominal, &rates, dt_s, field, config)
        .ok_or(RejectionReason::ImplausibleCovariance)?;
    Ok((phi, q_d))
}
