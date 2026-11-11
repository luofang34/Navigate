//! Explicit initialization of the inertial filter (ADR-0014, ADR-0017).
//!
//! The state is always in the body-centered Cartesian frame of the
//! regime. A surface initialization converts a geodetic position, a
//! local NED velocity, a body-to-NED attitude, and local NED covariances
//! into that frame with the local axes at the position.

use nalgebra::{Const, DimMin, Matrix3, Vector3};
use navigate_contract::{
    AttitudeQuaternion, CartesianPosition, CartesianVelocity, GeodeticPosition, MonotonicNanos,
    NedVelocity, SourceComposition, SymmetricCov3,
};

use super::InertialNavigationFilter;
use super::config::InertialConfig;
use super::estimate::InertialCore;
use super::local_level::{LocalLevel, body_fixed, quaternion_of};
use super::nominal::{ATT, ErrorMat, InertialNominal, POS, VEL, to_unit_quaternion};
use super::regime::Regime;
use super::stochastic;
use crate::admission::{check_composition, check_cov3, is_unit_quaternion};
use crate::kalman::cov3_to_matrix;
use crate::observation::MeasurementKind;
use crate::rejection::RejectionReason;

/// The initial state in the frame axes before the checks of the
/// nominal state. Each covariance is in the frame axes.
struct InitialState {
    p: Vector3<f64>,
    v: Vector3<f64>,
    q: nalgebra::UnitQuaternion<f64>,
    pos_cov: Matrix3<f64>,
    vel_cov: Matrix3<f64>,
    att_cov: Matrix3<f64>,
}

impl<const N: usize> InertialNavigationFilter<N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    /// Sets the state on the reference ellipsoid of the active regime at
    /// `at`. After a change of frame, the active regime is the new one.
    /// `attitude` rotates the body axes into the local NED axes at
    /// `position`. `pos_cov`, `vel_cov`, and `att_cov` are in the same
    /// local NED axes. The filter converts each value into the
    /// body-fixed frame: `p = r(position)`, `v = R·v_ned`,
    /// `q = q_ned→frame ⊗ attitude`, and `P = R·P_ned·Rᵀ`, with `R` the
    /// rotation from the local NED axes to the frame axes. The biases
    /// start from the prior of [`Self::set_imu_bias_prior`], or at zero
    /// with their stationary variances without one. An initialization
    /// forgets the history.
    ///
    /// # Errors
    ///
    /// [`RejectionReason::UnsupportedMeasurement`] in a regime without a
    /// reference ellipsoid, [`RejectionReason::NonFiniteValue`] for an
    /// implausible value or environment,
    /// [`RejectionReason::ImplausibleCovariance`] for a covariance that
    /// is not positive definite, and the composition refusals.
    #[expect(
        clippy::too_many_arguments,
        reason = "each argument is one required part of the initial state"
    )]
    pub fn initialize_surface(
        &mut self,
        position: GeodeticPosition,
        velocity: NedVelocity,
        attitude: AttitudeQuaternion,
        pos_cov: SymmetricCov3,
        vel_cov: SymmetricCov3,
        att_cov: SymmetricCov3,
        at: MonotonicNanos,
        composition: SourceComposition,
    ) -> Result<(), RejectionReason> {
        let result = (|| {
            let regime = self.active_regime();
            let Some(ellipsoid) = regime.surface else {
                return Err(RejectionReason::UnsupportedMeasurement {
                    kind: MeasurementKind::PositionFix,
                });
            };
            if !(regime.env.is_plausible() && position.is_plausible() && velocity.is_finite()) {
                return Err(RejectionReason::NonFiniteValue);
            }
            check_initial(composition, &attitude, [&pos_cov, &vel_cov, &att_cov])?;
            let local = LocalLevel::at(position);
            let v_ned = Vector3::new(velocity.north_mps, velocity.east_mps, velocity.down_mps);
            let initial = InitialState {
                p: body_fixed(&position, ellipsoid),
                v: local.ned_to_bf * v_ned,
                q: quaternion_of(&local.ned_to_bf) * to_unit_quaternion(&attitude),
                pos_cov: local.ned_to_bf_cov(&cov3_to_matrix(&pos_cov)),
                vel_cov: local.ned_to_bf_cov(&cov3_to_matrix(&vel_cov)),
                att_cov: local.ned_to_bf_cov(&cov3_to_matrix(&att_cov)),
            };
            self.start(regime, &initial, at, composition)
        })();
        self.record_failure(result)
    }

    /// Sets the state in the active frame at `at`. `attitude` rotates
    /// the body axes into the frame axes. Each covariance is in the frame
    /// axes. Each regime accepts this initialization, so a surface
    /// filter can start from a body-fixed Cartesian position. After a
    /// change of frame, the active frame is the new one. The biases
    /// start as in [`Self::initialize_surface`].
    ///
    /// # Errors
    ///
    /// [`RejectionReason::NonFiniteValue`] for an implausible value or
    /// environment, [`RejectionReason::ImplausibleCovariance`] for a
    /// covariance that is not positive definite, and the composition
    /// refusals.
    #[expect(
        clippy::too_many_arguments,
        reason = "each argument is one required part of the initial state"
    )]
    pub fn initialize_space(
        &mut self,
        position: CartesianPosition,
        velocity: CartesianVelocity,
        attitude: AttitudeQuaternion,
        pos_cov: SymmetricCov3,
        vel_cov: SymmetricCov3,
        att_cov: SymmetricCov3,
        at: MonotonicNanos,
        composition: SourceComposition,
    ) -> Result<(), RejectionReason> {
        let result = (|| {
            let regime = self.active_regime();
            if !(regime.env.is_plausible() && position.is_finite() && velocity.is_finite()) {
                return Err(RejectionReason::NonFiniteValue);
            }
            check_initial(composition, &attitude, [&pos_cov, &vel_cov, &att_cov])?;
            let initial = InitialState {
                p: Vector3::new(position.x_m, position.y_m, position.z_m),
                v: Vector3::new(velocity.vx_mps, velocity.vy_mps, velocity.vz_mps),
                q: to_unit_quaternion(&attitude),
                pos_cov: cov3_to_matrix(&pos_cov),
                vel_cov: cov3_to_matrix(&vel_cov),
                att_cov: cov3_to_matrix(&att_cov),
            };
            self.start(regime, &initial, at, composition)
        })();
        self.record_failure(result)
    }

    fn start(
        &mut self,
        regime: Regime,
        initial: &InitialState,
        at: MonotonicNanos,
        composition: SourceComposition,
    ) -> Result<(), RejectionReason> {
        let mut nominal = InertialNominal::at(initial.p, initial.v, initial.q);
        let mut p = initial_covariance(&self.config, initial, &nominal.c_b_to_f());
        if let Some(prior) = &self.bias_prior {
            prior.apply(&mut nominal, &mut p);
        }
        self.core = Some(InertialCore {
            layout: self.config.layout,
            regime,
            nominal,
            p,
            state_at: at,
            last_rates: None,
            mag_key: None,
            correlated: super::correlated::CorrelatedSources::default(),
            pending: None,
            mode: self.setup_mode,
        });
        self.history.initialize(at);
        self.ledger.initialize(composition, at);
        Ok(())
    }
}

/// Refuses an undeclared composition, an attitude that is not a unit
/// quaternion, and a covariance that is not finite and positive
/// definite.
fn check_initial(
    composition: SourceComposition,
    attitude: &AttitudeQuaternion,
    covariances: [&SymmetricCov3; 3],
) -> Result<(), RejectionReason> {
    check_composition(composition)?;
    if !is_unit_quaternion(attitude) {
        return Err(RejectionReason::NonFiniteValue);
    }
    for covariance in covariances {
        check_cov3(covariance)?;
    }
    Ok(())
}

/// The initial covariance. [`stochastic::set_initial_variances`] gives
/// the bias, empirical acceleration, and clock variances. The attitude
/// prior rotates from the frame
/// axes into the body axes, `P_θ = Cᵀ·P_frame·C`, and then each diagonal
/// variance rises to the configured floor. For a surface
/// initialization, `P_frame = R·P_ned·Rᵀ`, so `P_θ = C_bnᵀ·P_ned·C_bn`
/// with `C_bn` the rotation from the body axes to the local NED axes.
fn initial_covariance<const N: usize>(
    config: &InertialConfig,
    initial: &InitialState,
    c: &Matrix3<f64>,
) -> ErrorMat<N> {
    let mut p = ErrorMat::<N>::zeros();
    p.fixed_view_mut::<3, 3>(POS, POS)
        .copy_from(&initial.pos_cov);
    p.fixed_view_mut::<3, 3>(VEL, VEL)
        .copy_from(&initial.vel_cov);
    let mut attitude = c.transpose() * initial.att_cov * c;
    attitude = (attitude + attitude.transpose()) * 0.5;
    for (axis, floor) in config.initial_attitude_sigma_rad.iter().enumerate() {
        let floor_variance = floor * floor;
        if attitude[(axis, axis)] < floor_variance {
            attitude[(axis, axis)] = floor_variance;
        }
    }
    p.fixed_view_mut::<3, 3>(ATT, ATT).copy_from(&attitude);
    stochastic::set_initial_variances(&mut p, config);
    p
}
