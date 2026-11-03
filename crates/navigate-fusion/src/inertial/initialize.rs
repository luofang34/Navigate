//! Explicit initialization of the inertial filter (ADR-0014).

use std::sync::Arc;

use nalgebra::{Matrix3, Vector3};
use navigate_contract::{
    AttitudeQuaternion, CartesianPosition, CartesianVelocity, GeodeticPosition, MonotonicNanos,
    NedVelocity, SourceComposition, SymmetricCov3,
};
use navigate_geodesy::LocalTangentPlane;

use super::InertialNavigationFilter;
use super::config::InertialConfig;
use super::estimate::InertialCore;
use super::nominal::{
    ACCEL_BIAS, ALT_BIAS, ATT, GYRO_BIAS, InertialNominal, MAG_BIAS, Mat17, POS, VEL,
    to_unit_quaternion,
};
use super::regime::{Regime, Setup};
use crate::admission::{check_composition, check_cov3, is_unit_quaternion};
use crate::environment::SurfaceFrame;
use crate::filter::cov3_to_matrix;
use crate::observation::MeasurementKind;
use crate::rejection::RejectionReason;

/// The initial state of one regime before the checks.
struct InitialState {
    regime: Regime,
    v: Vector3<f64>,
    attitude: AttitudeQuaternion,
    pos_cov: SymmetricCov3,
    vel_cov: SymmetricCov3,
    att_cov: SymmetricCov3,
}

impl InertialNavigationFilter {
    /// Sets the state on the body surface at `at`. The tangent plane is
    /// anchored at `position`. `attitude` rotates the body axes into the
    /// local NED axes. `pos_cov`, `vel_cov`, and `att_cov` are in the
    /// local NED axes. The biases start at zero with their stationary
    /// variances. An initialization forgets the history.
    ///
    /// # Errors
    ///
    /// [`RejectionReason::UnsupportedMeasurement`] in a space filter,
    /// [`RejectionReason::NonFiniteValue`] for an implausible value or
    /// body, [`RejectionReason::ImplausibleCovariance`] for a covariance
    /// that is not positive definite, and the composition refusals.
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
            let Setup::BodySurface(body) = &self.setup else {
                return Err(RejectionReason::UnsupportedMeasurement {
                    kind: MeasurementKind::PositionFix,
                });
            };
            if !(body.is_plausible() && position.is_plausible() && velocity.is_finite()) {
                return Err(RejectionReason::NonFiniteValue);
            }
            let plane = LocalTangentPlane::with_ellipsoid(position, body.ellipsoid)
                .map_err(|_| RejectionReason::NonFiniteValue)?;
            let initial = InitialState {
                regime: Regime::BodySurface(Arc::new(SurfaceFrame::new(plane, *body))),
                v: Vector3::new(velocity.north_mps, velocity.east_mps, velocity.down_mps),
                attitude,
                pos_cov,
                vel_cov,
                att_cov,
            };
            self.start(initial, Vector3::zeros(), at, composition)
        })();
        self.record_failure(result)
    }

    /// Sets the state in the space frame at `at`. `attitude` rotates the
    /// body axes into the frame axes. Each covariance is in the frame
    /// axes.
    ///
    /// # Errors
    ///
    /// [`RejectionReason::UnsupportedMeasurement`] in a surface filter,
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
            let Setup::CartesianSpace(env) = &self.setup else {
                return Err(RejectionReason::UnsupportedMeasurement {
                    kind: MeasurementKind::CartesianPositionFix,
                });
            };
            if !(env.is_plausible() && position.is_finite() && velocity.is_finite()) {
                return Err(RejectionReason::NonFiniteValue);
            }
            let initial = InitialState {
                regime: Regime::CartesianSpace(env.clone()),
                v: Vector3::new(velocity.vx_mps, velocity.vy_mps, velocity.vz_mps),
                attitude,
                pos_cov,
                vel_cov,
                att_cov,
            };
            let p = Vector3::new(position.x_m, position.y_m, position.z_m);
            self.start(initial, p, at, composition)
        })();
        self.record_failure(result)
    }

    fn start(
        &mut self,
        initial: InitialState,
        p: Vector3<f64>,
        at: MonotonicNanos,
        composition: SourceComposition,
    ) -> Result<(), RejectionReason> {
        check_composition(composition)?;
        if !is_unit_quaternion(&initial.attitude) {
            return Err(RejectionReason::NonFiniteValue);
        }
        for covariance in [&initial.pos_cov, &initial.vel_cov, &initial.att_cov] {
            check_cov3(covariance)?;
        }
        let nominal = InertialNominal {
            p,
            v: initial.v,
            q: to_unit_quaternion(&initial.attitude),
            b_a: Vector3::zeros(),
            b_g: Vector3::zeros(),
            b_alt: 0.0,
            b_mag: 0.0,
        };
        let c = nominal.c_b_to_f();
        let p = initial_covariance(&self.config, &initial, &c);
        self.core = Some(InertialCore {
            regime: initial.regime,
            nominal,
            p,
            state_at: at,
            last_rates: None,
            mag_key: None,
        });
        self.history.initialize(at);
        self.ledger.initialize(composition, at);
        Ok(())
    }
}

/// The initial covariance. The attitude prior rotates from the
/// reference axes into the body axes, `P_θ = Cᵀ·P_ref·C`, and then each
/// diagonal variance rises to the configured floor.
fn initial_covariance(config: &InertialConfig, initial: &InitialState, c: &Matrix3<f64>) -> Mat17 {
    let mut p = Mat17::zeros();
    p.fixed_view_mut::<3, 3>(POS, POS)
        .copy_from(&cov3_to_matrix(&initial.pos_cov));
    p.fixed_view_mut::<3, 3>(VEL, VEL)
        .copy_from(&cov3_to_matrix(&initial.vel_cov));
    let mut attitude = c.transpose() * cov3_to_matrix(&initial.att_cov) * c;
    attitude = (attitude + attitude.transpose()) * 0.5;
    for (axis, floor) in config.initial_attitude_sigma_rad.iter().enumerate() {
        let floor_variance = floor * floor;
        if attitude[(axis, axis)] < floor_variance {
            attitude[(axis, axis)] = floor_variance;
        }
    }
    p.fixed_view_mut::<3, 3>(ATT, ATT).copy_from(&attitude);
    for axis in 0..3 {
        p[(ACCEL_BIAS + axis, ACCEL_BIAS + axis)] = config.accel_bias.stationary_variance();
        p[(GYRO_BIAS + axis, GYRO_BIAS + axis)] = config.gyro_bias.stationary_variance();
    }
    p[(ALT_BIAS, ALT_BIAS)] = config.baro_bias.stationary_variance();
    p[(MAG_BIAS, MAG_BIAS)] = config.mag_bias.stationary_variance();
    p
}
