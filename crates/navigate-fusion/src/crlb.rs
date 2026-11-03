//! Posterior Cramér–Rao lower bound along a reference trajectory
//! (ADR-0014).
//!
//! The tracker runs the covariance recursion of the inertial filter at a
//! known reference trajectory: `P' = Φ·P·Φᵀ + Q_d` with the transition
//! of the strapdown error state, and the Joseph-form update of each
//! measurement. At the true trajectory, the information form of this
//! recursion is the posterior Cramér–Rao bound of a linear-Gaussian
//! model. The bound shows how much a map leg can help before a flight.

use std::sync::Arc;

use nalgebra::{SMatrix, SVector, Vector3};
use navigate_contract::{AttitudeQuaternion, GeodeticPosition, SymmetricCov3};
use navigate_geodesy::LocalTangentPlane;

use crate::admission::{check_cov3, check_scalar_variance, is_unit_quaternion};
use crate::environment::{PlanetaryBody, SurfaceFrame};
use crate::filter::{KalmanState, cov3_to_matrix};
use crate::inertial::InertialConfig;
use crate::inertial::estimate::Linearized;
use crate::inertial::mechanization::{ImuRates, transition};
use crate::inertial::nominal::{
    ACCEL_BIAS, ALT_BIAS, ATT, GYRO_BIAS, InertialNominal, MAG_BIAS, Mat17, N, POS, VEL, Vec17,
    to_unit_quaternion,
};
use crate::inertial::regime::Regime;
use crate::rejection::RejectionReason;

#[cfg(test)]
mod tests;

/// One point of the reference trajectory in the tangent plane of the
/// tracker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferencePoint {
    /// Position in the plane axes in meters.
    pub position_plane_m: [f64; 3],
    /// Velocity in the plane axes in m/s.
    pub velocity_plane_mps: [f64; 3],
    /// Unit attitude from the body axes to the plane axes.
    pub attitude: AttitudeQuaternion,
    /// Specific force in m/s², in the body axes.
    pub specific_force_body_mps2: [f64; 3],
    /// Inertial angular rate in rad/s, in the body axes.
    pub angular_rate_body_rad_s: [f64; 3],
}

/// The posterior Cramér–Rao bound of the seventeen-state inertial error
/// on a body surface.
#[derive(Debug, Clone, PartialEq)]
pub struct CrlbTracker {
    config: InertialConfig,
    regime: Regime,
    p: Mat17,
}

impl CrlbTracker {
    /// Starts the bound at `origin` on `body`. The covariances are in the
    /// local NED axes at `origin`, which are the plane axes. The biases
    /// start at their stationary variances.
    ///
    /// # Errors
    ///
    /// [`RejectionReason::NonFiniteValue`] for an implausible body or
    /// origin, and [`RejectionReason::ImplausibleCovariance`].
    pub fn new_surface(
        config: InertialConfig,
        body: PlanetaryBody,
        origin: GeodeticPosition,
        position_cov: SymmetricCov3,
        velocity_cov: SymmetricCov3,
        attitude_cov: SymmetricCov3,
    ) -> Result<Self, RejectionReason> {
        if !body.is_plausible() {
            return Err(RejectionReason::NonFiniteValue);
        }
        let plane = LocalTangentPlane::with_ellipsoid(origin, body.ellipsoid)
            .map_err(|_| RejectionReason::NonFiniteValue)?;
        for covariance in [&position_cov, &velocity_cov, &attitude_cov] {
            check_cov3(covariance)?;
        }
        let mut p = Mat17::zeros();
        p.fixed_view_mut::<3, 3>(POS, POS)
            .copy_from(&cov3_to_matrix(&position_cov));
        p.fixed_view_mut::<3, 3>(VEL, VEL)
            .copy_from(&cov3_to_matrix(&velocity_cov));
        p.fixed_view_mut::<3, 3>(ATT, ATT)
            .copy_from(&cov3_to_matrix(&attitude_cov));
        for axis in 0..3 {
            p[(ACCEL_BIAS + axis, ACCEL_BIAS + axis)] = config.accel_bias.stationary_variance();
            p[(GYRO_BIAS + axis, GYRO_BIAS + axis)] = config.gyro_bias.stationary_variance();
        }
        p[(ALT_BIAS, ALT_BIAS)] = config.baro_bias.stationary_variance();
        p[(MAG_BIAS, MAG_BIAS)] = config.mag_bias.stationary_variance();
        Ok(Self {
            config,
            regime: Regime::BodySurface(Arc::new(SurfaceFrame::new(plane, body))),
            p,
        })
    }

    /// Propagates the bound by `dt_s` seconds with the transition at
    /// `reference`.
    ///
    /// # Errors
    ///
    /// [`RejectionReason::NonFiniteValue`] for an implausible reference
    /// or step, and [`RejectionReason::ImplausibleCovariance`] when the
    /// discretization fails. On an error nothing changes.
    pub fn propagate(
        &mut self,
        reference: &ReferencePoint,
        dt_s: f64,
    ) -> Result<(), RejectionReason> {
        let values = reference
            .position_plane_m
            .iter()
            .chain(&reference.velocity_plane_mps)
            .chain(&reference.specific_force_body_mps2)
            .chain(&reference.angular_rate_body_rad_s);
        let finite = values.into_iter().all(|v| v.is_finite());
        if !(finite && is_unit_quaternion(&reference.attitude) && dt_s.is_finite() && dt_s > 0.0) {
            return Err(RejectionReason::NonFiniteValue);
        }
        let nominal = InertialNominal {
            p: Vector3::from(reference.position_plane_m),
            v: Vector3::from(reference.velocity_plane_mps),
            q: to_unit_quaternion(&reference.attitude),
            b_a: Vector3::zeros(),
            b_g: Vector3::zeros(),
            b_alt: 0.0,
            b_mag: 0.0,
        };
        let rates = ImuRates {
            specific_force: Vector3::from(reference.specific_force_body_mps2),
            angular_rate: Vector3::from(reference.angular_rate_body_rad_s),
        };
        let (_, phi, q_d) = transition(&nominal, &rates, dt_s, &self.regime, &self.config)
            .ok_or(RejectionReason::ImplausibleCovariance)?;
        let p = phi * self.p * phi.transpose() + q_d;
        self.p = (p + p.transpose()) * 0.5;
        Ok(())
    }

    /// Applies a scalar map measurement with the gradient
    /// `[g_n, g_e, g_d]` in the plane axes and the variance
    /// `variance_t2`. The row is `H = [g, 0, …, 0, 1]`: the map bias
    /// enters as in the inertial filter.
    ///
    /// # Errors
    ///
    /// [`RejectionReason::NonFiniteValue`] for a non-finite gradient and
    /// [`RejectionReason::ImplausibleCovariance`] for a variance that is
    /// not finite and positive. On an error nothing changes.
    pub fn update_map_gradient(
        &mut self,
        gradient_ned_t_per_m: [f64; 3],
        variance_t2: f64,
    ) -> Result<(), RejectionReason> {
        if !gradient_ned_t_per_m.iter().all(|g| g.is_finite()) {
            return Err(RejectionReason::NonFiniteValue);
        }
        check_scalar_variance(variance_t2)?;
        let mut h = SMatrix::<f64, 1, N>::zeros();
        h.fixed_view_mut::<1, 3>(0, POS)
            .copy_from(&Vector3::from(gradient_ned_t_per_m).transpose());
        h[(0, MAG_BIAS)] = 1.0;
        self.update(h, SMatrix::<f64, 1, 1>::new(variance_t2))
    }

    /// The Joseph-form update with the row block `h` and the noise `r`.
    pub(crate) fn update<const M: usize>(
        &mut self,
        h: SMatrix<f64, M, N>,
        r: SMatrix<f64, M, M>,
    ) -> Result<(), RejectionReason> {
        let model = Linearized {
            residual: SVector::<f64, M>::zeros(),
            h,
            r,
        };
        let state = KalmanState::<N> {
            x: Vec17::zeros(),
            p: self.p,
        };
        let prepared = state
            .prepare_update(&model)
            .ok_or(RejectionReason::ImplausibleCovariance)?;
        let p = state.apply_update(&prepared).p;
        self.p = (p + p.transpose()) * 0.5;
        Ok(())
    }

    /// The 1-sigma bound of each position axis in meters, in the plane
    /// axes.
    #[must_use]
    pub fn position_1sigma_m(&self) -> [f64; 3] {
        [0, 1, 2].map(|axis| self.p[(POS + axis, POS + axis)].max(0.0).sqrt())
    }

    /// The horizontal distance root mean square `√(σ_n² + σ_e²)` in
    /// meters.
    #[must_use]
    pub fn horizontal_drms_m(&self) -> f64 {
        (self.p[(POS, POS)] + self.p[(POS + 1, POS + 1)])
            .max(0.0)
            .sqrt()
    }

    /// The 1-sigma bound of each body-axis attitude error in radians.
    #[must_use]
    pub fn attitude_1sigma_rad(&self) -> [f64; 3] {
        [0, 1, 2].map(|axis| self.p[(ATT + axis, ATT + axis)].max(0.0).sqrt())
    }
}
