//! Posterior Cramér–Rao lower bound along a reference trajectory
//! (ADR-0014, ADR-0016).
//!
//! The tracker runs the covariance recursion of the inertial filter of
//! `navigate-fusion` at a known reference trajectory: `P' = Φ·P·Φᵀ + Q_d`
//! with the transition of the strapdown error state, and the Joseph-form
//! update of each measurement. At the true trajectory, the information
//! form of this recursion is the posterior Cramér–Rao bound of a
//! linear-Gaussian model. The bound shows how much a map leg can help
//! before a flight.

use nalgebra::{SMatrix, Vector3};
use navigate_astrodynamics::{PlanetaryBody, SurfaceFrame};
use navigate_contract::{AttitudeQuaternion, GeodeticPosition, SymmetricCov3};
use navigate_fusion::RejectionReason;
use navigate_fusion::inertial::{
    ERROR_ATTITUDE, ERROR_MAG_BIAS, ERROR_POSITION, ERROR_STATES, ErrorMatrix, InertialConfig,
    NominalPoint, error_transition, initial_error_covariance,
};
use navigate_fusion::kalman::joseph_covariance_update;
use navigate_geodesy::LocalTangentPlane;

use crate::float_math::sqrt;

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

impl ReferencePoint {
    /// The point in the form of the inertial error transition.
    const fn nominal(&self) -> NominalPoint {
        NominalPoint {
            position_m: self.position_plane_m,
            velocity_mps: self.velocity_plane_mps,
            attitude: self.attitude,
            specific_force_body_mps2: self.specific_force_body_mps2,
            angular_rate_body_rad_s: self.angular_rate_body_rad_s,
        }
    }
}

/// The posterior Cramér–Rao bound of the seventeen-state inertial error
/// on a body surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrlbTracker {
    config: InertialConfig,
    frame: SurfaceFrame,
    p: ErrorMatrix,
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
        let plane = LocalTangentPlane::new(origin, body.ellipsoid)
            .map_err(|_| RejectionReason::NonFiniteValue)?;
        let p = initial_error_covariance(&config, &position_cov, &velocity_cov, &attitude_cov)?;
        Ok(Self {
            config,
            frame: SurfaceFrame::new(plane, body),
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
        let (phi, q_d) = error_transition(&self.config, &self.frame, &reference.nominal(), dt_s)?;
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
        if !(variance_t2.is_finite() && variance_t2 > 0.0) {
            return Err(RejectionReason::ImplausibleCovariance);
        }
        let mut h = SMatrix::<f64, 1, ERROR_STATES>::zeros();
        h.fixed_view_mut::<1, 3>(0, ERROR_POSITION)
            .copy_from(&Vector3::from(gradient_ned_t_per_m).transpose());
        h[(0, ERROR_MAG_BIAS)] = 1.0;
        self.p = joseph_covariance_update(&self.p, &h, &SMatrix::<f64, 1, 1>::new(variance_t2))
            .ok_or(RejectionReason::ImplausibleCovariance)?;
        Ok(())
    }

    /// The 1-sigma bound of each position axis in meters, in the plane
    /// axes.
    #[must_use]
    pub fn position_1sigma_m(&self) -> [f64; 3] {
        self.axis_1sigma(ERROR_POSITION)
    }

    /// The horizontal distance root mean square `√(σ_n² + σ_e²)` in
    /// meters.
    #[must_use]
    pub fn horizontal_drms_m(&self) -> f64 {
        let north = self.p[(ERROR_POSITION, ERROR_POSITION)];
        let east = self.p[(ERROR_POSITION + 1, ERROR_POSITION + 1)];
        sqrt((north + east).max(0.0))
    }

    /// The 1-sigma bound of each body-axis attitude error in radians.
    #[must_use]
    pub fn attitude_1sigma_rad(&self) -> [f64; 3] {
        self.axis_1sigma(ERROR_ATTITUDE)
    }

    /// The square roots of the three diagonal variances from `first`.
    fn axis_1sigma(&self, first: usize) -> [f64; 3] {
        [0, 1, 2].map(|axis| sqrt(self.p[(first + axis, first + axis)].max(0.0)))
    }
}
