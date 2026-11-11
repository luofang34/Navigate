//! Posterior Cramér–Rao lower bound along a reference trajectory
//! (ADR-0014, ADR-0016, ADR-0017).
//!
//! The tracker runs the covariance recursion of the inertial filter of
//! `navigate-fusion` at a known reference trajectory: `P' = Φ·P·Φᵀ + Q_d`
//! with the transition of the strapdown error state, and the Joseph-form
//! update of each measurement. At the true trajectory, the information
//! form of this recursion is the posterior Cramér–Rao bound of a
//! linear-Gaussian model. The bound shows how much a map leg can help
//! before a flight.
//!
//! The error state is in the rotating body-fixed frame of the body, as
//! in the filter. The reference points and the map gradients are in the
//! local NED axes, and the tracker rotates them into the body-fixed axes.
//! The reported position bounds are in the local NED axes at the most
//! recent reference point.

use nalgebra::{Matrix3, Quaternion, Rotation3, SMatrix, UnitQuaternion, Vector3};
use navigate_astrodynamics::{PlanetaryBody, SpaceEnvironment};
use navigate_contract::{AttitudeQuaternion, GeodeticPosition, NedVelocity, SymmetricCov3};
use navigate_fusion::RejectionReason;
use navigate_fusion::inertial::{
    ErrorMatrix, InertialConfig, NominalPoint, StateLayout, error_transition,
    initial_error_covariance, ned_to_body_fixed_dcm,
};
use navigate_fusion::kalman::joseph_covariance_update;
use navigate_geodesy::{BodyEllipsoid, geodetic_to_body_fixed};

use crate::float_math::sqrt;

/// The layout of the filter error states that the bound uses.
const LAYOUT: StateLayout = StateLayout::FULL;
/// Number of error states.
const ERROR_STATES: usize = LAYOUT.dimension();
/// First index of the position error.
const ERROR_POSITION: usize = LAYOUT.position();
/// First index of the velocity error.
const ERROR_VELOCITY: usize = LAYOUT.velocity();
/// First index of the attitude error.
const ERROR_ATTITUDE: usize = LAYOUT.attitude();

#[cfg(test)]
mod tests;

/// One point of the reference trajectory on the surface of the body of
/// the tracker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReferencePoint {
    /// Geodetic position on the ellipsoid of the body.
    pub position: GeodeticPosition,
    /// Velocity relative to the body in the local NED axes at
    /// `position`.
    pub velocity: NedVelocity,
    /// Unit attitude from the body axes to the local NED axes at
    /// `position`.
    pub attitude: AttitudeQuaternion,
    /// Specific force in m/s², in the body axes.
    pub specific_force_body_mps2: [f64; 3],
    /// Inertial angular rate in rad/s, in the body axes.
    pub angular_rate_body_rad_s: [f64; 3],
}

impl ReferencePoint {
    /// The point in the body-fixed axes, the form of the inertial error
    /// transition. `ned_to_bf` is the rotation from the local NED axes at
    /// the point to the body-fixed axes. The attitude keeps the norm of
    /// [`Self::attitude`], so the transition refuses a quaternion that
    /// is not a unit quaternion.
    fn nominal(&self, ellipsoid: BodyEllipsoid, ned_to_bf: &Matrix3<f64>) -> NominalPoint {
        let r = geodetic_to_body_fixed(&self.position, ellipsoid);
        let v = &self.velocity;
        let velocity = ned_to_bf * Vector3::new(v.north_mps, v.east_mps, v.down_mps);
        let rotation = Rotation3::from_matrix_unchecked(*ned_to_bf);
        let ned_to_frame = UnitQuaternion::from_rotation_matrix(&rotation).into_inner();
        let a = &self.attitude;
        let q = ned_to_frame * Quaternion::new(a.w, a.x, a.y, a.z);
        NominalPoint {
            position_m: [r.x_m, r.y_m, r.z_m],
            velocity_mps: velocity.into(),
            attitude: AttitudeQuaternion::new(q.w, q.i, q.j, q.k),
            specific_force_body_mps2: self.specific_force_body_mps2,
            angular_rate_body_rad_s: self.angular_rate_body_rad_s,
        }
    }
}

/// The posterior Cramér–Rao bound of the inertial error state on a body
/// surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrlbTracker {
    config: InertialConfig,
    env: SpaceEnvironment,
    ellipsoid: BodyEllipsoid,
    /// The rotation from the local NED axes at the most recent reference
    /// point to the body-fixed axes.
    ned_to_bf: Matrix3<f64>,
    p: ErrorMatrix,
}

impl CrlbTracker {
    /// Starts the bound at `origin` on `body`, in the body-fixed frame of
    /// [`SpaceEnvironment::from_body`]. `position_cov` and `velocity_cov`
    /// are in the local NED axes at `origin`, and the tracker rotates
    /// them into the body-fixed axes. `attitude_cov` is in the body axes.
    /// The biases start at their stationary variances.
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
        if !(body.is_plausible() && origin.is_plausible()) {
            return Err(RejectionReason::NonFiniteValue);
        }
        let ned_to_bf = ned_to_body_fixed_dcm(origin.latitude_rad, origin.longitude_rad);
        let mut p = initial_error_covariance(&config, &position_cov, &velocity_cov, &attitude_cov)?;
        for first in [ERROR_POSITION, ERROR_VELOCITY] {
            let block = p.fixed_view::<3, 3>(first, first).into_owned();
            let rotated = ned_to_bf * block * ned_to_bf.transpose();
            p.fixed_view_mut::<3, 3>(first, first)
                .copy_from(&((rotated + rotated.transpose()) * 0.5));
        }
        Ok(Self {
            config,
            env: SpaceEnvironment::from_body(body),
            ellipsoid: body.ellipsoid,
            ned_to_bf,
            p,
        })
    }

    /// Propagates the bound by `dt_s` seconds with the transition at
    /// `reference`. The local axes of the reported bounds then move to
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
        let position = &reference.position;
        if !(position.is_plausible() && reference.velocity.is_finite()) {
            return Err(RejectionReason::NonFiniteValue);
        }
        let ned_to_bf = ned_to_body_fixed_dcm(position.latitude_rad, position.longitude_rad);
        let nominal = reference.nominal(self.ellipsoid, &ned_to_bf);
        let (phi, q_d) = error_transition(&self.config, &self.env, &nominal, dt_s)?;
        let p = phi * self.p * phi.transpose() + q_d;
        self.p = (p + p.transpose()) * 0.5;
        self.ned_to_bf = ned_to_bf;
        Ok(())
    }

    /// Applies a scalar map measurement with the gradient
    /// `[g_n, g_e, g_d]` in the local NED axes at the most recent
    /// reference point and the variance `variance_t2`. The row is
    /// `H = [R·g, 0, …, 0, 1]`, with `R` the rotation into the body-fixed
    /// axes: the map bias enters as in the inertial filter.
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
        let gradient = self.ned_to_bf * Vector3::from(gradient_ned_t_per_m);
        let mut h = SMatrix::<f64, 1, ERROR_STATES>::zeros();
        h.fixed_view_mut::<1, 3>(0, ERROR_POSITION)
            .copy_from(&gradient.transpose());
        if let Some(mag_bias) = LAYOUT.mag_bias() {
            h[(0, mag_bias)] = 1.0;
        }
        self.p = joseph_covariance_update(&self.p, &h, &SMatrix::<f64, 1, 1>::new(variance_t2))
            .ok_or(RejectionReason::ImplausibleCovariance)?;
        Ok(())
    }

    /// The 1-sigma bound of each position axis in meters, in the local
    /// NED axes at the most recent reference point.
    #[must_use]
    pub fn position_1sigma_m(&self) -> [f64; 3] {
        let local = self.local_position_covariance();
        [0, 1, 2].map(|axis| sqrt(local[(axis, axis)].max(0.0)))
    }

    /// The horizontal distance root mean square `√(σ_n² + σ_e²)` in
    /// meters, in the local NED axes at the most recent reference point.
    #[must_use]
    pub fn horizontal_drms_m(&self) -> f64 {
        let local = self.local_position_covariance();
        sqrt((local[(0, 0)] + local[(1, 1)]).max(0.0))
    }

    /// The 1-sigma bound of each body-axis attitude error in radians.
    #[must_use]
    pub fn attitude_1sigma_rad(&self) -> [f64; 3] {
        [0, 1, 2].map(|axis| {
            let index = ERROR_ATTITUDE + axis;
            sqrt(self.p[(index, index)].max(0.0))
        })
    }

    /// The position covariance in the local NED axes at the most recent
    /// reference point.
    fn local_position_covariance(&self) -> Matrix3<f64> {
        let block = self
            .p
            .fixed_view::<3, 3>(ERROR_POSITION, ERROR_POSITION)
            .into_owned();
        self.ned_to_bf.transpose() * block * self.ned_to_bf
    }
}
