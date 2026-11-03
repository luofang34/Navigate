//! The orbital force model of a Cartesian frame (ADR-0018).
//!
//! [`OrbitalForceModel`] adds the gravity, the centrifugal and the
//! Coriolis terms of a rotating frame, the solar radiation pressure, and
//! the atmospheric drag. The result is the acceleration relative to the
//! frame, `v̇ = g(r) − ω × (ω × r) − 2·ω × v + a_srp + a_drag`, with its
//! Jacobians on the position and on the velocity. A coast propagation
//! integrates it with [`OrbitalForceModel::rk4_step_with`].
//!
//! The model borrows a spherical harmonic field for the lifetime `'a`. A
//! [`SpaceEnvironment`] holds only `'static` data, so
//! [`SpaceEnvironment::force_model`] gives a model that a filter can
//! build at each step from the environment of its state.

use nalgebra::{Matrix3, Vector3};

use crate::drag::ExponentialDrag;
use crate::gravity::{
    GravityField, SpaceEnvironment, SpaceGravityModel, SphericalHarmonicField, ThirdBodySet,
    centrifugal, centrifugal_gradient,
};
use crate::srp::CannonballSrp;

#[cfg(test)]
mod tests;

/// An acceleration and its Jacobians on the position and the velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForceEvaluation {
    /// Acceleration in m/s², in the frame axes.
    pub acceleration_mps2: Vector3<f64>,
    /// `∂a/∂r` in 1/s².
    pub position_jacobian: Matrix3<f64>,
    /// `∂a/∂v` in 1/s.
    pub velocity_jacobian: Matrix3<f64>,
}

impl ForceEvaluation {
    /// No acceleration.
    pub const ZERO: Self = Self {
        acceleration_mps2: Vector3::new(0.0, 0.0, 0.0),
        position_jacobian: Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        velocity_jacobian: Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    };

    fn plus(self, other: Self) -> Self {
        Self {
            acceleration_mps2: self.acceleration_mps2 + other.acceleration_mps2,
            position_jacobian: self.position_jacobian + other.position_jacobian,
            velocity_jacobian: self.velocity_jacobian + other.velocity_jacobian,
        }
    }
}

/// The gravity of an [`OrbitalForceModel`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(
    clippy::large_enum_variant,
    reason = "the third bodies are inline so that the model is Copy without a heap"
)]
pub enum OrbitalGravity<'a> {
    /// A gravity model of a [`SpaceEnvironment`].
    Model(SpaceGravityModel),
    /// A borrowed spherical harmonic field of the central body and the
    /// tidal terms of third bodies.
    Harmonics {
        /// The field of the central body at the frame center.
        field: SphericalHarmonicField<'a>,
        /// The third bodies at their positions at the current time.
        third_bodies: ThirdBodySet,
    },
}

impl OrbitalGravity<'_> {
    /// Whether every constant is plausible.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        match self {
            Self::Model(model) => model.is_plausible(),
            Self::Harmonics {
                field,
                third_bodies,
            } => field.is_plausible() && third_bodies.is_plausible(),
        }
    }

    /// The gravity in m/s² at `r` in meters and its gradient in 1/s².
    #[must_use]
    pub fn evaluate(&self, r: &Vector3<f64>) -> (Vector3<f64>, Matrix3<f64>) {
        match self {
            Self::Model(model) => model.evaluate(r),
            Self::Harmonics {
                field,
                third_bodies,
            } => third_bodies.add_tidal(field.evaluate(r), r),
        }
    }
}

/// The forces on a spacecraft in a Cartesian frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitalForceModel<'a> {
    /// The gravity of the central body and of the third bodies.
    pub gravity: OrbitalGravity<'a>,
    /// Rotation vector of the frame axes relative to inertial space, in
    /// rad/s in the frame axes. Zero for an inertial frame.
    pub frame_rotation_rad_per_s: [f64; 3],
    /// The solar radiation pressure, if any.
    pub srp: Option<CannonballSrp>,
    /// The atmospheric drag, if any.
    pub drag: Option<ExponentialDrag>,
}

impl SpaceEnvironment {
    /// The force model of this environment: its gravity, its frame
    /// rotation, its solar radiation pressure, and its drag.
    #[must_use]
    pub const fn force_model(&self) -> OrbitalForceModel<'static> {
        OrbitalForceModel {
            gravity: OrbitalGravity::Model(self.gravity),
            frame_rotation_rad_per_s: self.frame_rotation_rad_per_s,
            srp: self.srp,
            drag: self.drag,
        }
    }
}

impl OrbitalForceModel<'_> {
    /// Whether the rotation is finite and every constant is plausible.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        self.frame_rotation_rad_per_s.iter().all(|w| w.is_finite())
            && self.gravity.is_plausible()
            && self.srp.as_ref().is_none_or(CannonballSrp::is_plausible)
            && self.drag.as_ref().is_none_or(ExponentialDrag::is_plausible)
    }

    /// The solar radiation pressure and the drag at `r` with the velocity
    /// `v` relative to the frame. An accelerometer senses these forces.
    #[must_use]
    pub fn non_gravitational(&self, r: &Vector3<f64>, v: &Vector3<f64>) -> ForceEvaluation {
        let srp = self
            .srp
            .as_ref()
            .map_or(ForceEvaluation::ZERO, |srp| srp.evaluate(r));
        let drag = self
            .drag
            .as_ref()
            .map_or(ForceEvaluation::ZERO, |drag| drag.evaluate(r, v));
        srp.plus(drag)
    }

    /// The acceleration relative to the frame at `r` with the velocity
    /// `v` relative to the frame, and its Jacobians. It includes the
    /// centrifugal and the Coriolis terms of a rotating frame.
    #[must_use]
    pub fn evaluate(&self, r: &Vector3<f64>, v: &Vector3<f64>) -> ForceEvaluation {
        let omega = Vector3::from(self.frame_rotation_rad_per_s);
        let (g, gradient) = self.gravity.evaluate(r);
        let inertial = ForceEvaluation {
            acceleration_mps2: g + centrifugal(&omega, r) - 2.0 * omega.cross(v),
            position_jacobian: gradient + centrifugal_gradient(&omega),
            velocity_jacobian: -2.0 * omega.cross_matrix(),
        };
        inertial.plus(self.non_gravitational(r, v))
    }

    /// One fourth-order Runge–Kutta step of `dt_s` seconds from the
    /// position `r` and the velocity `v`. `extra(t)` gives an additional
    /// acceleration at the time `t` after the start of the step, for
    /// example an empirical acceleration.
    #[must_use]
    pub fn rk4_step_with(
        &self,
        r: &Vector3<f64>,
        v: &Vector3<f64>,
        dt_s: f64,
        extra: impl Fn(f64) -> Vector3<f64>,
    ) -> (Vector3<f64>, Vector3<f64>) {
        let accel = |t: f64, r: &Vector3<f64>, v: &Vector3<f64>| {
            self.evaluate(r, v).acceleration_mps2 + extra(t)
        };
        let half = 0.5 * dt_s;
        let (k1r, k1v) = (*v, accel(0.0, r, v));
        let (r2, v2) = (r + k1r * half, v + k1v * half);
        let (k2r, k2v) = (v2, accel(half, &r2, &v2));
        let (r3, v3) = (r + k2r * half, v + k2v * half);
        let (k3r, k3v) = (v3, accel(half, &r3, &v3));
        let (r4, v4) = (r + k3r * dt_s, v + k3v * dt_s);
        let (k4r, k4v) = (v4, accel(dt_s, &r4, &v4));
        let sixth = dt_s / 6.0;
        (
            r + (k1r + 2.0 * k2r + 2.0 * k3r + k4r) * sixth,
            v + (k1v + 2.0 * k2v + 2.0 * k3v + k4v) * sixth,
        )
    }

    /// One fourth-order Runge–Kutta step without an additional
    /// acceleration.
    #[must_use]
    pub fn rk4_step(
        &self,
        r: &Vector3<f64>,
        v: &Vector3<f64>,
        dt_s: f64,
    ) -> (Vector3<f64>, Vector3<f64>) {
        self.rk4_step_with(r, v, dt_s, |_| Vector3::zeros())
    }
}

impl GravityField for OrbitalForceModel<'_> {
    /// The gravity plus the centrifugal term, without the forces that an
    /// accelerometer senses.
    fn gravity(&self, p: &Vector3<f64>) -> (Vector3<f64>, Matrix3<f64>) {
        let omega = Vector3::from(self.frame_rotation_rad_per_s);
        let (g, gradient) = self.gravity.evaluate(p);
        (
            g + centrifugal(&omega, p),
            gradient + centrifugal_gradient(&omega),
        )
    }

    fn frame_rotation(&self) -> Vector3<f64> {
        Vector3::from(self.frame_rotation_rad_per_s)
    }
}
