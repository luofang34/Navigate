//! Gravity in a Cartesian frame centered on one body (ADR-0014).

use nalgebra::{Matrix3, Vector3};
use navigate_contract::FrameId;

use super::{CentralBodyGravity, GravityField, ThirdBodySet, centrifugal, centrifugal_gradient};

/// The gravity model of a [`SpaceEnvironment`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
#[allow(
    clippy::large_enum_variant,
    reason = "the third bodies are inline so that the model is Copy without a heap"
)]
pub enum SpaceGravityModel {
    /// No gravity, for a test or a short coast far from every body.
    Zero,
    /// The central body alone.
    CentralBody(CentralBodyGravity),
    /// The central body and the tidal terms of third bodies.
    MultiBody {
        /// The central body at the frame center.
        central: CentralBodyGravity,
        /// The third bodies at their positions at the current time.
        third_bodies: ThirdBodySet,
    },
}

impl SpaceGravityModel {
    /// Whether every constant of the model is plausible.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        match self {
            Self::Zero => true,
            Self::CentralBody(central) => central.is_plausible(),
            Self::MultiBody {
                central,
                third_bodies,
            } => {
                central.is_plausible()
                    && third_bodies
                        .as_slice()
                        .iter()
                        .all(super::PointMassBody::is_plausible)
            }
        }
    }

    /// The gravity in m/s² at `r` in meters and its gradient in 1/s².
    #[must_use]
    pub fn evaluate(&self, r: &Vector3<f64>) -> (Vector3<f64>, Matrix3<f64>) {
        match self {
            Self::Zero => (Vector3::zeros(), Matrix3::zeros()),
            Self::CentralBody(central) => (central.acceleration(r), central.gradient(r)),
            Self::MultiBody {
                central,
                third_bodies,
            } => third_bodies.as_slice().iter().fold(
                (central.acceleration(r), central.gradient(r)),
                |(g, gradient), body| {
                    (
                        g + body.tidal_acceleration(r),
                        gradient + body.tidal_gradient(r),
                    )
                },
            ),
        }
    }
}

/// A Cartesian navigation frame centered on one body (ADR-0014).
///
/// The environment is `Copy`, so a filter keeps it by value in each
/// state of its history.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpaceEnvironment {
    /// The frame of the state, the observations, and the solutions.
    pub frame: FrameId,
    /// Rotation vector of the frame axes relative to inertial space, in
    /// rad/s in the frame axes. Zero for an inertial frame.
    pub frame_rotation_rad_per_s: [f64; 3],
    /// The gravity model.
    pub gravity: SpaceGravityModel,
}

impl SpaceEnvironment {
    /// An environment in a frame with inertial axes.
    #[must_use]
    pub const fn inertial(frame: FrameId, gravity: SpaceGravityModel) -> Self {
        Self {
            frame,
            frame_rotation_rad_per_s: [0.0; 3],
            gravity,
        }
    }

    /// Whether the rotation is finite and every gravity constant is
    /// plausible.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        self.frame_rotation_rad_per_s.iter().all(|w| w.is_finite()) && self.gravity.is_plausible()
    }
}

impl GravityField for SpaceEnvironment {
    /// The gravity plus the centrifugal term of a rotating frame.
    fn gravity(&self, p: &Vector3<f64>) -> (Vector3<f64>, Matrix3<f64>) {
        let omega = self.frame_rotation();
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
