//! Gravity in a Cartesian frame centered on one body (ADR-0014,
//! ADR-0018).

use nalgebra::{Matrix3, Vector3};
use navigate_contract::FrameId;

use super::{
    CentralBodyGravity, GravityField, SphericalHarmonicField, ThirdBodySet, centrifugal,
    centrifugal_gradient,
};
use crate::body::PlanetaryBody;
use crate::drag::ExponentialDrag;
use crate::srp::CannonballSrp;

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
    /// A spherical harmonic field of the central body with `'static`
    /// coefficients, and the tidal terms of third bodies.
    SphericalHarmonics {
        /// The field of the central body at the frame center.
        field: SphericalHarmonicField<'static>,
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
            } => central.is_plausible() && third_bodies.is_plausible(),
            Self::SphericalHarmonics {
                field,
                third_bodies,
            } => field.is_plausible() && third_bodies.is_plausible(),
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
            } => third_bodies.add_tidal((central.acceleration(r), central.gradient(r)), r),
            Self::SphericalHarmonics {
                field,
                third_bodies,
            } => third_bodies.add_tidal(field.evaluate(r), r),
        }
    }
}

/// A Cartesian navigation frame centered on one body (ADR-0014,
/// ADR-0018).
///
/// The environment is `Copy` and `'static`, so a filter keeps it by
/// value in each state of its history. The solar radiation pressure and
/// the drag are optional. A force-model propagation uses them. An
/// accelerometer senses them, so [`GravityField::gravity`] does not
/// include them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpaceEnvironment {
    /// The frame of the state, the observations, and the solutions.
    pub frame: FrameId,
    /// Rotation vector of the frame axes relative to inertial space, in
    /// rad/s in the frame axes. Zero for an inertial frame.
    pub frame_rotation_rad_per_s: [f64; 3],
    /// The gravity model.
    pub gravity: SpaceGravityModel,
    /// The solar radiation pressure, if any.
    pub srp: Option<CannonballSrp>,
    /// The atmospheric drag, if any.
    pub drag: Option<ExponentialDrag>,
}

impl SpaceEnvironment {
    /// An environment in a frame with inertial axes, without solar
    /// radiation pressure and without drag.
    #[must_use]
    pub const fn inertial(frame: FrameId, gravity: SpaceGravityModel) -> Self {
        Self {
            frame,
            frame_rotation_rad_per_s: [0.0; 3],
            gravity,
            srp: None,
            drag: None,
        }
    }

    /// The body-fixed frame of `body`: the frame of its ellipsoid, the
    /// rotation `[0, 0, ω]` about the body-fixed `z` axis, and the point
    /// mass and the `J2` term of the body. With the centrifugal term of
    /// [`GravityField::gravity`], the gravity is the apparent gravity of
    /// [`PlanetaryBody::apparent_gravity_body_fixed`]. The environment has
    /// no solar radiation pressure and no drag.
    #[must_use]
    pub const fn from_body(body: PlanetaryBody) -> Self {
        Self {
            frame: body.ellipsoid.frame(),
            frame_rotation_rad_per_s: [0.0, 0.0, body.rotation_rate_rad_per_s],
            gravity: SpaceGravityModel::CentralBody(body.central_gravity()),
            srp: None,
            drag: None,
        }
    }

    /// This environment with the solar radiation pressure `srp`.
    #[must_use]
    pub const fn with_srp(mut self, srp: CannonballSrp) -> Self {
        self.srp = Some(srp);
        self
    }

    /// This environment with the atmospheric drag `drag`.
    #[must_use]
    pub const fn with_drag(mut self, drag: ExponentialDrag) -> Self {
        self.drag = Some(drag);
        self
    }

    /// Whether the rotation is finite and every gravity, pressure, and
    /// drag constant is plausible.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        self.force_model().is_plausible()
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
