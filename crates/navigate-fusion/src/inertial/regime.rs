//! The two navigation regimes of the inertial filter (ADR-0014).

use std::sync::Arc;

use nalgebra::{Matrix3, Vector3};
use navigate_contract::FrameId;

use crate::environment::{GravityField, PlanetaryBody, SpaceEnvironment, SurfaceFrame};
use crate::observation::MeasurementKind;

/// The regime that a filter starts in, before the initialization fixes
/// the tangent plane.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Setup {
    /// A tangent plane on a rotating body.
    BodySurface(PlanetaryBody),
    /// A Cartesian frame centered on one body.
    CartesianSpace(Arc<SpaceEnvironment>),
}

impl Setup {
    pub(crate) fn frame(&self) -> FrameId {
        match self {
            Self::BodySurface(body) => body.ellipsoid.frame(),
            Self::CartesianSpace(env) => env.frame,
        }
    }
}

/// The active regime of an initialized filter.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Regime {
    /// The state is in a tangent plane fixed to a rotating body.
    BodySurface(Arc<SurfaceFrame>),
    /// The state is in a Cartesian frame centered on one body.
    CartesianSpace(Arc<SpaceEnvironment>),
}

impl Regime {
    /// The frame that every observation and IMU sample must declare.
    pub(crate) fn frame(&self) -> FrameId {
        match self {
            Self::BodySurface(surface) => surface.body.ellipsoid.frame(),
            Self::CartesianSpace(env) => env.frame,
        }
    }

    /// The surface frame, or `None` in a space regime.
    pub(crate) fn surface(&self) -> Option<&SurfaceFrame> {
        match self {
            Self::BodySurface(surface) => Some(surface),
            Self::CartesianSpace(_) => None,
        }
    }
}

/// Whether the regime of `setup` has the model of `kind`.
pub(crate) const fn supports(setup: &Setup, kind: MeasurementKind) -> bool {
    match setup {
        Setup::BodySurface(_) => matches!(
            kind,
            MeasurementKind::PositionFix
                | MeasurementKind::VelocityFix
                | MeasurementKind::Range
                | MeasurementKind::Altitude
                | MeasurementKind::MagneticAnomaly
                | MeasurementKind::TerrainClearance
                | MeasurementKind::AttitudeFix
                | MeasurementKind::VisualPose
        ),
        Setup::CartesianSpace(_) => matches!(
            kind,
            MeasurementKind::CartesianPositionFix
                | MeasurementKind::CartesianVelocityFix
                | MeasurementKind::CartesianRange
                | MeasurementKind::AttitudeFix
        ),
    }
}

impl GravityField for Regime {
    fn gravity(&self, p: &Vector3<f64>) -> (Vector3<f64>, Matrix3<f64>) {
        match self {
            Self::BodySurface(surface) => surface.gravity(p),
            Self::CartesianSpace(env) => env.gravity(p),
        }
    }

    fn frame_rotation(&self) -> Vector3<f64> {
        match self {
            Self::BodySurface(surface) => surface.frame_rotation(),
            Self::CartesianSpace(env) => env.frame_rotation(),
        }
    }
}
