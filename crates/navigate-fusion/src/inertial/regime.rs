//! The navigation regime of the inertial filter (ADR-0014, ADR-0017).
//!
//! The state is always in a body-centered Cartesian frame: the frame of
//! a [`SpaceEnvironment`]. A reference ellipsoid on the same frame is
//! optional. When the ellipsoid is attached, the filter has the
//! geodetic observation models and publishes a geodetic solution in the
//! local NED axes at the vehicle.

use nalgebra::{Matrix3, Vector3};
use navigate_contract::FrameId;
use navigate_geodesy::BodyEllipsoid;

use crate::observation::MeasurementKind;
use navigate_astrodynamics::{GravityField, PlanetaryBody, SpaceEnvironment};

/// The frame, the gravity, and the optional surface of the state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Regime {
    /// The frame of the state with its gravity and rotation.
    pub(crate) env: SpaceEnvironment,
    /// The reference ellipsoid of the geodetic models and the geodetic
    /// solution. Its frame is the frame of `env`.
    pub(crate) surface: Option<BodyEllipsoid>,
}

impl Regime {
    /// The rotating body-fixed frame of `body` with its ellipsoid.
    pub(crate) const fn surface_of(body: &PlanetaryBody) -> Self {
        Self {
            env: SpaceEnvironment::from_body(*body),
            surface: Some(body.ellipsoid),
        }
    }

    /// The frame of `env` without an ellipsoid.
    pub(crate) const fn space(env: SpaceEnvironment) -> Self {
        Self { env, surface: None }
    }

    /// The frame that every observation and IMU sample must declare.
    pub(crate) const fn frame(&self) -> FrameId {
        self.env.frame
    }

    /// Whether an undeclared frame is refused. Only the WGS84 surface of
    /// the Earth accepts an undeclared frame, as
    /// [`crate::NavigationFilter`] does.
    pub(crate) fn requires_explicit_frame(&self) -> bool {
        self.surface != Some(BodyEllipsoid::WGS84)
    }

    /// Whether the regime has the model of `kind`. The Cartesian kinds,
    /// the attitude fix, the radiometric kinds, and the line of sight need
    /// no ellipsoid. The geodetic kinds need one.
    pub(crate) const fn supports(&self, kind: MeasurementKind) -> bool {
        let cartesian = matches!(
            kind,
            MeasurementKind::CartesianPositionFix
                | MeasurementKind::CartesianVelocityFix
                | MeasurementKind::CartesianRange
                | MeasurementKind::AttitudeFix
                | MeasurementKind::Pseudorange
                | MeasurementKind::PseudorangeRate
                | MeasurementKind::TwoWayRange
                | MeasurementKind::TwoWayRangeRate
                | MeasurementKind::LineOfSight
        );
        let geodetic = matches!(
            kind,
            MeasurementKind::PositionFix
                | MeasurementKind::VelocityFix
                | MeasurementKind::Range
                | MeasurementKind::Altitude
                | MeasurementKind::MagneticAnomaly
                | MeasurementKind::TerrainClearance
                | MeasurementKind::VisualPose
        );
        cartesian || (geodetic && self.surface.is_some())
    }
}

impl GravityField for Regime {
    fn gravity(&self, p: &Vector3<f64>) -> (Vector3<f64>, Matrix3<f64>) {
        self.env.gravity(p)
    }

    fn frame_rotation(&self) -> Vector3<f64> {
        self.env.frame_rotation()
    }
}
