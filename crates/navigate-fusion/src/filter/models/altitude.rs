//! Height above the reference ellipsoid of the tangent plane (ADR-0012,
//! ADR-0013).
//!
//! The model converts the state position to geodetic coordinates before
//! it reads the altitude. Thus the model has no flat-plane curvature
//! error. At 50 km from the tangent-plane origin, the plane `down` axis
//! is approximately 196 m away from the ellipsoidal height.

use nalgebra::{SMatrix, SVector, Vector3};
use navigate_contract::GeodeticPosition;
use navigate_geodesy::{LocalTangentPlane, NedOffset};

use crate::filter::{MeasurementModel, Vec6, rotation_between};

/// Ellipsoidal altitude measured against the state position.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AltitudeModel {
    plane: LocalTangentPlane,
    altitude_m: f64,
    variance_m2: f64,
}

impl AltitudeModel {
    pub(crate) const fn new(plane: LocalTangentPlane, altitude_m: f64, variance_m2: f64) -> Self {
        Self {
            plane,
            altitude_m,
            variance_m2,
        }
    }

    fn geodetic(&self, x: &Vec6) -> GeodeticPosition {
        self.plane.from_ned(&NedOffset::new(x[0], x[1], x[2]))
    }
}

impl MeasurementModel<1> for AltitudeModel {
    fn innovation(&self, x: &Vec6) -> SVector<f64, 1> {
        SVector::<f64, 1>::new(self.altitude_m - self.geodetic(x).altitude_m)
    }

    /// The local up direction at the state position, in plane axes. The
    /// altitude changes along this direction only.
    fn jacobian(&self, x: &Vec6) -> Option<SMatrix<f64, 1, 6>> {
        let local = LocalTangentPlane::new(self.geodetic(x), self.plane.ellipsoid()).ok()?;
        let up_plane = rotation_between(&local, &self.plane) * Vector3::new(0.0, 0.0, -1.0);
        Some(SMatrix::<f64, 1, 6>::new(
            up_plane.x, up_plane.y, up_plane.z, 0.0, 0.0, 0.0,
        ))
    }

    fn noise(&self) -> SMatrix<f64, 1, 1> {
        SMatrix::<f64, 1, 1>::new(self.variance_m2)
    }
}
