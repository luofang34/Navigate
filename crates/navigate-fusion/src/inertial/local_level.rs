//! The local NED axes at a body-fixed position (ADR-0017).
//!
//! The north, east, and down unit vectors come directly from the sines
//! and cosines of the geodetic latitude and longitude. No term divides
//! by the cosine of the latitude, so the axes have a value at each
//! latitude, the poles included. At a pole the longitude sets the north
//! direction: north points from the pole along the meridian of that
//! longitude, to the equator.

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector3};
use navigate_contract::GeodeticPosition;
use navigate_geodesy::{
    BodyEllipsoid, BodyFixedPosition, body_fixed_to_geodetic, geodetic_to_body_fixed,
};

use crate::float_math::sin_cos;

/// The rotation from the local NED axes at the geodetic latitude
/// `latitude_rad` and the longitude `longitude_rad` to the body-fixed
/// axes. The columns are the north, east, and down unit vectors in the
/// body-fixed axes:
///
/// - north `[−sin φ·cos λ, −sin φ·sin λ, cos φ]`
/// - east `[−sin λ, cos λ, 0]`
/// - down `[−cos φ·cos λ, −cos φ·sin λ, −sin φ]`
///
/// The matrix is orthonormal at each latitude, ±90° included.
#[must_use]
pub fn ned_to_body_fixed_dcm(latitude_rad: f64, longitude_rad: f64) -> Matrix3<f64> {
    let (sin_lat, cos_lat) = sin_cos(latitude_rad);
    let (sin_lon, cos_lon) = sin_cos(longitude_rad);
    Matrix3::new(
        -sin_lat * cos_lon,
        -sin_lon,
        -cos_lat * cos_lon,
        -sin_lat * sin_lon,
        cos_lon,
        -cos_lat * sin_lon,
        cos_lat,
        0.0,
        -sin_lat,
    )
}

/// The geodetic position of one body-fixed point and its local NED
/// axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LocalLevel {
    /// The geodetic position on the ellipsoid.
    pub(crate) position: GeodeticPosition,
    /// The rotation from the local NED axes to the body-fixed axes.
    pub(crate) ned_to_bf: Matrix3<f64>,
}

impl LocalLevel {
    /// The local axes at the geodetic position `position`.
    pub(crate) fn at(position: GeodeticPosition) -> Self {
        Self {
            position,
            ned_to_bf: ned_to_body_fixed_dcm(position.latitude_rad, position.longitude_rad),
        }
    }

    /// The local axes at the body-fixed position `p` in meters on
    /// `ellipsoid`.
    pub(crate) fn at_body_fixed(p: &Vector3<f64>, ellipsoid: BodyEllipsoid) -> Self {
        let point = BodyFixedPosition::new(p.x, p.y, p.z);
        Self::at(body_fixed_to_geodetic(&point, ellipsoid))
    }

    /// The rotation from the body-fixed axes to the local NED axes.
    pub(crate) fn bf_to_ned(&self) -> Matrix3<f64> {
        self.ned_to_bf.transpose()
    }

    /// The rotation from the body-fixed axes to the local NED axes as a
    /// unit quaternion.
    pub(crate) fn bf_to_ned_quaternion(&self) -> UnitQuaternion<f64> {
        quaternion_of(&self.bf_to_ned())
    }

    /// The outward normal of the ellipsoid in the body-fixed axes: the
    /// gradient of the ellipsoidal height.
    pub(crate) fn up(&self) -> Vector3<f64> {
        -self.ned_to_bf.column(2).into_owned()
    }

    /// The covariance `block` in the local NED axes, rotated into the
    /// body-fixed axes. The result is symmetric.
    pub(crate) fn ned_to_bf_cov(&self, block: &Matrix3<f64>) -> Matrix3<f64> {
        symmetric(&(self.ned_to_bf * block * self.ned_to_bf.transpose()))
    }

    /// The covariance `block` in the body-fixed axes, rotated into the
    /// local NED axes. The result is symmetric.
    pub(crate) fn bf_to_ned_cov(&self, block: &Matrix3<f64>) -> Matrix3<f64> {
        symmetric(&(self.ned_to_bf.transpose() * block * self.ned_to_bf))
    }
}

/// The body-fixed position in meters of `position` on `ellipsoid`.
pub(crate) fn body_fixed(position: &GeodeticPosition, ellipsoid: BodyEllipsoid) -> Vector3<f64> {
    let point = geodetic_to_body_fixed(position, ellipsoid);
    Vector3::new(point.x_m, point.y_m, point.z_m)
}

/// The unit quaternion of an orthonormal rotation matrix.
pub(crate) fn quaternion_of(rotation: &Matrix3<f64>) -> UnitQuaternion<f64> {
    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(*rotation))
}

fn symmetric(m: &Matrix3<f64>) -> Matrix3<f64> {
    (m + m.transpose()) * 0.5
}
