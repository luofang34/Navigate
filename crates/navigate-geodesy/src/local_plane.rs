//! Local tangent plane: North-East-Down offsets relative to a geodetic
//! origin, converted through the body-fixed frame with an exact
//! rotation — no small-angle approximations, so accuracy degrades only
//! with the (real) curvature divergence between the plane and the
//! ellipsoid.

use navigate_contract::GeodeticPosition;

use crate::ecef::{BodyFixedPosition, body_fixed_to_geodetic, geodetic_to_body_fixed};
use crate::ellipsoid::BodyEllipsoid;
use crate::error::GeodesyError;

/// An offset in a local North-East-Down frame. Down is positive toward
/// the ellipsoid, along the origin's ellipsoidal normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NedOffset {
    /// North component in meters.
    pub north_m: f64,
    /// East component in meters.
    pub east_m: f64,
    /// Down component in meters (positive descending).
    pub down_m: f64,
}

impl NedOffset {
    /// Builds an offset from its components.
    #[must_use]
    pub const fn new(north_m: f64, east_m: f64, down_m: f64) -> Self {
        Self {
            north_m,
            east_m,
            down_m,
        }
    }
}

/// A local tangent plane anchored at a geodetic origin on one
/// [`BodyEllipsoid`]. Precomputes the origin's body-fixed position and
/// the body-fixed→NED rotation so conversions are pure arithmetic. The
/// plane reads and returns geodetic positions on its own ellipsoid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalTangentPlane {
    ellipsoid: BodyEllipsoid,
    origin: GeodeticPosition,
    origin_ecef: BodyFixedPosition,
    sin_lat: f64,
    cos_lat: f64,
    sin_lon: f64,
    cos_lon: f64,
}

impl LocalTangentPlane {
    /// Anchors a plane at `origin` on [`BodyEllipsoid::WGS84`].
    ///
    /// # Errors
    ///
    /// [`GeodesyError::ImplausibleOrigin`] when `origin` fails the
    /// geodetic plausibility screen.
    pub fn new(origin: GeodeticPosition) -> Result<Self, GeodesyError> {
        Self::with_ellipsoid(origin, BodyEllipsoid::WGS84)
    }

    /// Anchors a plane at `origin` on `ellipsoid`. Rejecting an
    /// implausible origin here keeps [`Self::to_ned`] and
    /// [`Self::from_ned`] total.
    ///
    /// # Errors
    ///
    /// [`GeodesyError::ImplausibleOrigin`] when `origin` fails the
    /// geodetic plausibility screen.
    pub fn with_ellipsoid(
        origin: GeodeticPosition,
        ellipsoid: BodyEllipsoid,
    ) -> Result<Self, GeodesyError> {
        if !origin.is_plausible() {
            return Err(GeodesyError::ImplausibleOrigin {
                latitude_rad: origin.latitude_rad,
                longitude_rad: origin.longitude_rad,
                altitude_m: origin.altitude_m,
            });
        }
        let (sin_lat, cos_lat) = origin.latitude_rad.sin_cos();
        let (sin_lon, cos_lon) = origin.longitude_rad.sin_cos();
        Ok(Self {
            ellipsoid,
            origin,
            origin_ecef: geodetic_to_body_fixed(&origin, &ellipsoid),
            sin_lat,
            cos_lat,
            sin_lon,
            cos_lon,
        })
    }

    /// The anchoring origin.
    #[must_use]
    pub const fn origin(&self) -> GeodeticPosition {
        self.origin
    }

    /// The ellipsoid of the origin and of every converted position.
    #[must_use]
    pub const fn ellipsoid(&self) -> BodyEllipsoid {
        self.ellipsoid
    }

    /// The body-fixed position of the origin. A gravity model reads it
    /// to find the body-centered position of a plane offset.
    #[must_use]
    pub const fn origin_body_fixed(&self) -> BodyFixedPosition {
        self.origin_ecef
    }

    /// Expresses `position` as a NED offset from the origin: the
    /// body-fixed delta rotated into the origin's tangent frame.
    #[must_use]
    pub fn to_ned(&self, position: &GeodeticPosition) -> NedOffset {
        let ecef = geodetic_to_body_fixed(position, &self.ellipsoid);
        let dx = ecef.x_m - self.origin_ecef.x_m;
        let dy = ecef.y_m - self.origin_ecef.y_m;
        let dz = ecef.z_m - self.origin_ecef.z_m;
        // Rows of the body-fixed→NED rotation for the origin's latitude and
        // longitude.
        NedOffset {
            north_m: -self.sin_lat * self.cos_lon * dx - self.sin_lat * self.sin_lon * dy
                + self.cos_lat * dz,
            east_m: -self.sin_lon * dx + self.cos_lon * dy,
            down_m: -self.cos_lat * self.cos_lon * dx
                - self.cos_lat * self.sin_lon * dy
                - self.sin_lat * dz,
        }
    }

    /// Resolves a NED offset back to a geodetic position: the transpose
    /// rotation into the body-fixed frame, then
    /// [`body_fixed_to_geodetic`].
    #[must_use]
    pub fn from_ned(&self, offset: &NedOffset) -> GeodeticPosition {
        let n = offset.north_m;
        let e = offset.east_m;
        let d = offset.down_m;
        let ecef = BodyFixedPosition {
            x_m: self.origin_ecef.x_m
                - self.sin_lat * self.cos_lon * n
                - self.sin_lon * e
                - self.cos_lat * self.cos_lon * d,
            y_m: self.origin_ecef.y_m - self.sin_lat * self.sin_lon * n + self.cos_lon * e
                - self.cos_lat * self.sin_lon * d,
            z_m: self.origin_ecef.z_m + self.cos_lat * n - self.sin_lat * d,
        };
        body_fixed_to_geodetic(&ecef, &self.ellipsoid)
    }

    /// Computes the exact 3x3 rotation matrix that maps a North-East-Down
    /// vector from `self` into `other`.
    #[must_use]
    pub fn ned_rotation_to(&self, other: &Self) -> [[f64; 3]; 3] {
        let a = self.body_fixed_to_ned_rows();
        let b = other.body_fixed_to_ned_rows();
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                r[i][j] = b[i][0] * a[j][0] + b[i][1] * a[j][1] + b[i][2] * a[j][2];
            }
        }
        r
    }

    /// The rows of the rotation from the body-fixed axes to the NED axes
    /// of the origin. The transpose rotates a NED vector into the
    /// body-fixed axes.
    #[must_use]
    pub const fn body_fixed_to_ned_rows(&self) -> [[f64; 3]; 3] {
        [
            [
                -self.sin_lat * self.cos_lon,
                -self.sin_lat * self.sin_lon,
                self.cos_lat,
            ],
            [-self.sin_lon, self.cos_lon, 0.0],
            [
                -self.cos_lat * self.cos_lon,
                -self.cos_lat * self.sin_lon,
                -self.sin_lat,
            ],
        ]
    }
}

#[cfg(test)]
mod tests;
