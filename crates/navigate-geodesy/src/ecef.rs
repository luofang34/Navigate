//! Body-centered body-fixed positions and their conversions to and from
//! geodetic coordinates on a [`BodyEllipsoid`] (ADR-0013).
//!
//! The contract crate deliberately carries no body-fixed type; this
//! crate owns the frame because it owns the conversion math.
//! [`EcefPosition`] is the name of the type on the Earth.

use core::f64::consts::FRAC_PI_2;

use navigate_contract::GeodeticPosition;

use crate::ellipsoid::BodyEllipsoid;

/// Below this ratio of the cylindrical radius to the semi-major axis, a
/// position is on the polar axis. There the latitude is ±π/2 and the
/// Bowring form is not necessary.
const POLAR_AXIS_RATIO: f64 = 1e-12;

/// A position in the body-centered body-fixed frame of an ellipsoid: X
/// toward the prime meridian at the equator, Y toward 90°E at the
/// equator, Z toward the north pole. Meters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyFixedPosition {
    /// X component in meters.
    pub x_m: f64,
    /// Y component in meters.
    pub y_m: f64,
    /// Z component in meters.
    pub z_m: f64,
}

/// A body-fixed position on the Earth.
pub type EcefPosition = BodyFixedPosition;

impl BodyFixedPosition {
    /// Builds a position from its components.
    #[must_use]
    pub const fn new(x_m: f64, y_m: f64, z_m: f64) -> Self {
        Self { x_m, y_m, z_m }
    }

    /// Euclidean distance to `other` in meters.
    #[must_use]
    pub fn distance_m(&self, other: &Self) -> f64 {
        let dx = self.x_m - other.x_m;
        let dy = self.y_m - other.y_m;
        let dz = self.z_m - other.z_m;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

/// Converts a WGS84 geodetic position to ECEF. Closed form, exact to
/// floating-point rounding.
#[must_use]
pub fn geodetic_to_ecef(geodetic: &GeodeticPosition) -> EcefPosition {
    geodetic_to_body_fixed(geodetic, &BodyEllipsoid::WGS84)
}

/// Converts an ECEF position to WGS84 geodetic coordinates with
/// [`body_fixed_to_geodetic`].
#[must_use]
pub fn ecef_to_geodetic(ecef: &EcefPosition) -> GeodeticPosition {
    body_fixed_to_geodetic(ecef, &BodyEllipsoid::WGS84)
}

/// Converts a geodetic position on `ellipsoid` to the body-fixed frame
/// of `ellipsoid`. Closed form, exact to floating-point rounding.
#[must_use]
pub fn geodetic_to_body_fixed(
    geodetic: &GeodeticPosition,
    ellipsoid: &BodyEllipsoid,
) -> BodyFixedPosition {
    let (sin_lat, cos_lat) = geodetic.latitude_rad.sin_cos();
    let (sin_lon, cos_lon) = geodetic.longitude_rad.sin_cos();
    let e2 = ellipsoid.first_eccentricity_squared();
    let prime_vertical_m = ellipsoid.semi_major_axis_m() / (1.0 - e2 * sin_lat * sin_lat).sqrt();
    let h = geodetic.altitude_m;
    BodyFixedPosition {
        x_m: (prime_vertical_m + h) * cos_lat * cos_lon,
        y_m: (prime_vertical_m + h) * cos_lat * sin_lon,
        z_m: (prime_vertical_m * (1.0 - e2) + h) * sin_lat,
    }
}

/// Converts a body-fixed position to geodetic coordinates on
/// `ellipsoid`.
///
/// Method: Bowring's parametric-latitude scheme with a fixed second pass
/// (two evaluations, no unbounded iteration). The first pass seeds the
/// parametric latitude from the spherical direction; the second re-seeds
/// it from the first-pass geodetic latitude. Height uses the
/// all-latitude form `h = p·cosφ + z·sinφ − a²/N`, which stays stable at
/// the poles where the classic `p/cosφ − N` divides by zero.
///
/// No step divides by the eccentricity. On a sphere (`f = 0`) the
/// correction terms are exactly 0, so the latitude is `atan2(z, p)` and
/// the height is the radius minus `a`.
///
/// A position closer to the polar axis than `1e-12·a` takes a direct
/// branch: the latitude is ±π/2 by the sign of `z` and the height is
/// `|z| − b`.
///
/// Accuracy: round trips through [`geodetic_to_body_fixed`] agree to
/// better than 1e-6 m over the tested envelopes of the preset ellipsoids
/// (pinned by tests across latitudes including the poles and the
/// equator).
///
/// Where longitude is undefined (on the polar axis) it is reported as 0
/// by the `atan2(0, 0)` convention.
#[must_use]
pub fn body_fixed_to_geodetic(
    pos: &BodyFixedPosition,
    ellipsoid: &BodyEllipsoid,
) -> GeodeticPosition {
    let a = ellipsoid.semi_major_axis_m();
    let b = ellipsoid.semi_minor_axis_m();
    let p = pos.x_m.hypot(pos.y_m);
    let longitude_rad = pos.y_m.atan2(pos.x_m);
    if p <= POLAR_AXIS_RATIO * a {
        let latitude_rad = if pos.z_m < 0.0 { -FRAC_PI_2 } else { FRAC_PI_2 };
        return GeodeticPosition::new(latitude_rad, longitude_rad, pos.z_m.abs() - b);
    }

    let beta = (pos.z_m * a).atan2(p * b);
    let first_pass = bowring_latitude_rad(p, pos.z_m, beta, ellipsoid);
    let (sin_first, cos_first) = first_pass.sin_cos();
    let refined_beta = ((1.0 - ellipsoid.flattening()) * sin_first).atan2(cos_first);
    let latitude_rad = bowring_latitude_rad(p, pos.z_m, refined_beta, ellipsoid);

    let (sin_lat, cos_lat) = latitude_rad.sin_cos();
    let e2 = ellipsoid.first_eccentricity_squared();
    let prime_vertical_m = a / (1.0 - e2 * sin_lat * sin_lat).sqrt();
    let altitude_m = p * cos_lat + pos.z_m * sin_lat - a * a / prime_vertical_m;

    GeodeticPosition::new(latitude_rad, longitude_rad, altitude_m)
}

/// One Bowring evaluation: geodetic latitude from the parametric
/// latitude `beta` and the cylindrical coordinates `(p, z)`.
fn bowring_latitude_rad(p: f64, z: f64, beta: f64, ellipsoid: &BodyEllipsoid) -> f64 {
    let (sin_beta, cos_beta) = beta.sin_cos();
    let numerator = z + ellipsoid.second_eccentricity_squared()
        * ellipsoid.semi_minor_axis_m()
        * sin_beta
        * sin_beta
        * sin_beta;
    let denominator = p - ellipsoid.first_eccentricity_squared()
        * ellipsoid.semi_major_axis_m()
        * cos_beta
        * cos_beta
        * cos_beta;
    numerator.atan2(denominator)
}

#[cfg(test)]
mod tests;
