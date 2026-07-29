//! Earth-centered Earth-fixed (ECEF) positions and their conversions to
//! and from WGS84 geodetic coordinates.
//!
//! The contract crate deliberately carries no ECEF type; this crate owns
//! the frame because it owns the conversion math.

use navigate_contract::GeodeticPosition;

use crate::wgs84;

/// A position in the Earth-centered Earth-fixed frame: X toward the
/// prime meridian at the equator, Y toward 90°E at the equator, Z toward
/// the north pole. Meters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EcefPosition {
    /// X component in meters.
    pub x_m: f64,
    /// Y component in meters.
    pub y_m: f64,
    /// Z component in meters.
    pub z_m: f64,
}

impl EcefPosition {
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
    let (sin_lat, cos_lat) = geodetic.latitude_rad.sin_cos();
    let (sin_lon, cos_lon) = geodetic.longitude_rad.sin_cos();
    let prime_vertical_m = wgs84::SEMI_MAJOR_AXIS_M
        / (1.0 - wgs84::FIRST_ECCENTRICITY_SQUARED * sin_lat * sin_lat).sqrt();
    let h = geodetic.altitude_m;
    EcefPosition {
        x_m: (prime_vertical_m + h) * cos_lat * cos_lon,
        y_m: (prime_vertical_m + h) * cos_lat * sin_lon,
        z_m: (prime_vertical_m * (1.0 - wgs84::FIRST_ECCENTRICITY_SQUARED) + h) * sin_lat,
    }
}

/// Converts an ECEF position to WGS84 geodetic coordinates.
///
/// Method: Bowring's parametric-latitude scheme with a fixed second pass
/// (two evaluations, no unbounded iteration). The first pass seeds the
/// parametric latitude from the spherical direction; the second re-seeds
/// it from the first-pass geodetic latitude. Height uses the
/// all-latitude form `h = p·cosφ + z·sinφ − a²/N`, which stays stable at
/// the poles where the classic `p/cosφ − N` divides by zero.
///
/// Accuracy: round trips through [`geodetic_to_ecef`] agree to better
/// than 1e-6 m over the tested envelope of −400 m to 100 km above the
/// ellipsoid (pinned by tests across latitudes including the poles and
/// the equator).
///
/// Where longitude is undefined (on the polar axis) it is reported as 0
/// by the `atan2(0, 0)` convention.
#[must_use]
pub fn ecef_to_geodetic(ecef: &EcefPosition) -> GeodeticPosition {
    let p = ecef.x_m.hypot(ecef.y_m);
    let longitude_rad = ecef.y_m.atan2(ecef.x_m);

    let beta = (ecef.z_m * wgs84::SEMI_MAJOR_AXIS_M).atan2(p * wgs84::SEMI_MINOR_AXIS_M);
    let first_pass = bowring_latitude_rad(p, ecef.z_m, beta);
    let (sin_first, cos_first) = first_pass.sin_cos();
    let refined_beta = ((1.0 - wgs84::FLATTENING) * sin_first).atan2(cos_first);
    let latitude_rad = bowring_latitude_rad(p, ecef.z_m, refined_beta);

    let (sin_lat, cos_lat) = latitude_rad.sin_cos();
    let prime_vertical_m = wgs84::SEMI_MAJOR_AXIS_M
        / (1.0 - wgs84::FIRST_ECCENTRICITY_SQUARED * sin_lat * sin_lat).sqrt();
    let altitude_m = p * cos_lat + ecef.z_m * sin_lat
        - wgs84::SEMI_MAJOR_AXIS_M * wgs84::SEMI_MAJOR_AXIS_M / prime_vertical_m;

    GeodeticPosition::new(latitude_rad, longitude_rad, altitude_m)
}

/// One Bowring evaluation: geodetic latitude from the parametric
/// latitude `beta` and the cylindrical coordinates `(p, z)`.
fn bowring_latitude_rad(p: f64, z: f64, beta: f64) -> f64 {
    let (sin_beta, cos_beta) = beta.sin_cos();
    let numerator = z + wgs84::SECOND_ECCENTRICITY_SQUARED
        * wgs84::SEMI_MINOR_AXIS_M
        * sin_beta
        * sin_beta
        * sin_beta;
    let denominator = p - wgs84::FIRST_ECCENTRICITY_SQUARED
        * wgs84::SEMI_MAJOR_AXIS_M
        * cos_beta
        * cos_beta
        * cos_beta;
    numerator.atan2(denominator)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;
    use core::f64::consts::FRAC_PI_2;

    #[test]
    fn equator_prime_meridian_is_on_the_x_axis() {
        let ecef = geodetic_to_ecef(&GeodeticPosition::new(0.0, 0.0, 0.0));
        assert!((ecef.x_m - wgs84::SEMI_MAJOR_AXIS_M).abs() < 1e-9);
        assert!(ecef.y_m.abs() < 1e-9);
        assert!(ecef.z_m.abs() < 1e-9);
    }

    #[test]
    fn north_pole_is_on_the_z_axis_at_the_semi_minor_axis() {
        let ecef = geodetic_to_ecef(&GeodeticPosition::new(FRAC_PI_2, 0.0, 0.0));
        assert!(ecef.x_m.abs() < 1e-8);
        assert!(ecef.y_m.abs() < 1e-8);
        assert!((ecef.z_m - wgs84::SEMI_MINOR_AXIS_M).abs() < 1e-6);
    }

    #[test]
    fn round_trip_error_stays_below_a_micrometer() {
        let latitudes_deg: [f64; 8] = [-90.0, -60.0, -33.5, 0.0, 20.0, 45.0, 89.9, 90.0];
        let longitudes_deg: [f64; 5] = [-179.0, -73.8, 0.0, 8.55, 179.999];
        let altitudes_m: [f64; 5] = [-400.0, 0.0, 11_000.0, 80_000.0, 100_000.0];
        for lat in latitudes_deg {
            for lon in longitudes_deg {
                for alt in altitudes_m {
                    let original = GeodeticPosition::new(lat.to_radians(), lon.to_radians(), alt);
                    let ecef = geodetic_to_ecef(&original);
                    let round_tripped = ecef_to_geodetic(&ecef);
                    // The metric-space check: re-projecting through the
                    // closed-form forward conversion measures the round-trip
                    // error in meters without a pole-singular longitude
                    // comparison.
                    let reprojected = geodetic_to_ecef(&round_tripped);
                    let error_m = ecef.distance_m(&reprojected);
                    assert!(
                        error_m < 1e-6,
                        "round trip error {error_m} m at lat {lat}° lon {lon}° alt {alt} m"
                    );
                    assert!(
                        (round_tripped.altitude_m - alt).abs() < 1e-6,
                        "altitude error at lat {lat}° lon {lon}° alt {alt} m: {}",
                        round_tripped.altitude_m
                    );
                    assert!(round_tripped.is_plausible());
                }
            }
        }
    }

    #[test]
    fn polar_axis_reports_latitude_of_ninety_degrees() {
        let below_pole = EcefPosition::new(0.0, 0.0, wgs84::SEMI_MINOR_AXIS_M - 1_000.0);
        let geodetic = ecef_to_geodetic(&below_pole);
        assert!((geodetic.latitude_rad - FRAC_PI_2).abs() < 1e-12);
        assert!((geodetic.altitude_m + 1_000.0).abs() < 1e-6);
        assert!((geodetic.longitude_rad).abs() < 1e-12);
    }
}
