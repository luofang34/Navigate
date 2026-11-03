//! The WGS84 ellipsoid: its constants and the Earth-only shorthand.
//!
//! The two defining parameters are the semi-major axis and the
//! flattening. They come from [`Ellipsoid::WGS84`]. Everything else is
//! derived from them at compile time so the values can never drift
//! apart.
//!
//! Each function here calls the crate-root function of the same purpose
//! with [`BodyEllipsoid::WGS84`]. Use them only where the code is Earth
//! code by intent, for example a Web Mercator tile or an ECEF sensor
//! model. Code that has an ellipsoid at hand passes it to the crate-root
//! function instead.

use navigate_contract::{Ellipsoid, GeodeticPosition};

use crate::ecef::{BodyFixedPosition, body_fixed_to_geodetic, geodetic_to_body_fixed};
use crate::ellipsoid::BodyEllipsoid;
use crate::error::GeodesyError;
use crate::great_circle;
use crate::local_plane::LocalTangentPlane;

/// Semi-major (equatorial) axis of the WGS84 ellipsoid in meters.
/// Defining parameter.
pub const SEMI_MAJOR_AXIS_M: f64 = Ellipsoid::WGS84.semi_major_axis_m;

/// Flattening of the WGS84 ellipsoid, dimensionless. Defining parameter
/// (the reciprocal 298.257223563 is the published form).
pub const FLATTENING: f64 = Ellipsoid::WGS84.flattening;

/// Semi-minor (polar) axis in meters: `a·(1 − f)`.
pub const SEMI_MINOR_AXIS_M: f64 = SEMI_MAJOR_AXIS_M * (1.0 - FLATTENING);

/// First eccentricity squared, dimensionless: `f·(2 − f)`.
pub const FIRST_ECCENTRICITY_SQUARED: f64 = FLATTENING * (2.0 - FLATTENING);

/// Second eccentricity squared, dimensionless: `e² / (1 − e²)`.
pub const SECOND_ECCENTRICITY_SQUARED: f64 =
    FIRST_ECCENTRICITY_SQUARED / (1.0 - FIRST_ECCENTRICITY_SQUARED);

/// IUGG mean radius `R₁ = (2a + b) / 3` in meters. The sphere the
/// great-circle helpers run on.
pub const MEAN_RADIUS_M: f64 = (2.0 * SEMI_MAJOR_AXIS_M + SEMI_MINOR_AXIS_M) / 3.0;

/// A body-fixed position on the Earth (Earth-centered, Earth-fixed).
pub type EcefPosition = BodyFixedPosition;

/// [`geodetic_to_body_fixed`] on WGS84.
#[must_use]
pub fn geodetic_to_ecef(geodetic: &GeodeticPosition) -> EcefPosition {
    geodetic_to_body_fixed(geodetic, BodyEllipsoid::WGS84)
}

/// [`body_fixed_to_geodetic`] on WGS84.
#[must_use]
pub fn ecef_to_geodetic(ecef: &EcefPosition) -> GeodeticPosition {
    body_fixed_to_geodetic(ecef, BodyEllipsoid::WGS84)
}

/// [`LocalTangentPlane::new`] on WGS84.
///
/// # Errors
///
/// [`GeodesyError::ImplausibleOrigin`] when `origin` fails the geodetic
/// plausibility screen.
pub fn tangent_plane(origin: GeodeticPosition) -> Result<LocalTangentPlane, GeodesyError> {
    LocalTangentPlane::new(origin, BodyEllipsoid::WGS84)
}

/// [`great_circle::distance_m`] on the WGS84 mean-radius sphere.
#[must_use]
pub fn distance_m(a: &GeodeticPosition, b: &GeodeticPosition) -> f64 {
    great_circle::distance_m(a, b, BodyEllipsoid::WGS84)
}

/// [`great_circle::cross_track_m`] on the WGS84 mean-radius sphere.
///
/// # Errors
///
/// [`GeodesyError::DegenerateTrack`] when the endpoints are closer than
/// 1 mm.
pub fn cross_track_m(
    point: &GeodeticPosition,
    track_start: &GeodeticPosition,
    track_end: &GeodeticPosition,
) -> Result<f64, GeodesyError> {
    great_circle::cross_track_m(point, track_start, track_end, BodyEllipsoid::WGS84)
}

/// [`great_circle::along_track_m`] on the WGS84 mean-radius sphere.
///
/// # Errors
///
/// [`GeodesyError::DegenerateTrack`] when the endpoints are closer than
/// 1 mm.
pub fn along_track_m(
    point: &GeodeticPosition,
    track_start: &GeodeticPosition,
    track_end: &GeodeticPosition,
) -> Result<f64, GeodesyError> {
    great_circle::along_track_m(point, track_start, track_end, BodyEllipsoid::WGS84)
}

/// [`great_circle::cross_track_from_course_m`] on the WGS84 mean-radius
/// sphere.
#[must_use]
pub fn cross_track_from_course_m(
    point: &GeodeticPosition,
    anchor: &GeodeticPosition,
    course_rad: f64,
) -> f64 {
    great_circle::cross_track_from_course_m(point, anchor, course_rad, BodyEllipsoid::WGS84)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_constants_match_published_values() {
        // Published derived values for WGS84 (NIMA TR8350.2).
        assert!((SEMI_MINOR_AXIS_M - 6_356_752.314_245_179).abs() < 1e-6);
        assert!((FIRST_ECCENTRICITY_SQUARED - 6.694_379_990_141_3e-3).abs() < 1e-15);
        assert!((SECOND_ECCENTRICITY_SQUARED - 6.739_496_742_276_4e-3).abs() < 1e-15);
        assert!((MEAN_RADIUS_M - 6_371_008.771_415_059).abs() < 1e-6);
    }

    #[test]
    fn the_shorthand_is_the_explicit_function_on_wgs84() {
        let a = GeodeticPosition::new(0.82, 0.14, 420.0);
        let b = GeodeticPosition::new(0.83, 0.16, 0.0);
        let c = GeodeticPosition::new(0.825, 0.13, 0.0);
        let earth = BodyEllipsoid::WGS84;
        assert_eq!(geodetic_to_ecef(&a), geodetic_to_body_fixed(&a, earth));
        let ecef = geodetic_to_ecef(&a);
        assert_eq!(
            ecef_to_geodetic(&ecef),
            body_fixed_to_geodetic(&ecef, earth)
        );
        assert_eq!(tangent_plane(a), LocalTangentPlane::new(a, earth));
        assert_eq!(distance_m(&a, &b), great_circle::distance_m(&a, &b, earth));
        assert_eq!(
            cross_track_m(&c, &a, &b),
            great_circle::cross_track_m(&c, &a, &b, earth)
        );
        assert_eq!(
            along_track_m(&c, &a, &b),
            great_circle::along_track_m(&c, &a, &b, earth)
        );
        assert_eq!(
            cross_track_from_course_m(&c, &a, 0.3),
            great_circle::cross_track_from_course_m(&c, &a, 0.3, earth)
        );
    }
}
