//! WGS84 ellipsoid constants.
//!
//! The two defining parameters are the semi-major axis and the
//! flattening; everything else is derived from them at compile time so
//! the values can never drift apart.

/// Semi-major (equatorial) axis of the WGS84 ellipsoid in meters.
/// Defining parameter.
pub const SEMI_MAJOR_AXIS_M: f64 = 6_378_137.0;

/// Flattening of the WGS84 ellipsoid, dimensionless. Defining parameter
/// (the reciprocal 298.257223563 is the published form).
pub const FLATTENING: f64 = 1.0 / 298.257_223_563;

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

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn derived_constants_match_published_values() {
        // Published derived values for WGS84 (NIMA TR8350.2).
        assert!((SEMI_MINOR_AXIS_M - 6_356_752.314_245_179).abs() < 1e-6);
        assert!((FIRST_ECCENTRICITY_SQUARED - 6.694_379_990_141_3e-3).abs() < 1e-15);
        assert!((SECOND_ECCENTRICITY_SQUARED - 6.739_496_742_276_4e-3).abs() < 1e-15);
        assert!((MEAN_RADIUS_M - 6_371_008.771_415_059).abs() < 1e-6);
    }
}
