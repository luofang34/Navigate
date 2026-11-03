//! Typed errors for geodesy constructors and track-relative geometry.

use thiserror::Error;

/// Errors raised by geodesy constructors and the track-deviation
/// helpers. Point-to-point conversion functions are total over plausible
/// inputs and return values, not results.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum GeodesyError {
    /// A local-tangent-plane origin failed the geodetic plausibility
    /// screen (non-finite component or latitude/longitude out of range).
    /// Rejecting at construction keeps every subsequent conversion total.
    #[error(
        "local tangent plane origin is not a plausible geodetic position: \
         latitude {latitude_rad} rad, longitude {longitude_rad} rad, \
         altitude {altitude_m} m"
    )]
    ImplausibleOrigin {
        /// Latitude of the rejected origin in radians.
        latitude_rad: f64,
        /// Longitude of the rejected origin in radians.
        longitude_rad: f64,
        /// Altitude of the rejected origin in meters.
        altitude_m: f64,
    },

    /// The ellipsoid cannot define geodetic coordinates: the semi-major
    /// axis is not finite and positive, the flattening is not in
    /// `[0, 1)`, or the frame is inertial.
    #[error(
        "ellipsoid is not plausible: semi-major axis {semi_major_axis_m} m, \
         flattening {flattening}"
    )]
    ImplausibleEllipsoid {
        /// Semi-major axis of the refused ellipsoid in meters.
        semi_major_axis_m: f64,
        /// Flattening of the refused ellipsoid.
        flattening: f64,
    },

    /// The track endpoints are closer than the 1 mm floor below which
    /// the track bearing is numerically meaningless: a track needs
    /// direction, and two coincident points define none.
    #[error(
        "track endpoints are {separation_m} m apart, below the 1e-3 m \
         floor needed to define a track direction"
    )]
    DegenerateTrack {
        /// Great-circle separation of the rejected track endpoints in
        /// meters.
        separation_m: f64,
    },
}
