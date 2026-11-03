//! Reference ellipsoids with derived constants (ADR-0013).
//!
//! [`BodyEllipsoid`] wraps a contract [`Ellipsoid`] and computes the
//! derived constants once. The conversions in this crate read the
//! constants and do not compute them again.

use core::f64::consts::{PI, TAU};

use navigate_contract::{Ellipsoid, FrameId};

use crate::error::GeodesyError;
use crate::float_math::{atan2, rem_euclid, sin, sin_cos, sqrt};

/// A plausible reference ellipsoid and its derived constants.
///
/// Only [`Self::try_from_contract`] and the presets make a value, so
/// every value has a positive semi-major axis and a flattening in
/// `[0, 1)`. A flattening of 0 gives a sphere. Then both eccentricities
/// are exactly 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyEllipsoid {
    contract: Ellipsoid,
    semi_minor_axis_m: f64,
    first_eccentricity_squared: f64,
    second_eccentricity_squared: f64,
    mean_radius_m: f64,
}

impl TryFrom<Ellipsoid> for BodyEllipsoid {
    type Error = GeodesyError;

    /// The same as [`BodyEllipsoid::try_from_contract`].
    fn try_from(ellipsoid: Ellipsoid) -> Result<Self, GeodesyError> {
        Self::try_from_contract(ellipsoid)
    }
}

impl BodyEllipsoid {
    /// [`Ellipsoid::WGS84`] with its derived constants.
    pub const WGS84: Self = Self::derive(Ellipsoid::WGS84);
    /// [`Ellipsoid::GRS80`] with its derived constants.
    pub const GRS80: Self = Self::derive(Ellipsoid::GRS80);
    /// [`Ellipsoid::MOON_ME`] with its derived constants.
    pub const MOON_ME: Self = Self::derive(Ellipsoid::MOON_ME);
    /// [`Ellipsoid::MOON_PA`] with its derived constants.
    pub const MOON_PA: Self = Self::derive(Ellipsoid::MOON_PA);
    /// [`Ellipsoid::MARS_IAU`] with its derived constants.
    pub const MARS_IAU: Self = Self::derive(Ellipsoid::MARS_IAU);

    /// Derives the constants of `ellipsoid`.
    ///
    /// # Errors
    ///
    /// [`GeodesyError::ImplausibleEllipsoid`] when
    /// [`Ellipsoid::is_plausible`] refuses `ellipsoid`.
    pub fn try_from_contract(ellipsoid: Ellipsoid) -> Result<Self, GeodesyError> {
        if !ellipsoid.is_plausible() {
            return Err(GeodesyError::ImplausibleEllipsoid {
                semi_major_axis_m: ellipsoid.semi_major_axis_m,
                flattening: ellipsoid.flattening,
            });
        }
        Ok(Self::derive(ellipsoid))
    }

    /// The derivation has no division by the eccentricity, so a sphere
    /// gives exact zeros.
    const fn derive(contract: Ellipsoid) -> Self {
        let a = contract.semi_major_axis_m;
        let f = contract.flattening;
        let semi_minor_axis_m = a * (1.0 - f);
        let first_eccentricity_squared = f * (2.0 - f);
        Self {
            contract,
            semi_minor_axis_m,
            first_eccentricity_squared,
            second_eccentricity_squared: first_eccentricity_squared
                / (1.0 - first_eccentricity_squared),
            mean_radius_m: (2.0 * a + semi_minor_axis_m) / 3.0,
        }
    }

    /// The contract ellipsoid.
    #[must_use]
    pub const fn contract(&self) -> Ellipsoid {
        self.contract
    }

    /// The body-fixed frame of the ellipsoid.
    #[must_use]
    pub const fn frame(&self) -> FrameId {
        self.contract.frame
    }

    /// IUGG mean radius `R₁ = (2a + b) / 3` in meters. The great-circle
    /// helpers use a sphere of this radius.
    #[must_use]
    pub const fn mean_radius_m(&self) -> f64 {
        self.mean_radius_m
    }

    /// Semi-major (equatorial) axis `a` in meters.
    #[must_use]
    pub const fn semi_major_axis_m(&self) -> f64 {
        self.contract.semi_major_axis_m
    }

    /// Semi-minor (polar) axis `b = a·(1 − f)` in meters.
    #[must_use]
    pub const fn semi_minor_axis_m(&self) -> f64 {
        self.semi_minor_axis_m
    }

    /// Flattening `f`. It has no unit.
    #[must_use]
    pub const fn flattening(&self) -> f64 {
        self.contract.flattening
    }

    /// First eccentricity squared `e² = f·(2 − f)`. It has no unit.
    #[must_use]
    pub const fn first_eccentricity_squared(&self) -> f64 {
        self.first_eccentricity_squared
    }

    /// Second eccentricity squared `e′² = e² / (1 − e²)`. It has no unit.
    #[must_use]
    pub const fn second_eccentricity_squared(&self) -> f64 {
        self.second_eccentricity_squared
    }

    /// The meridian and prime-vertical radii of curvature at
    /// `latitude_rad`, in meters, as `(meridian_m, prime_vertical_m)`.
    #[must_use]
    pub fn radii_of_curvature_m(&self, latitude_rad: f64) -> (f64, f64) {
        let a = self.semi_major_axis_m();
        let e2 = self.first_eccentricity_squared;
        let sin_lat = sin(latitude_rad);
        let w2 = 1.0 - e2 * sin_lat * sin_lat;
        let w = sqrt(w2);
        (a * (1.0 - e2) / (w2 * w), a / w)
    }

    /// Converts a planetocentric latitude of a surface point to the
    /// planetographic (geodetic) latitude: `tan φg = tan φc / (1 − e²)`.
    /// The `atan2` form is total at the poles. On a sphere the two
    /// latitudes are equal.
    #[must_use]
    pub fn planetocentric_to_geodetic_lat(&self, planetocentric_lat_rad: f64) -> f64 {
        let (sin_lat, cos_lat) = sin_cos(planetocentric_lat_rad);
        atan2(sin_lat, (1.0 - self.first_eccentricity_squared) * cos_lat)
    }

    /// Converts a planetographic (geodetic) latitude of a surface point
    /// to the planetocentric latitude: `tan φc = (1 − e²)·tan φg`.
    #[must_use]
    pub fn geodetic_to_planetocentric_lat(&self, geodetic_lat_rad: f64) -> f64 {
        let (sin_lat, cos_lat) = sin_cos(geodetic_lat_rad);
        atan2((1.0 - self.first_eccentricity_squared) * sin_lat, cos_lat)
    }
}

/// Converts a west-positive longitude, such as an IAU planetographic
/// longitude of Mars, to the east-positive longitude in `[-π, π]` that
/// [`navigate_contract::GeodeticPosition`] uses.
#[must_use]
pub fn west_to_east_lon_rad(west_lon_rad: f64) -> f64 {
    let east = rem_euclid(-west_lon_rad, TAU);
    if east > PI { east - TAU } else { east }
}

#[cfg(test)]
mod tests;
