//! Vertical datums: the height of an equipotential surface above a
//! reference ellipsoid (ADR-0015).
//!
//! The geoid of the Earth, the selenoid of the Moon, and the areoid of
//! Mars are equipotential surfaces. The undulation `N` is the height of
//! that surface above the reference ellipsoid. An orthometric height `H`
//! is a height above the equipotential surface, so the ellipsoidal
//! height is `h = H + N`.
//!
//! A terrain database usually gives orthometric heights, and a
//! navigation solution gives ellipsoidal heights. A [`GeoidUndulation`]
//! model connects the two. The two conversion functions refuse a
//! position outside the model coverage, a non-finite value, and a height
//! on a different ellipsoid. They do not substitute a zero undulation.

use navigate_contract::Ellipsoid;

/// A model of the undulation `N` of one equipotential surface above one
/// reference ellipsoid.
pub trait GeoidUndulation {
    /// The reference ellipsoid that the model measures `N` from.
    fn ellipsoid(&self) -> Ellipsoid;

    /// The undulation `N` in meters at a planetographic latitude and an
    /// east-positive longitude on [`Self::ellipsoid`]. The result is
    /// `None` outside the coverage of the model.
    fn undulation_m(&self, latitude_rad: f64, longitude_rad: f64) -> Option<f64>;
}

/// A model with `N = 0` everywhere: the elevations of the data are
/// already ellipsoidal heights on the wrapped ellipsoid.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZeroUndulation(pub Ellipsoid);

impl GeoidUndulation for ZeroUndulation {
    fn ellipsoid(&self) -> Ellipsoid {
        self.0
    }

    /// Zero at every finite position, and `None` at a non-finite one.
    fn undulation_m(&self, latitude_rad: f64, longitude_rad: f64) -> Option<f64> {
        (latitude_rad.is_finite() && longitude_rad.is_finite()).then_some(0.0)
    }
}

/// The ellipsoidal height `h = H + N` in meters of the orthometric
/// height `orthometric_m` at the given position on `ellipsoid`.
///
/// The result is `None` when `ellipsoid` is not the ellipsoid of
/// `model`, when the position is outside the model coverage, or when an
/// input, `N`, or the result is not finite.
#[must_use]
pub fn orthometric_to_ellipsoidal_m<M: GeoidUndulation + ?Sized>(
    model: &M,
    ellipsoid: Ellipsoid,
    latitude_rad: f64,
    longitude_rad: f64,
    orthometric_m: f64,
) -> Option<f64> {
    let undulation_m = admitted_undulation_m(model, ellipsoid, latitude_rad, longitude_rad)?;
    finite(orthometric_m + undulation_m)
}

/// The orthometric height `H = h − N` in meters of the ellipsoidal
/// height `ellipsoidal_m` at the given position on `ellipsoid`.
///
/// The result is `None` under the conditions of
/// [`orthometric_to_ellipsoidal_m`].
#[must_use]
pub fn ellipsoidal_to_orthometric_m<M: GeoidUndulation + ?Sized>(
    model: &M,
    ellipsoid: Ellipsoid,
    latitude_rad: f64,
    longitude_rad: f64,
    ellipsoidal_m: f64,
) -> Option<f64> {
    let undulation_m = admitted_undulation_m(model, ellipsoid, latitude_rad, longitude_rad)?;
    finite(ellipsoidal_m - undulation_m)
}

/// The undulation of `model` at the position, when the position, the
/// ellipsoid, and the value are admissible.
fn admitted_undulation_m<M: GeoidUndulation + ?Sized>(
    model: &M,
    ellipsoid: Ellipsoid,
    latitude_rad: f64,
    longitude_rad: f64,
) -> Option<f64> {
    if model.ellipsoid() != ellipsoid || !latitude_rad.is_finite() || !longitude_rad.is_finite() {
        return None;
    }
    finite(model.undulation_m(latitude_rad, longitude_rad)?)
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests;
