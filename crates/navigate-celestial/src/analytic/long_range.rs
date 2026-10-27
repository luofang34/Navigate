//! Earth and Moon series for -2000 to 3000, with the `long-range` feature.
//!
//! The Earth is VSOP87B (Bretagnon and Francou 1988), heliocentric in the
//! dynamical ecliptic and equinox of J2000. The Moon is ELP2000-82B
//! (Chapront-Touzé and Chapront 1988) with the tidal acceleration of the NASA
//! Five Millennium Canon. Both are in J2000 ecliptic axes, so no precession of
//! date is necessary. `fixtures/long_range_tables.py` writes the tables and
//! describes the truncation.

use super::{ASTRONOMICAL_UNIT_M, earth as vsop87};
use crate::precession::frame_x;
use nalgebra::Vector3;

mod earth_table;
mod moon;

use vsop87::Term;

/// Obliquity of the VSOP87 J2000 ecliptic to the J2000 equator.
const OBLIQUITY_J2000_RAD: f64 = 84_381.411_36 * std::f64::consts::PI / 648_000.0;

/// Heliocentric Earth in J2000 equator axes, in metres. `t` is TDB Julian
/// centuries from J2000.
pub(super) fn earth(t: f64) -> Vector3<f64> {
    let tau = t / 10.0;
    let (longitude, latitude, distance) = (
        vsop87::sum(&earth_table::L, tau),
        vsop87::sum(&earth_table::B, tau),
        vsop87::sum(&earth_table::R, tau),
    );
    to_equator(super::spherical(
        longitude,
        latitude,
        distance * ASTRONOMICAL_UNIT_M,
    ))
}

/// Geocentric Moon in J2000 equator axes, in metres.
pub(super) fn moon(t: f64) -> Vector3<f64> {
    to_equator(moon::geocentric_ecliptic(t))
}

fn to_equator(ecliptic: Vector3<f64>) -> Vector3<f64> {
    frame_x(-OBLIQUITY_J2000_RAD) * ecliptic
}
