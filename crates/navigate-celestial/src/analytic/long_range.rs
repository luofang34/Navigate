//! Earth and Moon series for -2000 to 3000, with the `long-range` feature.
//!
//! The Earth is VSOP87B (Bretagnon and Francou 1988), heliocentric in the
//! dynamical ecliptic and equinox of J2000. The Moon is ELP2000-82B
//! (Chapront-Touzé and Chapront 1988). Both are in J2000 ecliptic axes, so no
//! precession of date is necessary. `examples/long_range_tables.rs` writes the
//! tables and describes the truncation.
//!
//! Both series are fitted to JPL DE431 (`examples/long_range_fit.rs`): cubic
//! corrections to the mean longitudes of the Moon, its perigee and node, and
//! the Sun, and to the VSOP87B longitude. DE431 is the long version of
//! DE430, the lunar ephemeris that the Delta T of Stephenson, Morrison and
//! Hohenkerk pairs with. JPL DE441 is not: before 500 its Moon departs from
//! DE431 by 65 arcseconds at -762 and 200 at -2000, which would move an
//! ancient eclipse by minutes. The fit gives the tidal acceleration of the
//! Moon, -25.80 arcseconds per century squared. The largest errors against
//! the DE431 fixture, in arcseconds and metres:
//!
//! | Millennium | Sun | Moon | Moon-Sun elongation | Moon range |
//! |---|---|---|---|---|
//! | -2000 | 0.88 | 4.3 | 5.0 | 790 |
//! | -1000 | 0.37 | 2.2 | 2.5 | 400 |
//! | 0 | 0.30 | 0.92 | 1.1 | 140 |
//! | 1000 | 0.25 | 0.29 | 0.31 | 130 |
//! | 2000 | 0.16 | 0.37 | 0.29 | 140 |

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
    let correction: f64 = earth_table::LONGITUDE_CORRECTION
        .iter()
        .rev()
        .fold(0.0, |total, coefficient| total * t + coefficient);
    let (longitude, latitude, distance) = (
        vsop87::sum(&earth_table::L, tau) + correction,
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
