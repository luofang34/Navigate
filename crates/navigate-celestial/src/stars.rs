//! Bright stars from the Yale Bright Star Catalogue, 5th revised edition.
//!
//! The source is CDS catalogue V/50 (Hoffleit and Warren 1991), prepared by
//! the NASA Astronomical Data Center. It is a United States government work
//! and is in the public domain. The 9096 entries with a J2000 position are
//! embedded as `data/bsc5.bin`, which the `star_catalog` example writes, with
//! the trigonometric parallax and the radial velocity of each star.

use crate::{
    AnalyticEphemeris, AnalyticError, Ephemeris, Epoch, civil_time, earth_rotation, observation,
};
use nalgebra::Vector3;

const CATALOG: &[u8] = include_bytes!("../data/bsc5.bin");
const RECORD_BYTES: usize = 22;
const LIGHT_SPEED_M_S: f64 = 299_792_458.0;
const MAS: f64 = std::f64::consts::PI / 648_000_000.0;
/// Parsecs per year at 1 km/s.
const PARSEC_PER_YEAR_PER_KM_S: f64 = 1.022_712_165_053_707_7e-6;

/// A catalogue star as seen from the Earth at one epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarSight {
    /// Harvard Revised (HR) catalogue number.
    pub hr: u16,
    /// Unit direction in ECEF axes, with the same axes as
    /// [`EarthFixedSight`](crate::EarthFixedSight).
    pub direction: [f64; 3],
    /// Visual magnitude in the Johnson V band.
    pub v_magnitude: f64,
    /// B-V colour index, if the catalogue gives one.
    pub b_minus_v: Option<f64>,
}

/// The catalogue stars at least as bright as `faintest_v_magnitude`.
///
/// `epoch` is a civil clock reading, as for [`sun_ecef`](crate::sun_ecef).
///
/// Each star moves on a straight line in space from its J2000 position, with
/// its proper motion, and with its radial velocity where the catalogue gives
/// a trigonometric parallax. Then annual aberration from the Earth's velocity
/// applies, and the Earth rotation model of [`sun_ecef`](crate::sun_ecef)
/// turns it to ECEF axes. Annual parallax (less than 0.8 arcsecond), diurnal
/// aberration (0.3 arcsecond), gravitational deflection, and the orbits of
/// binary stars are omitted. From 1975 to 2026 the tests bound the error
/// relative to Skyfield with IERS UT1 at 3 arcseconds, and the space motion
/// matches Skyfield's to 0.002 arcsecond from -2000 to 3000.
///
/// Far from J2000 the catalogue proper motions limit the accuracy. Compared
/// with the Hipparcos catalogue propagated the same way, the stars to
/// magnitude 6.5 differ by a median of 1.4 arcseconds in 1700, 5 in 1000, 13
/// in -330, and 23 in -2000; 99 percent are within 9, 34, 81, and 141
/// arcseconds. Hipparcos is not truth for long spans either: its proper
/// motions come from three years of observations. Close binaries, such as
/// Alpha Centauri, can differ by several arcminutes in the past.
///
/// # Errors
/// Returns an error for an epoch outside the range of
/// [`AnalyticEphemeris`]: 1800 to 2200 TDB, or -2000 to 3000 with the
/// `long-range` feature.
pub fn stars_ecef(
    epoch: Epoch,
    faintest_v_magnitude: f64,
) -> Result<Vec<StarSight>, AnalyticError> {
    stars_ecef_tt(civil_time(epoch).terrestrial, faintest_v_magnitude)
}

/// The catalogue stars at the instant `epoch`, without a civil reading.
///
/// `epoch` is the instant itself, in any time scale, as for
/// [`sun_ecef_tt`](crate::sun_ecef_tt). The model, bounds, and range are the
/// same as for [`stars_ecef`].
///
/// # Errors
/// Returns an error for an epoch outside the range of
/// [`AnalyticEphemeris`].
pub fn stars_ecef_tt(
    epoch: Epoch,
    faintest_v_magnitude: f64,
) -> Result<Vec<StarSight>, AnalyticError> {
    // The ephemeris range check comes first.
    let beta =
        Vector3::from(AnalyticEphemeris.body_state(399, epoch)?.velocity_m_s) / LIGHT_SPEED_M_S;
    let to_earth_fixed = earth_rotation::j2000_to_earth_fixed(epoch);
    let years = epoch.to_tdb_centuries_since_j2000() * 100.0;
    Ok(CATALOG
        .as_chunks::<RECORD_BYTES>()
        .0
        .iter()
        .map(decode)
        .filter(|star| star.v_magnitude <= faintest_v_magnitude)
        .map(|star| {
            let apparent = observation::aberrate(space_motion(&star, years), beta);
            StarSight {
                hr: star.hr,
                direction: (to_earth_fixed * apparent).normalize().into(),
                v_magnitude: star.v_magnitude,
                b_minus_v: star.b_minus_v,
            }
        })
        .collect())
}

/// The J2000 direction of a star `years` Julian years after J2000.
///
/// The star moves on a straight line in space. The light-time factor
/// 1 / (1 - v_r / c) turns the observed proper motion and radial velocity
/// into space motion (Stumpff 1985; SOFA `eraStarpv`). Without a measured
/// parallax the radial term is unknown and omitted, which leaves the proper
/// motion along a straight line on the tangent plane.
fn space_motion(star: &Record, years: f64) -> Vector3<f64> {
    let (sin_ra, cos_ra) = star.ra.sin_cos();
    let (sin_dec, cos_dec) = star.dec.sin_cos();
    let position = Vector3::new(cos_dec * cos_ra, cos_dec * sin_ra, sin_dec);
    let east = Vector3::new(-sin_ra, cos_ra, 0.0);
    let north = Vector3::new(-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec);
    let (light_time, radial_rate) = if star.parallax_mas > 0.0 {
        (
            1.0 / (1.0 - star.radial_velocity_km_s * 1000.0 / LIGHT_SPEED_M_S),
            // Radial velocity over distance, per year.
            star.radial_velocity_km_s * star.parallax_mas / 1000.0 * PARSEC_PER_YEAR_PER_KM_S,
        )
    } else {
        (1.0, 0.0)
    };
    let tangential = east * star.pm_ra + north * star.pm_dec;
    (position * (1.0 + light_time * radial_rate * years) + tangential * (light_time * years))
        .normalize()
}

struct Record {
    hr: u16,
    ra: f64,
    dec: f64,
    pm_ra: f64,
    pm_dec: f64,
    v_magnitude: f64,
    b_minus_v: Option<f64>,
    parallax_mas: f64,
    radial_velocity_km_s: f64,
}

/// Decode one record. The layout is in the `star_catalog` example.
fn decode(record: &[u8; RECORD_BYTES]) -> Record {
    let &[
        h0,
        h1,
        r0,
        r1,
        r2,
        r3,
        d0,
        d1,
        d2,
        d3,
        a0,
        a1,
        p0,
        p1,
        v0,
        v1,
        c0,
        c1,
        x0,
        x1,
        w0,
        w1,
    ] = record;
    let colour = i16::from_le_bytes([c0, c1]);
    Record {
        hr: u16::from_le_bytes([h0, h1]),
        ra: f64::from(u32::from_le_bytes([r0, r1, r2, r3])) / 4_294_967_296.0
            * std::f64::consts::TAU,
        dec: f64::from(i32::from_le_bytes([d0, d1, d2, d3])) / f64::from(i32::MAX)
            * std::f64::consts::FRAC_PI_2,
        pm_ra: f64::from(i16::from_le_bytes([a0, a1])) * MAS,
        pm_dec: f64::from(i16::from_le_bytes([p0, p1])) * MAS,
        v_magnitude: f64::from(i16::from_le_bytes([v0, v1])) / 100.0,
        b_minus_v: (colour != i16::MIN).then(|| f64::from(colour) / 100.0),
        parallax_mas: f64::from(u16::from_le_bytes([x0, x1])),
        radial_velocity_km_s: f64::from(i16::from_le_bytes([w0, w1])),
    }
}

#[cfg(test)]
mod tests;
