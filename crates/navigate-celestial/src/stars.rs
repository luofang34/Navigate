//! Bright stars from the Yale Bright Star Catalogue, 5th revised edition.
//!
//! The source is CDS catalogue V/50 (Hoffleit and Warren 1991, NASA ADC).
//! Its ReadMe gives no licence or use restriction, and NASA ADC and CDS
//! distribute it freely. The 9096 entries with a J2000 position are embedded
//! as `data/bsc5.bin`, which the `star_catalog` example writes.

use crate::{
    AnalyticEphemeris, AnalyticError, Ephemeris, Epoch, civil_time, earth_rotation, observation,
};
use nalgebra::Vector3;

const CATALOG: &[u8] = include_bytes!("../data/bsc5.bin");
const RECORD_BYTES: usize = 18;
const LIGHT_SPEED_M_S: f64 = 299_792_458.0;
const MAS: f64 = std::f64::consts::PI / 648_000_000.0;

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
/// Each J2000 position moves by its linear proper motion, then annual
/// aberration from the Earth's velocity applies, then the Earth rotation
/// model of [`sun_ecef`](crate::sun_ecef) turns it to ECEF axes. Parallax
/// (less than 0.8 arcsecond), radial velocity, diurnal aberration (0.3
/// arcsecond), and gravitational deflection are omitted. The tests bound
/// the error relative to Skyfield with IERS UT1 at 3 arcseconds.
///
/// # Errors
/// Returns an error for an epoch outside 1850 to 2100 TDB.
pub fn stars_ecef(
    epoch: Epoch,
    faintest_v_magnitude: f64,
) -> Result<Vec<StarSight>, AnalyticError> {
    let epoch = civil_time(epoch).terrestrial;
    let to_earth_fixed = earth_rotation::j2000_to_earth_fixed(epoch);
    let beta =
        Vector3::from(AnalyticEphemeris.body_state(399, epoch)?.velocity_m_s) / LIGHT_SPEED_M_S;
    let years = epoch.to_tdb_centuries_since_j2000() * 100.0;
    Ok(CATALOG
        .as_chunks::<RECORD_BYTES>()
        .0
        .iter()
        .map(decode)
        .filter(|star| star.v_magnitude <= faintest_v_magnitude)
        .map(|star| {
            let (sin_ra, cos_ra) = star.ra.sin_cos();
            let (sin_dec, cos_dec) = star.dec.sin_cos();
            let position = Vector3::new(cos_dec * cos_ra, cos_dec * sin_ra, sin_dec);
            let east = Vector3::new(-sin_ra, cos_ra, 0.0);
            let north = Vector3::new(-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec);
            let moved = (position + (east * star.pm_ra + north * star.pm_dec) * years).normalize();
            let apparent = observation::aberrate(moved, beta);
            StarSight {
                hr: star.hr,
                direction: (to_earth_fixed * apparent).normalize().into(),
                v_magnitude: star.v_magnitude,
                b_minus_v: star.b_minus_v,
            }
        })
        .collect())
}

struct Record {
    hr: u16,
    ra: f64,
    dec: f64,
    pm_ra: f64,
    pm_dec: f64,
    v_magnitude: f64,
    b_minus_v: Option<f64>,
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
    }
}

#[cfg(test)]
mod tests;
