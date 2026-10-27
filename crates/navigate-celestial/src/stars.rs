//! Bright stars in Earth-fixed axes.
//!
//! The catalogue and the space motion come from the `flamsteed` crate, which
//! embeds the Yale Bright Star Catalogue (public domain). This module adds
//! aberration and the Earth rotation.

use crate::{
    AnalyticEphemeris, AnalyticError, Ephemeris, Epoch, civil_time, earth_rotation, observation,
};
use flamsteed::Catalog;
use nalgebra::Vector3;

const LIGHT_SPEED_M_S: f64 = 299_792_458.0;

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
/// Each star moves in space as [`flamsteed::Star::direction_at`] gives,
/// which also states the catalogue accuracy far from J2000. Then annual
/// aberration from the Earth's velocity applies, and the Earth rotation model
/// of [`sun_ecef`](crate::sun_ecef) turns it to ECEF axes. Annual parallax
/// (less than 0.8 arcsecond), diurnal aberration (0.3 arcsecond), and
/// gravitational deflection are omitted. From 1975 to 2026 the tests bound
/// the error relative to Skyfield with IERS UT1 at 3 arcseconds.
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
    Ok(Catalog::bright_star_catalogue()
        .stars()
        .filter(|star| star.v_magnitude <= faintest_v_magnitude)
        .map(|star| {
            let apparent = observation::aberrate(Vector3::from(star.direction_at(years)), beta);
            StarSight {
                hr: star.hr,
                direction: (to_earth_fixed * apparent).normalize().into(),
                v_magnitude: star.v_magnitude,
                b_minus_v: star.b_minus_v,
            }
        })
        .collect())
}

#[cfg(test)]
mod tests;
