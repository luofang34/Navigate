#[cfg(not(feature = "long-range"))]
use crate::precession;
use crate::{Ephemeris, Epoch, FrameId, State, earth_rotation};
use nalgebra::Vector3;

mod earth;
#[cfg(feature = "long-range")]
mod long_range;
#[cfg(not(feature = "long-range"))]
mod moon;

const SUN: i32 = 10;
const MOON: i32 = 301;
const EARTH: i32 = 399;
const J2000: i32 = 1;
const ITRF93: i32 = FrameId::EARTH_ITRF93.orientation;
const ASTRONOMICAL_UNIT_M: f64 = 149_597_870_700.0;
const SECONDS_PER_CENTURY: f64 = 36_525.0 * 86_400.0;
/// The supported range in TDB Julian centuries from J2000: 1800 to 2200, or
/// -2000 to 3000 with the `long-range` feature.
#[cfg(not(feature = "long-range"))]
const CENTURIES: std::ops::RangeInclusive<f64> = -2.0..=2.0;
#[cfg(feature = "long-range")]
const CENTURIES: std::ops::RangeInclusive<f64> = -40.0..=10.0;
/// Half-width of the central difference for velocity. The position is the
/// mean of the two evaluations, which differs from the series by a h^2 / 2,
/// less than 0.1 m for the heliocentric acceleration a, so each state costs
/// two evaluations. Rounding in the series (the mean longitudes reach
/// thousands of radians) gives velocity errors of 4 mm/s at the ends of 1800
/// to 2200, and 0.08 m/s at -2000 with `long-range`; the aberration changes
/// by less than 0.1 milliarcsecond. A longer step would reduce them but move
/// the mean position further from the series.
const VELOCITY_HALF_STEP_S: f64 = 4.0;

/// A failed analytic state evaluation.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
pub enum AnalyticError {
    /// The provider has no model for this NAIF body.
    #[error("the analytic ephemeris has no model for NAIF body {0}")]
    UnsupportedBody(i32),
    /// The provider cannot convert a state with this origin or orientation.
    #[error(
        "the analytic ephemeris cannot convert a state with origin {} and orientation {}",
        .0.center,
        .0.orientation
    )]
    UnsupportedFrame(FrameId),
    /// The epoch is outside the range where the accuracy bounds apply: 1800
    /// to 2200 TDB, or -2000 to 3000 with the `long-range` feature.
    #[error("epoch {0} is outside the analytic ephemeris range")]
    EpochOutOfRange(Epoch),
}

/// Sun, Earth, and Moon states from truncated analytic series.
///
/// This provider needs no kernel files, reads no clock, and does no I/O.
/// By default the Earth comes from VSOP87D and the Moon from the ELP-2000/82
/// series in Meeus, chapter 47, and IAU 1976 precession rotates both to
/// J2000 axes. With the `long-range` feature the Earth comes from VSOP87B and
/// the Moon from ELP2000-82B, both fitted to JPL DE441 and in J2000 axes.
///
/// The origin is the centre of the Sun, not the solar system barycentre.
/// The two are up to 1.5e9 m apart, so only differences between states of
/// this provider are valid. The provider rejects a caller state with a
/// barycentric origin for this reason. The offset changes the Sun's velocity
/// by less than 20 m/s, which changes aberration by less than 0.02 arcsecond.
///
/// Through [`observe_body`](crate::observe_body) with
/// [`Correction::VacuumReception`](crate::Correction::VacuumReception) for a
/// geocentric observer, the tests bound the direction error relative to JPL
/// DE421 (1900 to 2050) and DE440 (1800 to 2200) at 1.5 arcseconds for the
/// Sun and 18 arcseconds for the Moon, and the Moon range error at 15 km. The
/// supported range is 1800 to 2200 TDB.
///
/// With `long-range`, the same tests bound the Sun at 0.3 arcsecond and the
/// Moon at 0.4 arcsecond and 250 m, and the range is -2000 to 3000 TDB.
/// Relative to DE441 the tests bound the Sun, the Moon, and the Moon-Sun
/// elongation by millennium: 0.25, 0.4, and 0.35 arcsecond from 2000 to
/// 3000, and 1.1, 5.6, and 6.5 arcseconds from -2000 to -1000.
///
/// Supported bodies are the Sun (10), the Earth (399), and the Moon (301).
/// Caller states must use J2000 axes (orientation 1) and one of these bodies
/// as origin, or [`FrameId::EARTH_ITRF93`]. The Earth-fixed axes come from
/// the model in [`sun_ecef`](crate::sun_ecef), not from ITRF93 data.
#[derive(Clone, Copy, Debug, Default)]
pub struct AnalyticEphemeris;

impl Ephemeris for AnalyticEphemeris {
    type Error = AnalyticError;

    fn barycentric_state(&self, state: State) -> Result<State, Self::Error> {
        let position = Vector3::from(state.position_m);
        let velocity = Vector3::from(state.velocity_m_s);
        let (position, velocity) = match (state.frame.center, state.frame.orientation) {
            (SUN | EARTH | MOON, J2000) => (position, velocity),
            (EARTH, ITRF93) => {
                let to_j2000 = earth_rotation::j2000_to_earth_fixed(state.epoch).transpose();
                let velocity = velocity + earth_rotation::rotation_velocity(position);
                (to_j2000 * position, to_j2000 * velocity)
            }
            _ => return Err(AnalyticError::UnsupportedFrame(state.frame)),
        };
        let origin = self.body_state(state.frame.center, state.epoch)?;
        Ok(State {
            position_m: (Vector3::from(origin.position_m) + position).into(),
            velocity_m_s: (Vector3::from(origin.velocity_m_s) + velocity).into(),
            epoch: state.epoch,
            frame: FrameId::SSB_J2000,
        })
    }

    fn body_state(&self, body: i32, epoch: Epoch) -> Result<State, Self::Error> {
        let t = epoch.to_tdb_centuries_since_j2000();
        if !CENTURIES.contains(&t) {
            return Err(AnalyticError::EpochOutOfRange(epoch));
        }
        let step = VELOCITY_HALF_STEP_S / SECONDS_PER_CENTURY;
        let (early, late) = (t - step, t + step);
        let (before, after) = (position(body, early)?, position(body, late)?);
        // The rounded times enter exactly: the difference of two close values
        // has no rounding error.
        let velocity = (after - before) / ((late - early) * SECONDS_PER_CENTURY);
        Ok(State {
            position_m: ((before + after) / 2.0).into(),
            velocity_m_s: velocity.into(),
            epoch,
            frame: FrameId::SSB_J2000,
        })
    }
}

/// Heliocentric position in J2000 axes, in metres.
#[cfg(feature = "long-range")]
fn position(body: i32, t: f64) -> Result<Vector3<f64>, AnalyticError> {
    match body {
        SUN => Ok(Vector3::zeros()),
        EARTH => Ok(long_range::earth(t)),
        MOON => Ok(long_range::earth(t) + long_range::moon(t)),
        _ => Err(AnalyticError::UnsupportedBody(body)),
    }
}

/// Heliocentric position in J2000 axes, in metres.
#[cfg(not(feature = "long-range"))]
fn position(body: i32, t: f64) -> Result<Vector3<f64>, AnalyticError> {
    match body {
        SUN => Ok(Vector3::zeros()),
        EARTH => Ok(earth(t)),
        MOON => {
            let (longitude, latitude, distance) = moon::geocentric(t);
            Ok(earth(t) + of_date_to_j2000(t, spherical(longitude, latitude, distance)))
        }
        _ => Err(AnalyticError::UnsupportedBody(body)),
    }
}

#[cfg(not(feature = "long-range"))]
fn earth(t: f64) -> Vector3<f64> {
    let (longitude, latitude, distance) = earth::heliocentric(t);
    of_date_to_j2000(
        t,
        spherical(longitude, latitude, distance * ASTRONOMICAL_UNIT_M),
    )
}

fn spherical(longitude: f64, latitude: f64, distance: f64) -> Vector3<f64> {
    let (sin_lon, cos_lon) = longitude.sin_cos();
    let (sin_lat, cos_lat) = latitude.sin_cos();
    Vector3::new(cos_lat * cos_lon, cos_lat * sin_lon, sin_lat) * distance
}

#[cfg(not(feature = "long-range"))]
/// Rotate mean ecliptic-of-date coordinates to J2000 equator axes.
fn of_date_to_j2000(t: f64, ecliptic: Vector3<f64>) -> Vector3<f64> {
    precession::j2000_to_mean_of_date(t).transpose() * precession::ecliptic_to_equator(t) * ecliptic
}

#[cfg(test)]
mod tests;
