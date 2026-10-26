use crate::{Ephemeris, Epoch, FrameId, State, precession};
use nalgebra::Vector3;

mod earth;
mod moon;

const SUN: i32 = 10;
const MOON: i32 = 301;
const EARTH: i32 = 399;
const J2000: i32 = 1;
const ASTRONOMICAL_UNIT_M: f64 = 149_597_870_700.0;
const SECONDS_PER_CENTURY: f64 = 36_525.0 * 86_400.0;
/// Half-width of the central difference for velocity. The series are
/// smooth on this scale, and the rounding error stays below 1 µm/s.
const VELOCITY_STEP_S: f64 = 60.0;

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
    /// The epoch is outside the range where the accuracy bounds apply.
    #[error("epoch {0} is outside the analytic ephemeris range 1900 to 2100 TDB")]
    EpochOutOfRange(Epoch),
}

/// Sun, Earth, and Moon states from truncated analytic series.
///
/// This provider needs no kernel files, reads no clock, and does no I/O.
/// The Earth comes from VSOP87D and the Moon from the ELP-2000/82 series in
/// Meeus, chapter 47. IAU 1976 precession rotates both to J2000 axes.
///
/// The origin is the centre of the Sun, not the solar system barycentre.
/// The two are up to 1.5e9 m apart, so only differences between states of
/// this provider are valid. The provider rejects a caller state with a
/// barycentric origin for this reason. The offset changes the Sun's velocity
/// by less than 20 m/s, which changes aberration by less than 0.02 arcsecond.
///
/// From 1900 to 2050, through [`observe_body`](crate::observe_body) with
/// [`Correction::VacuumReception`](crate::Correction::VacuumReception) for a
/// geocentric observer, the tests bound the direction error relative to JPL
/// DE421 at 1.5 arcseconds for the Sun and 18 arcseconds for the Moon, and
/// the Moon range error at 15 km. The supported range is 1900 to 2100 TDB.
///
/// Supported bodies are the Sun (10), the Earth (399), and the Moon (301).
/// Caller states must use J2000 axes (orientation 1) and one of these bodies
/// as origin.
#[derive(Clone, Copy, Debug, Default)]
pub struct AnalyticEphemeris;

impl Ephemeris for AnalyticEphemeris {
    type Error = AnalyticError;

    fn barycentric_state(&self, state: State) -> Result<State, Self::Error> {
        if state.frame.orientation != J2000 || !matches!(state.frame.center, SUN | EARTH | MOON) {
            return Err(AnalyticError::UnsupportedFrame(state.frame));
        }
        let origin = self.body_state(state.frame.center, state.epoch)?;
        Ok(State {
            position_m: std::array::from_fn(|i| origin.position_m[i] + state.position_m[i]),
            velocity_m_s: std::array::from_fn(|i| origin.velocity_m_s[i] + state.velocity_m_s[i]),
            epoch: state.epoch,
            frame: FrameId::SSB_J2000,
        })
    }

    fn body_state(&self, body: i32, epoch: Epoch) -> Result<State, Self::Error> {
        let t = epoch.to_tdb_centuries_since_j2000();
        if !(-1.0..=1.0).contains(&t) {
            return Err(AnalyticError::EpochOutOfRange(epoch));
        }
        let step = VELOCITY_STEP_S / SECONDS_PER_CENTURY;
        let velocity =
            (position(body, t + step)? - position(body, t - step)?) / (2.0 * VELOCITY_STEP_S);
        Ok(State {
            position_m: position(body, t)?.into(),
            velocity_m_s: velocity.into(),
            epoch,
            frame: FrameId::SSB_J2000,
        })
    }
}

/// Heliocentric position in J2000 axes, in metres.
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

/// Rotate mean ecliptic-of-date coordinates to J2000 equator axes.
fn of_date_to_j2000(t: f64, ecliptic: Vector3<f64>) -> Vector3<f64> {
    precession::j2000_to_mean_of_date(t).transpose() * precession::ecliptic_to_equator(t) * ecliptic
}

#[cfg(test)]
mod tests;
