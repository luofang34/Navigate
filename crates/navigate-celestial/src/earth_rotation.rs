//! J2000 to Earth-fixed axes from precession, nutation, and sidereal time.
//!
//! The model applies IAU 1976 precession, the four largest IAU 1980
//! nutation terms (Meeus, chapter 22: 0.5 arcsecond in longitude and 0.1
//! arcsecond in obliquity), and GAST from the IAU 1982 GMST. It takes UT1
//! as equal to UTC, which rotates the axes by up to 0.9 s × 15 arcseconds/s
//! (13.5 arcseconds) about the pole. It omits polar motion, which is less
//! than 0.6 arcsecond. Leap seconds come from hifitime.
//!
//! Leap seconds keep UT1 - UTC within 0.9 s from 1972. Before 1972, UTC is
//! not that close to UT1, so the model rejects those epochs. After the last
//! leap second that hifitime knows, the bound holds only while the IERS keeps
//! UT1 - UTC within 0.9 s.

use crate::{
    AnalyticError, Epoch, precession,
    precession::{frame_x, frame_z},
};
use nalgebra::{Matrix3, Vector3};

const ARCSECOND: f64 = std::f64::consts::PI / 648_000.0;
/// Earth rotation rate relative to the precessing equinox, in rad/s.
pub(crate) const ROTATION_RATE_RAD_S: f64 = 7.292_115_146_706_979e-5;

/// Rotate J2000 axes to Earth-fixed axes at `epoch`.
pub(crate) fn j2000_to_earth_fixed(epoch: Epoch) -> Result<Matrix3<f64>, AnalyticError> {
    if epoch.to_tai_seconds() < Epoch::from_gregorian_utc_at_midnight(1972, 1, 1).to_tai_seconds() {
        return Err(AnalyticError::EarthRotationOutOfRange(epoch));
    }
    let t = epoch.to_tdb_centuries_since_j2000();
    let (longitude, obliquity) = nutation(t);
    let mean_obliquity = precession::mean_obliquity(t);
    let nutation =
        frame_x(-(mean_obliquity + obliquity)) * frame_z(-longitude) * frame_x(mean_obliquity);
    let sidereal = gmst(epoch) + longitude * mean_obliquity.cos();
    Ok(frame_z(sidereal) * nutation * precession::j2000_to_mean_of_date(t))
}

/// Velocity in J2000 axes of a point fixed on the rotating Earth.
pub(crate) fn rotation_velocity(earth_fixed_m: Vector3<f64>) -> Vector3<f64> {
    Vector3::new(0.0, 0.0, ROTATION_RATE_RAD_S).cross(&earth_fixed_m)
}

/// Nutation in longitude and obliquity, in radians.
fn nutation(t: f64) -> (f64, f64) {
    let node = (125.044_52 - 1_934.136_261 * t).to_radians();
    let sun = (280.466_5 + 36_000.769_8 * t).to_radians();
    let moon = (218.316_5 + 481_267.881_3 * t).to_radians();
    let longitude = -17.20 * node.sin() - 1.32 * (2.0 * sun).sin() - 0.23 * (2.0 * moon).sin()
        + 0.21 * (2.0 * node).sin();
    let obliquity = 9.20 * node.cos() + 0.57 * (2.0 * sun).cos() + 0.10 * (2.0 * moon).cos()
        - 0.09 * (2.0 * node).cos();
    (longitude * ARCSECOND, obliquity * ARCSECOND)
}

/// Greenwich mean sidereal time in radians (Meeus 12.4), with UT1 = UTC.
fn gmst(epoch: Epoch) -> f64 {
    let days = epoch.to_jde_utc_days() - 2_451_545.0;
    let t = days / 36_525.0;
    let degrees =
        280.460_618_37 + 360.985_647_366_29 * days + t * t * (0.000_387_933 - t / 38_710_000.0);
    degrees.rem_euclid(360.0).to_radians()
}
