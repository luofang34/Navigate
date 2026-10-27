//! J2000 to Earth-fixed axes from precession, nutation, and sidereal time.
//!
//! The model applies the Vondrák et al. 2011 long-term precession, the four
//! largest IAU 1980 nutation terms (Meeus, chapter 22: 0.5 arcsecond in
//! longitude and 0.1 arcsecond in obliquity), and GAST from a GMST that the
//! same precession gives (see [`sidereal`]). The epochs are
//! TT, and UT1 is TT - Delta T from [`delta_t`](fn@crate::delta_t). It omits
//! polar motion, which is less than 0.6 arcsecond.

use crate::{
    Epoch, delta_t, precession,
    precession::{frame_x, frame_z},
};
use nalgebra::{Matrix3, Vector3};

mod sidereal;

const ARCSECOND: f64 = std::f64::consts::PI / 648_000.0;
/// Earth rotation rate relative to the precessing equinox, in rad/s.
pub(crate) const ROTATION_RATE_RAD_S: f64 = 7.292_115_146_706_979e-5;

/// Rotate J2000 axes to Earth-fixed axes at the TT epoch `epoch`.
pub(crate) fn j2000_to_earth_fixed(epoch: Epoch) -> Matrix3<f64> {
    let t = epoch.to_tdb_centuries_since_j2000();
    let (longitude, obliquity) = nutation(t);
    let mean_obliquity = precession::long_term_mean_obliquity(t);
    let nutation =
        frame_x(-(mean_obliquity + obliquity)) * frame_z(-longitude) * frame_x(mean_obliquity);
    let ut1_days = delta_t::ut1_julian_days(epoch) - 2_451_545.0;
    let sidereal = sidereal::gmst(ut1_days, t) + longitude * mean_obliquity.cos();
    frame_z(sidereal) * nutation * precession::long_term_j2000_to_mean_of_date(t)
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
