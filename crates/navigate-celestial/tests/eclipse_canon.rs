//! Navigate's Sun and Moon against every eclipse of the Five Millennium
//! Canon in the years this build covers.
//!
//! The test finds the instant of least gamma near the Canon's TD of greatest
//! eclipse and compares gamma and that instant with the Canon. TD is TT, so
//! the Delta T of the Canon does not enter.

use eclipse_canon::{lunar_eclipses, solar_eclipses};
use navigate_celestial::{Epoch, TimeScale, moon_ecef_tt, sun_ecef_tt};

const EARTH_RADIUS_M: f64 = 6_378_137.0;
const POLE: [f64; 3] = [0.0, 0.0, 1.0];

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn scale(a: [f64; 3], k: f64) -> [f64; 3] {
    a.map(|x| x * k)
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// The part of `a` at right angles to the unit vector `u`.
fn across(a: [f64; 3], u: [f64; 3]) -> [f64; 3] {
    sub(a, scale(u, dot(a, u)))
}

/// Signed gamma: the length of `miss` in Earth radii, positive where it
/// points to the north side of the plane at right angles to `axis`.
fn signed_gamma(miss: [f64; 3], axis: [f64; 3]) -> f64 {
    dot(miss, miss)
        .sqrt()
        .copysign(dot(miss, across(POLE, axis)))
        / EARTH_RADIUS_M
}

/// The apparent geocentric Sun and Moon in Earth-fixed axes, in metres.
fn sun_and_moon(jd_tt: f64) -> Option<([f64; 3], [f64; 3])> {
    let epoch = Epoch::from_jde_in_time_scale(jd_tt, TimeScale::TT);
    let sun = sun_ecef_tt(epoch, None).ok()?;
    let moon = moon_ecef_tt(epoch, None).ok()?;
    Some((
        scale(sun.direction, sun.distance_m),
        scale(moon.direction, moon.distance_m),
    ))
}

/// Gamma of a solar eclipse: the axis of the Moon's shadow runs from the
/// Sun through the Moon.
fn solar_gamma(jd_tt: f64) -> Option<f64> {
    let (sun, moon) = sun_and_moon(jd_tt)?;
    let axis = sub(sun, moon);
    let axis = scale(axis, 1.0 / dot(axis, axis).sqrt());
    Some(signed_gamma(across(moon, axis), axis))
}

/// Gamma of a lunar eclipse: the axis of the Earth's shadow runs from the
/// Sun through the Earth's centre.
fn lunar_gamma(jd_tt: f64) -> Option<f64> {
    let (sun, moon) = sun_and_moon(jd_tt)?;
    let axis = scale(sun, 1.0 / dot(sun, sun).sqrt());
    Some(signed_gamma(across(moon, axis), axis))
}

/// The instant of least |gamma| near `jd_tt` and gamma there. The square of
/// gamma is close to a parabola in time over a few minutes, because the
/// shadow axis moves nearly in a straight line.
fn least_gamma(gamma: fn(f64) -> Option<f64>, jd_tt: f64) -> Option<(f64, f64)> {
    let mut t = jd_tt;
    for step_s in [600.0, 60.0, 6.0] {
        let h = step_s / 86_400.0;
        let [a, b, c] = [t - h, t, t + h].map(|x| gamma(x).map(|g| g * g));
        let (a, b, c) = (a?, b?, c?);
        let curvature = a - 2.0 * b + c;
        if curvature <= 0.0 {
            return None;
        }
        t += h * (a - c) / (2.0 * curvature);
    }
    Some((t, gamma(t)?))
}

/// Bounds for an era: the years, then the largest gamma difference in Earth
/// radii and the largest timing difference in seconds, for solar and then
/// lunar eclipses. The timing differences grow before 1000 with the square of
/// the time from the present: DE431 and the Canon's lunar theory differ in
/// the tidal acceleration of the Moon by about 0.06 arcsecond per century
/// squared. The Canon's penumbral eclipses take greatest eclipse at the
/// greatest penumbral magnitude, up to about 15 s from the least gamma.
type Era = (i16, i16, f64, f64, f64, f64);

#[cfg(feature = "long-range")]
const ERAS: &[Era] = &[
    (-1999, -1000, 0.000_7, 100.0, 0.000_8, 110.0),
    (-1000, 0, 0.000_6, 60.0, 0.000_6, 65.0),
    (0, 1000, 0.000_4, 30.0, 0.000_45, 40.0),
    (1000, 2000, 0.000_25, 10.0, 0.000_3, 25.0),
    (2000, 3000, 0.000_25, 10.0, 0.000_3, 25.0),
];

#[cfg(not(feature = "long-range"))]
const ERAS: &[Era] = &[(1800, 2200, 0.001_5, 25.0, 0.001_5, 30.0)];

/// From 1900 to 2100 the Canon's ephemeris and the `long-range` series agree
/// closely, so the solar timing must agree to its rounding to the second.
const MODERN_YEARS: core::ops::Range<i16> = 1900..2100;
const MODERN_SOLAR_TIME_S: f64 = 1.5;

/// One eclipse in a debug build of every `STRIDE`, from every era: a debug
/// build is about 30 times slower. `ci.sh` runs every eclipse in a release
/// build.
const STRIDE: usize = if cfg!(debug_assertions) { 40 } else { 1 };

fn era(year: i16) -> Option<&'static Era> {
    ERAS.iter()
        .find(|&&(first, end, ..)| (first..end).contains(&year))
}

/// The number of eclipses compared, after checking each against its era's
/// bounds.
#[cfg(test)]
fn compare(
    eclipses: impl Iterator<Item = (i16, f64, f64)>,
    gamma: fn(f64) -> Option<f64>,
    solar: bool,
) -> usize {
    let mut compared = 0;
    for (year, canon_jd, canon_gamma) in eclipses.step_by(STRIDE) {
        let Some(&(.., solar_gamma, solar_time, lunar_gamma, lunar_time)) = era(year) else {
            continue;
        };
        let (gamma_bound, time_bound) = if solar {
            (solar_gamma, solar_time)
        } else {
            (lunar_gamma, lunar_time)
        };
        let (t, computed) = least_gamma(gamma, canon_jd)
            .unwrap_or_else(|| panic!("no least gamma near JD {canon_jd} TT"));
        let time_s = (t - canon_jd) * 86_400.0;
        assert!(
            (computed - canon_gamma).abs() < gamma_bound,
            "JD {canon_jd} TT: gamma {computed:.5} against the Canon's {canon_gamma:.4}"
        );
        assert!(
            time_s.abs() < time_bound,
            "JD {canon_jd} TT: least gamma {time_s:.1} s from the Canon's greatest eclipse"
        );
        if cfg!(feature = "long-range") && solar && MODERN_YEARS.contains(&year) {
            assert!(
                time_s.abs() < MODERN_SOLAR_TIME_S,
                "JD {canon_jd} TT: {time_s:.2} s"
            );
        }
        compared += 1;
    }
    compared
}

#[test]
fn solar_gamma_and_greatest_eclipse_match_the_canon() {
    let eclipses = solar_eclipses().map(|e| (e.date.year, e.greatest_eclipse_jd_tt(), e.gamma));
    let covered = eclipses
        .clone()
        .step_by(STRIDE)
        .filter(|e| era(e.0).is_some())
        .count();
    assert!(covered > 0);
    assert_eq!(compare(eclipses, solar_gamma, true), covered);
}

#[test]
fn lunar_gamma_and_greatest_eclipse_match_the_canon() {
    let eclipses = lunar_eclipses().map(|e| (e.date.year, e.greatest_eclipse_jd_tt(), e.gamma));
    let covered = eclipses
        .clone()
        .step_by(STRIDE)
        .filter(|e| era(e.0).is_some())
        .count();
    assert!(covered > 0);
    assert_eq!(compare(eclipses, lunar_gamma, false), covered);
}
