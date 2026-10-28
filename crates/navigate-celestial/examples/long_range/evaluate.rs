//! The long-range series as the crate evaluates them, with tables that the
//! fit can change.

use super::theory::{ARCSECONDS_PER_RADIAN, MoonTerms, Vsop};

/// Obliquity of the VSOP87 J2000 ecliptic to the J2000 equator.
const OBLIQUITY: f64 = 84_381.411_36 / ARCSECONDS_PER_RADIAN;
/// Laskar precession from the ecliptic of date to the J2000 ecliptic.
const P: [f64; 5] = [
    0.101_803_91e-4,
    0.470_204_39e-6,
    -0.541_736_7e-9,
    -0.250_794_8e-11,
    0.463_486e-14,
];
const Q: [f64; 5] = [
    -0.113_469_002e-3,
    0.123_726_74e-6,
    0.126_541_7e-8,
    -0.137_180_8e-11,
    -0.320_334e-14,
];
const DISTANCE_SCALE: f64 = 384_747.980_644_895_4 / 384_747.980_674_316_5;

fn polynomial(coefficients: &[f64], t: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0, |total, coefficient| total * t + coefficient)
}

/// Rotate J2000 equator axes to the J2000 ecliptic.
pub fn ecliptic([x, y, z]: [f64; 3]) -> [f64; 3] {
    let (sin, cos) = OBLIQUITY.sin_cos();
    [x, cos * y + sin * z, -sin * y + cos * z]
}

/// Longitude, latitude, and distance of a vector.
pub fn spherical([x, y, z]: [f64; 3]) -> [f64; 3] {
    let r = (x * x + y * y + z * z).sqrt();
    [y.atan2(x), (z / r).asin(), r]
}

/// Geocentric Moon in the J2000 ecliptic, in metres.
pub fn moon(t: f64, terms: &MoonTerms, mean_longitude: &[f64; 5]) -> [f64; 3] {
    let sum = |variable: usize| {
        terms.main[variable]
            .iter()
            .map(|(amplitude, phase)| amplitude * polynomial(phase, t).sin())
            .sum::<f64>()
            + terms.perturbation[variable]
                .iter()
                .map(|(amplitude, power, phase)| {
                    amplitude * t.powi(*power) * polynomial(phase, t).sin()
                })
                .sum::<f64>()
    };
    let longitude = sum(0) / ARCSECONDS_PER_RADIAN + polynomial(mean_longitude, t);
    let latitude = sum(1) / ARCSECONDS_PER_RADIAN;
    let distance = sum(2) * DISTANCE_SCALE * 1000.0;
    let (x, y, z) = (
        latitude.cos() * longitude.cos() * distance,
        latitude.cos() * longitude.sin() * distance,
        latitude.sin() * distance,
    );
    let p = polynomial(&P, t) * t;
    let q = polynomial(&Q, t) * t;
    let scale = 2.0 * (1.0 - p * p - q * q).sqrt();
    let (pq, p2, q2) = (2.0 * p * q, 1.0 - 2.0 * p * p, 1.0 - 2.0 * q * q);
    let (p, q) = (p * scale, q * scale);
    [
        p2 * x + pq * y + p * z,
        pq * x + q2 * y - q * z,
        -p * x + q * y + (p2 + q2 - 1.0) * z,
    ]
}

/// Geocentric longitude of the Sun from VSOP87B, before the correction.
pub fn sun_longitude(t: f64, series: &Vsop) -> f64 {
    let tau = t / 10.0;
    series
        .range((1, 0)..(2, 0))
        .rev()
        .fold(0.0, |total, (_, terms)| {
            total * tau
                + terms
                    .iter()
                    .map(|(amplitude, phase, frequency)| {
                        amplitude * (phase + frequency * tau).cos()
                    })
                    .sum::<f64>()
        })
        + std::f64::consts::PI
}
