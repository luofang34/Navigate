//! ELP2000-82B evaluation, after the IMCCE Fortran subroutine ELP82B_2.

use nalgebra::Vector3;

mod distance;
mod latitude;
mod longitude;

const ARCSECOND: f64 = std::f64::consts::PI / 648_000.0;
/// The ELP distance scale: the mean distance of DE200 over that of the theory.
const DISTANCE_SCALE: f64 = 384_747.980_644_895_4 / 384_747.980_674_316_5;
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

/// Geocentric Moon in the J2000 ecliptic, in metres.
pub(super) fn geocentric_ecliptic(t: f64) -> Vector3<f64> {
    let powers = [1.0, t, t * t, t * t * t, t * t * t * t];
    let polynomial = |coefficients: &[f64; 5]| {
        coefficients
            .iter()
            .zip(powers)
            .map(|(c, p)| c * p)
            .sum::<f64>()
    };
    let sum = |main: &[(f64, [f64; 5])], perturbation: &[(f64, i32, f64, f64)]| {
        main.iter()
            .map(|(amplitude, phase)| amplitude * polynomial(phase).sin())
            .sum::<f64>()
            + perturbation
                .iter()
                .map(|(amplitude, power, phase, rate)| {
                    amplitude * t.powi(*power) * (phase + rate * t).sin()
                })
                .sum::<f64>()
    };
    let longitude = sum(&longitude::MAIN, &longitude::PERTURBATION) * ARCSECOND
        + polynomial(&longitude::MEAN_LONGITUDE);
    let latitude = sum(&latitude::MAIN, &latitude::PERTURBATION) * ARCSECOND;
    let distance_m = sum(&distance::MAIN, &distance::PERTURBATION) * DISTANCE_SCALE * 1000.0;
    let of_date = super::super::spherical(longitude, latitude, distance_m);

    let p = polynomial(&P) * t;
    let q = polynomial(&Q) * t;
    let scale = 2.0 * (1.0 - p * p - q * q).sqrt();
    let (pq, p2, q2) = (2.0 * p * q, 1.0 - 2.0 * p * p, 1.0 - 2.0 * q * q);
    let (p, q) = (p * scale, q * scale);
    Vector3::new(
        p2 * of_date.x + pq * of_date.y + p * of_date.z,
        pq * of_date.x + q2 * of_date.y - q * of_date.z,
        -p * of_date.x + q * of_date.y + (p2 + q2 - 1.0) * of_date.z,
    )
}
