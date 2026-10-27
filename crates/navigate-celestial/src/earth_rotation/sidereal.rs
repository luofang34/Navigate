//! Mean sidereal time from the Earth rotation angle and the equation of the
//! origins of the Vondrák et al. 2011 precession.
//!
//! GMST = ERA - EO. EO comes from the mean precession matrix and the CIO
//! locator s, as in ERFA `eraEors`. The CIO locator is the integral
//! s = -∫ (X dY/dt - Y dX/dt) / (1 + Z) dt from J2000 (IERS Conventions
//! 2010, chapter 5), with the 94 microarcsecond constant omitted. The pole
//! is the mean pole, so the nutation terms of s are omitted (less than 3
//! milliarcseconds). `eraEors` is from ERFA under the licence in
//! `LICENSE-ERFA`.

use crate::precession;
use nalgebra::Matrix3;

/// Gauss-Legendre nodes and weights on [0, 1].
const GAUSS_LEGENDRE: [[f64; 2]; 16] = [
    [0.005299532504175031, 0.013576229705877027],
    [0.0277124884633837, 0.031126761969323805],
    [0.06718439880608418, 0.0475792558412465],
    [0.12229779582249845, 0.06231448562776697],
    [0.19106187779867811, 0.07479799440828833],
    [0.2709916111713863, 0.08457825969750131],
    [0.35919822461037054, 0.0913017075224618],
    [0.4524937450811813, 0.09472530522753429],
    [0.5475062549188188, 0.09472530522753429],
    [0.6408017753896295, 0.0913017075224618],
    [0.7290083888286136, 0.08457825969750131],
    [0.8089381222013219, 0.07479799440828833],
    [0.8777022041775016, 0.06231448562776697],
    [0.9328156011939158, 0.0475792558412465],
    [0.9722875115366163, 0.031126761969323805],
    [0.994700467495825, 0.013576229705877027],
];

/// Greenwich mean sidereal time in radians. `ut1_days` is UT1 Julian days
/// from J2000 and `t` is TT Julian centuries from J2000.
pub(crate) fn gmst(ut1_days: f64, t: f64) -> f64 {
    let precession = precession::long_term_j2000_to_mean_of_date(t);
    (earth_rotation_angle(ut1_days) - equation_of_origins(&precession, cio_locator(t)))
        .rem_euclid(std::f64::consts::TAU)
}

/// IERS 2003 Earth rotation angle in radians (ERFA `eraEra00`).
pub(crate) fn earth_rotation_angle(ut1_days: f64) -> f64 {
    // The fraction of a day keeps the precision of the large day count.
    let turns =
        ut1_days.rem_euclid(1.0) + 0.779_057_273_264_0 + 0.002_737_811_911_354_48 * ut1_days;
    turns.rem_euclid(1.0) * std::f64::consts::TAU
}

/// CIO locator s of the mean pole, in radians.
pub(crate) fn cio_locator(t: f64) -> f64 {
    integrate(t, &GAUSS_LEGENDRE)
}

fn integrate(t: f64, rule: &[[f64; 2]]) -> f64 {
    -t * rule
        .iter()
        .map(|[node, weight]| {
            let tau = node * t;
            let pole = precession::equator_pole(tau);
            let (x_rate, y_rate) = precession::equator_pole_rate(tau);
            weight * (pole.x * y_rate - pole.y * x_rate) / (1.0 + pole.z)
        })
        .sum::<f64>()
}

/// Equation of the origins from a precession matrix and s (ERFA `eraEors`).
fn equation_of_origins(precession: &Matrix3<f64>, s: f64) -> f64 {
    let x = precession[(2, 0)];
    let a = x / (1.0 + precession[(2, 2)]);
    let (xs, ys, zs) = (1.0 - a * x, -a * precession[(2, 1)], -x);
    let p = precession[(0, 0)] * xs + precession[(0, 1)] * ys + precession[(0, 2)] * zs;
    let q = precession[(1, 0)] * xs + precession[(1, 1)] * ys + precession[(1, 2)] * zs;
    if p == 0.0 && q == 0.0 {
        s
    } else {
        s - q.atan2(p)
    }
}

#[cfg(test)]
mod tests;
