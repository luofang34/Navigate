//! Precession and mean obliquity, in two roles. `t` is TDB Julian centuries
//! from J2000.
//!
//! IAU 1976 precession (Lieske et al. 1977) decodes the analytic series:
//! VSOP87D and the Meeus Moon give coordinates of date in the frame of that
//! precession era, so their inverse rotation uses it. Its precession rate
//! differs from IAU 2006 by about 0.3 arcsecond per century.
//!
//! The Vondrák et al. 2011 long-term precession turns J2000 axes to the
//! Earth's equator of date. It agrees with IAU 2006 to 0.1 milliarcsecond in
//! the 20th and 21st centuries and holds to a few arcseconds over the
//! historical period. Its coefficients and matrix come from ERFA
//! `eraLtpequ`, `eraLtpecl`, and `eraLtp`, Copyright (C) 2013-2023, NumFOCUS
//! Foundation, derived with permission from the IAU SOFA library, under the
//! licence in `LICENSE-ERFA`.

use nalgebra::{Matrix3, Vector3};

const ARCSECOND: f64 = std::f64::consts::PI / 648_000.0;

/// Mean obliquity of the ecliptic of date, in radians.
pub(crate) fn mean_obliquity(t: f64) -> f64 {
    (84_381.448 + t * (-46.815 + t * (-0.000_59 + t * 0.001_813))) * ARCSECOND
}

/// Rotate mean ecliptic-of-date axes to mean equator-of-date axes.
pub(crate) fn ecliptic_to_equator(t: f64) -> Matrix3<f64> {
    frame_x(-mean_obliquity(t))
}

/// Rotate J2000 mean equator axes to mean equator-of-date axes.
pub(crate) fn j2000_to_mean_of_date(t: f64) -> Matrix3<f64> {
    let zeta = t * (2_306.218_1 + t * (0.301_88 + t * 0.017_998)) * ARCSECOND;
    let z = t * (2_306.218_1 + t * (1.094_68 + t * 0.018_203)) * ARCSECOND;
    let theta = t * (2_004.310_9 + t * (-0.426_65 - t * 0.041_833)) * ARCSECOND;
    frame_z(-z) * frame_y(theta) * frame_z(-zeta)
}

/// A rotation of the coordinate axes by `angle` about x.
pub(crate) fn frame_x(angle: f64) -> Matrix3<f64> {
    let (sin, cos) = angle.sin_cos();
    Matrix3::new(1.0, 0.0, 0.0, 0.0, cos, sin, 0.0, -sin, cos)
}

/// A rotation of the coordinate axes by `angle` about z.
pub(crate) fn frame_z(angle: f64) -> Matrix3<f64> {
    let (sin, cos) = angle.sin_cos();
    Matrix3::new(cos, sin, 0.0, -sin, cos, 0.0, 0.0, 0.0, 1.0)
}

fn frame_y(angle: f64) -> Matrix3<f64> {
    let (sin, cos) = angle.sin_cos();
    Matrix3::new(cos, 0.0, -sin, 0.0, 1.0, 0.0, sin, 0.0, cos)
}

/// Vondrák et al. 2011 equator pole of date in J2000 axes (ERFA `eraLtpequ`).
pub(crate) fn equator_pole(t: f64) -> Vector3<f64> {
    let (x, y) = series(t, &EQUATOR_POLYNOMIAL, &EQUATOR_PERIODIC);
    Vector3::new(x, y, (1.0 - x * x - y * y).max(0.0).sqrt())
}

/// The X and Y rates of [`equator_pole`], in radians per century.
pub(crate) fn equator_pole_rate(t: f64) -> (f64, f64) {
    series_rate(t, &EQUATOR_POLYNOMIAL, &EQUATOR_PERIODIC)
}

/// Vondrák et al. 2011 ecliptic pole of date in J2000 axes (ERFA `eraLtpecl`).
pub(crate) fn ecliptic_pole(t: f64) -> Vector3<f64> {
    let (p, q) = series(t, &ECLIPTIC_POLYNOMIAL, &ECLIPTIC_PERIODIC);
    let w = (1.0 - p * p - q * q).max(0.0).sqrt();
    let (sin, cos) = (84_381.406 * ARCSECOND).sin_cos();
    Vector3::new(p, -q * cos - w * sin, -q * sin + w * cos)
}

/// Vondrák et al. 2011 rotation from J2000 axes to mean equator and equinox
/// of date (ERFA `eraLtp`).
pub(crate) fn long_term_j2000_to_mean_of_date(t: f64) -> Matrix3<f64> {
    let pole = equator_pole(t);
    let equinox = pole.cross(&ecliptic_pole(t)).normalize();
    let middle = pole.cross(&equinox);
    Matrix3::from_rows(&[equinox.transpose(), middle.transpose(), pole.transpose()])
}

/// Vondrák et al. 2011 mean obliquity: the angle between the two poles.
pub(crate) fn long_term_mean_obliquity(t: f64) -> f64 {
    let (equator, ecliptic) = (equator_pole(t), ecliptic_pole(t));
    equator
        .cross(&ecliptic)
        .norm()
        .atan2(equator.dot(&ecliptic))
}

/// Polynomial plus periodic terms in arcseconds, returned in radians.
fn series(t: f64, polynomial: &[[f64; 2]; 4], periodic: &[[f64; 5]]) -> (f64, f64) {
    let mut sum = polynomial.iter().rev().fold([0.0, 0.0], |total, term| {
        [total[0] * t + term[0], total[1] * t + term[1]]
    });
    for [period, cos_a, cos_b, sin_a, sin_b] in periodic {
        let (sin, cos) = (std::f64::consts::TAU * t / period).sin_cos();
        sum[0] += cos * cos_a + sin * sin_a;
        sum[1] += cos * cos_b + sin * sin_b;
    }
    (sum[0] * ARCSECOND, sum[1] * ARCSECOND)
}

/// The derivative of [`series`] with respect to `t`, in radians per century.
fn series_rate(t: f64, polynomial: &[[f64; 2]; 4], periodic: &[[f64; 5]]) -> (f64, f64) {
    let mut rate = [0.0, 0.0];
    let mut power = 1.0;
    for (n, term) in polynomial.iter().enumerate().skip(1) {
        let n = n as f64;
        rate[0] += n * term[0] * power;
        rate[1] += n * term[1] * power;
        power *= t;
    }
    for [period, cos_a, cos_b, sin_a, sin_b] in periodic {
        let frequency = std::f64::consts::TAU / period;
        let (sin, cos) = (frequency * t).sin_cos();
        rate[0] += frequency * (cos * sin_a - sin * cos_a);
        rate[1] += frequency * (cos * sin_b - sin * cos_b);
    }
    (rate[0] * ARCSECOND, rate[1] * ARCSECOND)
}

/// X and Y polynomial coefficients, in arcseconds per power of `t`.
const EQUATOR_POLYNOMIAL: [[f64; 2]; 4] = [
    [5453.282155, -73750.930350],
    [0.4252841, -0.7675452],
    [-0.00037173, -0.00018725],
    [-0.000000152, 0.000000231],
];

/// Period in centuries, then the cosine X and Y and the sine X and Y
/// amplitudes in arcseconds.
const EQUATOR_PERIODIC: [[f64; 5]; 14] = [
    [256.75, -819.940624, 75004.344875, 81491.287984, 1558.515853],
    [708.15, -8444.676815, 624.033993, 787.163481, 7774.939698],
    [274.20, 2600.009459, 1251.136893, 1251.296102, -2219.534038],
    [
        241.45,
        2755.175630,
        -1102.212834,
        -1257.950837,
        -2523.969396,
    ],
    [2309.00, -167.659835, -2660.664980, -2966.799730, 247.850422],
    [492.20, 871.855056, 699.291817, 639.744522, -846.485643],
    [396.10, 44.769698, 153.167220, 131.600209, -1393.124055],
    [288.90, -512.313065, -950.865637, -445.040117, 368.526116],
    [231.10, -819.415595, 499.754645, 584.522874, 749.045012],
    [1610.00, -538.071099, -145.188210, -89.756563, 444.704518],
    [620.00, -189.793622, 558.116553, 524.429630, 235.934465],
    [157.87, -402.922932, -23.923029, -13.549067, 374.049623],
    [220.30, 179.516345, -165.405086, -210.157124, -171.330180],
    [1200.00, -9.814756, 9.344131, -44.919798, -22.899655],
];

/// P_A and Q_A polynomial coefficients, in arcseconds per power of `t`.
const ECLIPTIC_POLYNOMIAL: [[f64; 2]; 4] = [
    [5851.607687, -1600.886300],
    [-0.1189000, 1.1689818],
    [-0.00028913, -0.00000020],
    [0.000000101, -0.000000437],
];

/// Period in centuries, then the cosine P and Q and the sine P and Q
/// amplitudes in arcseconds.
const ECLIPTIC_PERIODIC: [[f64; 5]; 8] = [
    [708.15, -5486.751211, -684.661560, 667.666730, -5523.863691],
    [2309.00, -17.127623, 2446.283880, -2354.886252, -549.747450],
    [1620.00, -617.517403, 399.671049, -428.152441, -310.998056],
    [492.20, 413.442940, -356.652376, 376.202861, 421.535876],
    [1183.00, 78.614193, -186.387003, 184.778874, -36.776172],
    [622.00, -180.732815, -316.800070, 335.321713, -145.278396],
    [882.00, -87.676083, 198.296701, -185.138669, -34.744450],
    [547.00, 46.140315, 101.135679, -120.972830, 22.885731],
];
