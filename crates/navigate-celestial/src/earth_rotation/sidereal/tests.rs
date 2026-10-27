use super::*;

const ERFA_LTP: &str = include_str!("../../../fixtures/erfa-ltp.txt");
const MAS: f64 = std::f64::consts::PI / 648_000_000.0;
const J2000_JD: f64 = 2_451_545.0;

#[test]
fn long_term_precession_matches_erfa_from_minus_2000_to_3000() {
    let mut rows = 0;
    for line in ERFA_LTP.lines().filter(|line| !line.starts_with('#')) {
        let v: Vec<f64> = line.split(' ').map(|x| x.parse().unwrap()).collect();
        let t = (v[0] - 2000.0) / 100.0;
        let matrix = precession::long_term_j2000_to_mean_of_date(t);
        for row in 0..3 {
            for column in 0..3 {
                let error = (matrix[(row, column)] - v[1 + 3 * row + column]).abs();
                assert!(error < 1e-12, "epoch {} ({row}, {column}): {error}", v[0]);
            }
        }
        rows += 1;
    }
    assert_eq!(rows, 14);
}

#[test]
fn earth_rotation_angle_matches_erfa() {
    for days in [-146_100.0, -36_525.25, 0.0, 9_662.5, 36_525.75, 73_050.125] {
        let expected = erfa::earth::earth_rotation_angle_00(J2000_JD, days);
        let error = (earth_rotation_angle(days) - expected).abs();
        assert!(error < 1e-11, "{days}: {error}");
    }
}

/// The measured largest difference is 23 milliarcseconds.
#[test]
fn gmst_matches_iau_2006_from_1800_to_2200() {
    let mut worst = 0.0_f64;
    for step in 0..=400 {
        // UT1 = TT here; the comparison is of the sidereal model alone.
        let days = -73_048.5 + f64::from(step) * 365.2425 + 0.37 * f64::from(step);
        let t = days / 36_525.0;
        let expected = erfa::time::gmst06(J2000_JD, days, J2000_JD, days);
        let difference = (gmst(days, t) - expected + std::f64::consts::PI)
            .rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI;
        worst = worst.max(difference.abs());
    }
    assert!(worst < 50.0 * MAS, "{} mas", worst / MAS);
}

#[test]
fn cio_locator_quadrature_has_converged() {
    for t in [-40.0, -2.0, 2.0, 10.0] {
        let halves = integrate(t / 2.0, &GAUSS_LEGENDRE) + second_half(t);
        let error = (cio_locator(t) - halves).abs();
        assert!(error < 1e-12, "t = {t}: {error}");
    }
}

#[test]
fn equator_pole_rate_is_the_derivative_of_the_pole() {
    let step = 1e-4;
    for t in [-40.0, -2.0, 0.0, 2.0, 10.0] {
        let (x_rate, y_rate) = precession::equator_pole_rate(t);
        let difference = (precession::equator_pole(t + step) - precession::equator_pole(t - step))
            / (2.0 * step);
        assert!((x_rate - difference.x).abs() < 1e-10, "t = {t}");
        assert!((y_rate - difference.y).abs() < 1e-10, "t = {t}");
    }
}

/// The integral from t/2 to t with its own 16 nodes.
fn second_half(t: f64) -> f64 {
    let half = t / 2.0;
    -half
        * GAUSS_LEGENDRE
            .iter()
            .map(|[node, weight]| {
                let tau = half + node * half;
                let pole = precession::equator_pole(tau);
                let (x_rate, y_rate) = precession::equator_pole_rate(tau);
                weight * (pole.x * y_rate - pole.y * x_rate) / (1.0 + pole.z)
            })
            .sum::<f64>()
}
