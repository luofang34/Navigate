//! The facade gives the same values with `std` and with `libm`. The CI
//! runs these tests in both builds.

use core::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

use super::*;

fn close(actual: f64, expected: f64) -> bool {
    (actual - expected).abs() <= 1e-15 * expected.abs().max(1.0)
}

#[test]
fn trigonometric_functions_hit_exact_reference_values() {
    assert!(close(sin(FRAC_PI_2), 1.0));
    assert!(close(cos(PI), -1.0));
    assert!(close(tan(FRAC_PI_4), 1.0));
    assert!(close(asin(1.0), FRAC_PI_2));
    assert!(close(acos(-1.0), PI));
    assert!(close(atan(1.0), FRAC_PI_4));
    assert!(close(atan2(1.0, -1.0), 3.0 * FRAC_PI_4));
    assert!(close(atan2(-0.0, -1.0), -PI), "the sign of zero selects -π");
    let (s, c) = sin_cos(PI / 6.0);
    assert!(close(s, 0.5) && close(c, 3.0_f64.sqrt() / 2.0));
}

#[test]
fn roots_and_remainders_hit_exact_reference_values() {
    assert_eq!(sqrt(6_378_137.0 * 6_378_137.0), 6_378_137.0);
    assert_eq!(hypot(3e300, 4e300), 5e300, "no intermediate overflow");
    assert_eq!(rem_euclid(-1.0, TAU), TAU - 1.0);
    assert_eq!(rem_euclid(7.5, 2.0), 1.5);
    assert_eq!(
        rem_euclid(-7.5, -2.0),
        0.5,
        "the remainder is never negative"
    );
}
