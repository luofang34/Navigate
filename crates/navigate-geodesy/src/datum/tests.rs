//! The datum conversions apply `h = H + N` and fail closed.

use core::f64::consts::FRAC_PI_4;

use navigate_contract::Ellipsoid;

use super::*;

/// A toy geoid: 30 m north of the equator, −12 m south of it, and no
/// coverage poleward of 60°.
struct BandGeoid;

impl GeoidUndulation for BandGeoid {
    fn ellipsoid(&self) -> Ellipsoid {
        Ellipsoid::WGS84
    }

    fn undulation_m(&self, latitude_rad: f64, _longitude_rad: f64) -> Option<f64> {
        if latitude_rad.abs() > 60.0_f64.to_radians() {
            None
        } else if latitude_rad >= 0.0 {
            Some(30.0)
        } else {
            Some(-12.0)
        }
    }
}

#[test]
fn an_orthometric_height_gains_the_undulation_and_the_inverse_removes_it() {
    let h = orthometric_to_ellipsoidal_m(&BandGeoid, Ellipsoid::WGS84, FRAC_PI_4, 0.1, 100.0);
    assert_eq!(h, Some(130.0));
    let south = orthometric_to_ellipsoidal_m(&BandGeoid, Ellipsoid::WGS84, -0.3, 0.1, 100.0);
    assert_eq!(south, Some(88.0));
    let back = ellipsoidal_to_orthometric_m(&BandGeoid, Ellipsoid::WGS84, FRAC_PI_4, 0.1, 130.0);
    assert_eq!(back, Some(100.0));
}

#[test]
fn absent_coverage_a_non_finite_value_and_another_ellipsoid_are_refused() {
    let polar = 70.0_f64.to_radians();
    assert_eq!(
        orthometric_to_ellipsoidal_m(&BandGeoid, Ellipsoid::WGS84, polar, 0.0, 10.0),
        None,
        "no coverage is not a zero undulation"
    );
    assert_eq!(
        ellipsoidal_to_orthometric_m(&BandGeoid, Ellipsoid::WGS84, 0.1, f64::NAN, 10.0),
        None
    );
    assert_eq!(
        orthometric_to_ellipsoidal_m(&BandGeoid, Ellipsoid::WGS84, 0.1, 0.0, f64::INFINITY),
        None
    );
    assert_eq!(
        orthometric_to_ellipsoidal_m(&BandGeoid, Ellipsoid::GRS80, 0.1, 0.0, 10.0),
        None,
        "a height on GRS80 does not take a WGS84 undulation"
    );
}

#[test]
fn a_zero_undulation_keeps_the_height_on_its_own_ellipsoid_only() {
    let selenoid = ZeroUndulation(Ellipsoid::MOON_ME);
    assert_eq!(selenoid.ellipsoid(), Ellipsoid::MOON_ME);
    let as_dyn: &dyn GeoidUndulation = &selenoid;
    assert_eq!(
        orthometric_to_ellipsoidal_m(as_dyn, Ellipsoid::MOON_ME, -1.5, 3.0, -2_500.0),
        Some(-2_500.0)
    );
    assert_eq!(
        orthometric_to_ellipsoidal_m(&selenoid, Ellipsoid::MOON_PA, -1.5, 3.0, -2_500.0),
        None,
        "MOON_PA and MOON_ME are different ellipsoids"
    );
    assert_eq!(selenoid.undulation_m(f64::NAN, 0.0), None);
}
