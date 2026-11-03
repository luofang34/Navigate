use core::f64::consts::FRAC_PI_2;

use super::*;
use crate::wgs84::{self, EcefPosition, ecef_to_geodetic, geodetic_to_ecef};

const LATITUDES_DEG: [f64; 10] = [
    -90.0, -89.9, -60.0, -33.5, 0.0, 20.0, 45.0, 89.9, 89.999_999, 90.0,
];
const LONGITUDES_DEG: [f64; 5] = [-179.0, -73.8, 0.0, 8.55, 179.999];

/// Asserts round trips through `ellipsoid` below a micrometer, in the
/// metric space and in height.
fn assert_round_trips(ellipsoid: BodyEllipsoid, altitudes_m: &[f64]) {
    for lat in LATITUDES_DEG {
        for lon in LONGITUDES_DEG {
            for &alt in altitudes_m {
                let original = GeodeticPosition::new(lat.to_radians(), lon.to_radians(), alt);
                let fixed = geodetic_to_body_fixed(&original, ellipsoid);
                let round_tripped = body_fixed_to_geodetic(&fixed, ellipsoid);
                // The metric-space check: re-projecting through the
                // closed-form forward conversion measures the round-trip
                // error in meters without a pole-singular longitude
                // comparison.
                let reprojected = geodetic_to_body_fixed(&round_tripped, ellipsoid);
                let error_m = fixed.distance_m(&reprojected);
                assert!(
                    error_m < 1e-6,
                    "{:?}: round trip error {error_m} m at lat {lat}° lon {lon}° alt {alt} m",
                    ellipsoid.frame()
                );
                assert!(
                    (round_tripped.altitude_m - alt).abs() < 1e-6,
                    "{:?}: altitude error at lat {lat}° lon {lon}° alt {alt} m: {}",
                    ellipsoid.frame(),
                    round_tripped.altitude_m
                );
                assert!(round_tripped.is_plausible());
            }
        }
    }
}

#[test]
fn equator_prime_meridian_is_on_the_x_axis() {
    let ecef = geodetic_to_ecef(&GeodeticPosition::new(0.0, 0.0, 0.0));
    assert!((ecef.x_m - wgs84::SEMI_MAJOR_AXIS_M).abs() < 1e-9);
    assert!(ecef.y_m.abs() < 1e-9);
    assert!(ecef.z_m.abs() < 1e-9);
}

#[test]
fn north_pole_is_on_the_z_axis_at_the_semi_minor_axis() {
    let ecef = geodetic_to_ecef(&GeodeticPosition::new(FRAC_PI_2, 0.0, 0.0));
    assert!(ecef.x_m.abs() < 1e-8);
    assert!(ecef.y_m.abs() < 1e-8);
    assert!((ecef.z_m - wgs84::SEMI_MINOR_AXIS_M).abs() < 1e-6);
}

#[test]
fn round_trip_error_stays_below_a_micrometer() {
    assert_round_trips(
        BodyEllipsoid::WGS84,
        &[-400.0, 0.0, 11_000.0, 80_000.0, 100_000.0],
    );
    assert_round_trips(BodyEllipsoid::GRS80, &[-400.0, 0.0, 100_000.0]);
}

#[test]
fn ecef_helpers_are_the_wgs84_body_fixed_conversions() {
    let position = GeodeticPosition::new(0.4, -1.2, 950.0);
    let ecef = geodetic_to_ecef(&position);
    assert_eq!(
        ecef,
        geodetic_to_body_fixed(&position, BodyEllipsoid::WGS84)
    );
    assert_eq!(
        ecef_to_geodetic(&ecef),
        body_fixed_to_geodetic(&ecef, BodyEllipsoid::WGS84)
    );
}

#[test]
fn polar_axis_reports_latitude_of_ninety_degrees() {
    let below_pole = EcefPosition::new(0.0, 0.0, wgs84::SEMI_MINOR_AXIS_M - 1_000.0);
    let geodetic = ecef_to_geodetic(&below_pole);
    assert!((geodetic.latitude_rad - FRAC_PI_2).abs() < 1e-12);
    assert!((geodetic.altitude_m + 1_000.0).abs() < 1e-6);
    assert!((geodetic.longitude_rad).abs() < 1e-12);
}

#[test]
fn lunar_spheres_round_trip_below_a_micrometer() {
    let altitudes_m = [-9_000.0, 0.0, 10_700.0, 100_000.0, 500_000.0];
    assert_round_trips(BodyEllipsoid::MOON_ME, &altitudes_m);
    assert_round_trips(BodyEllipsoid::MOON_PA, &altitudes_m);
}

#[test]
fn martian_ellipsoid_round_trips_below_a_micrometer() {
    assert_round_trips(
        BodyEllipsoid::MARS_IAU,
        &[-8_200.0, 0.0, 21_900.0, 100_000.0],
    );
}

#[test]
fn a_sphere_gives_the_spherical_latitude_and_radius() {
    let moon = BodyEllipsoid::MOON_ME;
    let radius_m = moon.semi_major_axis_m() + 2_500.0;
    let latitude_rad = 0.7_f64;
    let position = BodyFixedPosition::new(
        radius_m * latitude_rad.cos(),
        0.0,
        radius_m * latitude_rad.sin(),
    );
    let geodetic = body_fixed_to_geodetic(&position, moon);
    assert!((geodetic.latitude_rad - latitude_rad).abs() < 1e-15);
    assert!((geodetic.altitude_m - 2_500.0).abs() < 1e-8);
}

#[test]
fn polar_branches_hold_on_every_body() {
    for ellipsoid in [
        BodyEllipsoid::WGS84,
        BodyEllipsoid::MOON_ME,
        BodyEllipsoid::MARS_IAU,
    ] {
        let b = ellipsoid.semi_minor_axis_m();
        let tiny = 1e-13 * ellipsoid.semi_major_axis_m();
        let north =
            body_fixed_to_geodetic(&BodyFixedPosition::new(tiny, 0.0, b + 300.0), ellipsoid);
        let south = body_fixed_to_geodetic(&BodyFixedPosition::new(0.0, 0.0, -b - 40.0), ellipsoid);
        assert_eq!(north.latitude_rad, FRAC_PI_2);
        assert!((north.altitude_m - 300.0).abs() < 1e-9);
        assert_eq!(south.latitude_rad, -FRAC_PI_2);
        assert!((south.altitude_m - 40.0).abs() < 1e-9);
        assert_eq!(south.longitude_rad, 0.0);
    }
}

#[test]
fn martian_pole_is_on_the_z_axis_at_the_semi_minor_axis() {
    let mars = BodyEllipsoid::MARS_IAU;
    let pole = geodetic_to_body_fixed(&GeodeticPosition::new(-FRAC_PI_2, 0.0, 0.0), mars);
    assert!(pole.x_m.abs() < 1e-8);
    assert!((pole.z_m + 3_376_200.0).abs() < 1e-6);
}
