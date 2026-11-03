use core::f64::consts::{FRAC_PI_2, PI};

use navigate_contract::{Ellipsoid, FrameId};

use super::{BodyEllipsoid, west_to_east_lon_rad};
use crate::error::GeodesyError;
use crate::wgs84;

const PRESETS: [BodyEllipsoid; 5] = [
    BodyEllipsoid::WGS84,
    BodyEllipsoid::GRS80,
    BodyEllipsoid::MOON_ME,
    BodyEllipsoid::MOON_PA,
    BodyEllipsoid::MARS_IAU,
];

#[test]
fn wgs84_preset_matches_the_wgs84_constants_exactly() {
    let e = BodyEllipsoid::WGS84;
    assert_eq!(e.contract(), Ellipsoid::WGS84);
    assert_eq!(e.frame(), FrameId::EARTH_ITRF93);
    assert_eq!(e.semi_major_axis_m(), wgs84::SEMI_MAJOR_AXIS_M);
    assert_eq!(e.flattening(), wgs84::FLATTENING);
    assert_eq!(e.semi_minor_axis_m(), wgs84::SEMI_MINOR_AXIS_M);
    assert_eq!(
        e.first_eccentricity_squared(),
        wgs84::FIRST_ECCENTRICITY_SQUARED
    );
    assert_eq!(
        e.second_eccentricity_squared(),
        wgs84::SECOND_ECCENTRICITY_SQUARED
    );
    assert_eq!(e.mean_radius_m(), wgs84::MEAN_RADIUS_M);
}

#[test]
fn presets_are_the_converted_contract_presets() {
    for preset in PRESETS {
        let converted = BodyEllipsoid::try_from_contract(preset.contract());
        assert_eq!(converted, Ok(preset));
    }
}

#[test]
fn grs80_differs_from_wgs84_below_a_millimeter_at_the_pole() {
    let difference_m =
        BodyEllipsoid::GRS80.semi_minor_axis_m() - BodyEllipsoid::WGS84.semi_minor_axis_m();
    assert!(difference_m.abs() < 1e-3, "{difference_m}");
    assert!(difference_m != 0.0);
}

#[test]
fn lunar_spheres_have_zero_eccentricity() {
    for moon in [BodyEllipsoid::MOON_ME, BodyEllipsoid::MOON_PA] {
        assert_eq!(moon.semi_minor_axis_m(), 1_737_400.0);
        assert_eq!(moon.first_eccentricity_squared(), 0.0);
        assert_eq!(moon.second_eccentricity_squared(), 0.0);
        assert_eq!(moon.mean_radius_m(), 1_737_400.0);
        assert_eq!(moon.radii_of_curvature_m(0.4), (1_737_400.0, 1_737_400.0));
    }
    assert_ne!(
        BodyEllipsoid::MOON_ME.frame(),
        BodyEllipsoid::MOON_PA.frame()
    );
}

#[test]
fn martian_constants_match_the_iau_values() {
    let mars = BodyEllipsoid::MARS_IAU;
    assert_eq!(mars.frame(), FrameId::MARS_IAU);
    assert!((mars.semi_minor_axis_m() - 3_376_200.0).abs() < 1e-6);
    let expected_e2 = 1.0 - (3_376_200.0_f64 / 3_396_190.0).powi(2);
    assert!((mars.first_eccentricity_squared() - expected_e2).abs() < 1e-15);
    assert!((mars.mean_radius_m() - 3_389_526.666_666_667).abs() < 1e-6);
}

#[test]
fn implausible_contract_ellipsoids_are_refused() {
    let refused = [
        Ellipsoid::new(FrameId::MOON_ME, -1.0, 0.0),
        Ellipsoid::new(FrameId::MARS_IAU, 3_396_190.0, 1.0),
        Ellipsoid::new(FrameId::MARS_IAU, f64::NAN, 0.0),
        Ellipsoid::sphere(FrameId::MOON_J2000, 1_737_400.0),
    ];
    for ellipsoid in refused {
        assert!(
            matches!(
                BodyEllipsoid::try_from_contract(ellipsoid),
                Err(GeodesyError::ImplausibleEllipsoid { .. })
            ),
            "{ellipsoid:?}"
        );
    }
}

#[test]
fn radii_of_curvature_follow_the_ellipsoid() {
    let e = BodyEllipsoid::WGS84;
    let (meridian_eq, prime_eq) = e.radii_of_curvature_m(0.0);
    assert!((prime_eq - wgs84::SEMI_MAJOR_AXIS_M).abs() < 1e-6);
    let b = wgs84::SEMI_MINOR_AXIS_M;
    let a = wgs84::SEMI_MAJOR_AXIS_M;
    assert!((meridian_eq - b * b / a).abs() < 1e-6);
    let (meridian_pole, prime_pole) = e.radii_of_curvature_m(FRAC_PI_2);
    assert!((meridian_pole - a * a / b).abs() < 1e-6);
    assert!((prime_pole - a * a / b).abs() < 1e-6);
}

#[test]
fn latitude_conversions_round_trip_on_every_preset() {
    for preset in PRESETS {
        for lat_deg in [-90.0_f64, -60.0, -1.0, 0.0, 22.5, 45.0, 89.0, 90.0] {
            let geodetic = lat_deg.to_radians();
            let centric = preset.geodetic_to_planetocentric_lat(geodetic);
            let back = preset.planetocentric_to_geodetic_lat(centric);
            assert!((back - geodetic).abs() < 1e-14, "{preset:?} at {lat_deg}°");
            assert!(centric.abs() <= geodetic.abs() + 1e-15);
        }
    }
}

#[test]
fn latitude_conversions_are_the_identity_on_a_sphere() {
    let moon = BodyEllipsoid::MOON_ME;
    for lat in [-1.2, -0.2, 0.0, 0.9, FRAC_PI_2] {
        assert!((moon.geodetic_to_planetocentric_lat(lat) - lat).abs() < 1e-15);
        assert!((moon.planetocentric_to_geodetic_lat(lat) - lat).abs() < 1e-15);
    }
}

#[test]
fn martian_planetographic_latitude_exceeds_the_planetocentric_latitude() {
    let mars = BodyEllipsoid::MARS_IAU;
    let centric = 45.0_f64.to_radians();
    let graphic = mars.planetocentric_to_geodetic_lat(centric);
    // tan φg = tan φc / (b/a)², so at 45° φg ≈ 45.34°.
    let expected = (1.0 / (3_376_200.0_f64 / 3_396_190.0).powi(2)).atan();
    assert!((graphic - expected).abs() < 1e-12);
    assert!(
        (graphic.to_degrees() - 45.34).abs() < 0.01,
        "{}",
        graphic.to_degrees()
    );
    assert!((mars.planetocentric_to_geodetic_lat(FRAC_PI_2) - FRAC_PI_2).abs() < 1e-15);
}

#[test]
fn west_longitudes_become_east_longitudes_in_range() {
    let cases = [
        (0.0, 0.0),
        (90.0, -90.0),
        (180.0, 180.0),
        (270.0, 90.0),
        (359.0, 1.0),
        (-45.0, 45.0),
        (137.4, -137.4),
        (720.0 + 10.0, -10.0),
    ];
    for (west_deg, east_deg) in cases {
        let east = west_to_east_lon_rad(f64::to_radians(west_deg));
        assert!(
            (east - f64::to_radians(east_deg)).abs() < 1e-12,
            "{west_deg}°W gives {}°",
            east.to_degrees()
        );
        assert!((-PI..=PI).contains(&east));
    }
}
