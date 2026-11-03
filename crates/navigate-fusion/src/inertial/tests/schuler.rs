//! The free-inertial error oscillates with the Schuler period
//! `2π·√(r/g)` on each body.

use std::vec::Vec;

use core::f64::consts::TAU;

use nalgebra::{UnitQuaternion, Vector3};
use navigate_contract::GeodeticPosition;
use navigate_geodesy::LocalTangentPlane;

use crate::inertial::config::InertialConfig;
use crate::inertial::mechanization::{ImuRates, mechanize};
use crate::inertial::nominal::InertialNominal;
use navigate_astrodynamics::{GravityField, PlanetaryBody, SurfaceFrame};

/// The measured period from the second zero crossing of the north error
/// after a 1 m/s north velocity error, and the expected period.
fn schuler_period_s(body: PlanetaryBody, lat_deg: f64) -> (f64, f64) {
    let origin = GeodeticPosition::new(lat_deg.to_radians(), 0.3, 0.0);
    let plane = LocalTangentPlane::new(origin, body.ellipsoid).expect("plausible");
    let frame = SurfaceFrame::new(plane, body);
    let (g, _) = frame.gravity(&Vector3::zeros());
    let radius_m = frame.body_fixed_position(&Vector3::zeros()).norm();
    let expected = TAU * (radius_m / g.norm()).sqrt();
    let rates = ImuRates {
        specific_force: -g,
        angular_rate: frame.frame_rotation(),
    };
    let config = InertialConfig::default();
    let mut x = InertialNominal {
        p: Vector3::zeros(),
        v: Vector3::new(1.0, 0.0, 0.0),
        q: UnitQuaternion::identity(),
        b_a: Vector3::zeros(),
        b_g: Vector3::zeros(),
        b_alt: 0.0,
        b_mag: 0.0,
    };
    let dt_s = 1.0;
    let mut crossings = Vec::new();
    let mut time_s = 0.0;
    while crossings.len() < 2 && time_s < 2.0 * expected {
        let (next, _) = mechanize(&x, &rates, dt_s, &frame, &config);
        assert!(next.p.iter().chain(next.v.iter()).all(|c| c.is_finite()));
        if time_s > 0.0 && x.p.x.signum() != next.p.x.signum() {
            let fraction = x.p.x / (x.p.x - next.p.x);
            crossings.push(time_s + fraction * dt_s);
        }
        x = next;
        time_s += dt_s;
    }
    assert_eq!(crossings.len(), 2, "no full period");
    (crossings[1], expected)
}

#[test]
fn the_schuler_period_holds_on_the_earth_the_moon_and_mars() {
    let cases = [
        (PlanetaryBody::EARTH_WGS84, 47.0, 84.4),
        (PlanetaryBody::MOON_ME, 0.0, 108.3),
        (PlanetaryBody::MARS_IAU, 20.0, 100.1),
    ];
    for (body, lat_deg, nominal_min) in cases {
        let (measured, expected) = schuler_period_s(body, lat_deg);
        assert!(
            (measured / expected - 1.0).abs() < 0.01,
            "{lat_deg}: {measured} vs {expected}"
        );
        assert!((expected / 60.0 - nominal_min).abs() < 1.0, "{expected}");
    }
}

#[test]
fn the_lunar_south_pole_has_no_singularity() {
    let (pole, expected_pole) = schuler_period_s(PlanetaryBody::MOON_ME, -90.0);
    let (equator, _) = schuler_period_s(PlanetaryBody::MOON_ME, 0.0);
    assert!(
        (pole / expected_pole - 1.0).abs() < 0.01,
        "{pole} vs {expected_pole}"
    );
    assert!((pole / equator - 1.0).abs() < 0.01, "{pole} vs {equator}");
}
