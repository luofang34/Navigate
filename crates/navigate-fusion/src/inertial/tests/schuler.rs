//! The free-inertial error oscillates with the Schuler period
//! `2π·√(r/g)` on each body, in the body-fixed frame of the body.

use std::vec::Vec;

use core::f64::consts::TAU;

use navigate_contract::GeodeticPosition;

use crate::inertial::config::InertialConfig;
use crate::inertial::local_level::{LocalLevel, body_fixed, quaternion_of};
use crate::inertial::mechanization::{ImuRates, mechanize};
use crate::inertial::nominal::InertialNominal;
use navigate_astrodynamics::{GravityField, PlanetaryBody, SpaceEnvironment};

/// The measured period from the second zero crossing of the north error
/// after a 1 m/s north velocity error, and the expected period.
fn schuler_period_s(body: PlanetaryBody, lat_deg: f64) -> (f64, f64) {
    let origin = GeodeticPosition::new(lat_deg.to_radians(), 0.3, 0.0);
    let field = SpaceEnvironment::from_body(body);
    let local = LocalLevel::at(origin);
    let p0 = body_fixed(&origin, body.ellipsoid);
    let north = local.ned_to_bf.column(0).into_owned();
    let (g, _) = field.gravity(&p0);
    let expected = TAU * (p0.norm() / g.norm()).sqrt();
    let c = local.ned_to_bf;
    let rates = ImuRates {
        specific_force: c.transpose() * -g,
        angular_rate: c.transpose() * field.frame_rotation(),
    };
    let config = InertialConfig::default();
    let mut x = InertialNominal::at(p0, north, quaternion_of(&c));
    let dt_s = 1.0;
    let mut crossings = Vec::new();
    let mut time_s = 0.0;
    let north_error = |x: &InertialNominal| north.dot(&(x.p - p0));
    while crossings.len() < 2 && time_s < 2.0 * expected {
        let (next, _) = mechanize(&x, &rates, dt_s, &field, &config);
        assert!(next.p.iter().chain(next.v.iter()).all(|c| c.is_finite()));
        let (before, after) = (north_error(&x), north_error(&next));
        if time_s > 0.0 && before.signum() != after.signum() {
            let fraction = before / (before - after);
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
fn the_poles_have_no_singularity() {
    let (pole, expected_pole) = schuler_period_s(PlanetaryBody::MOON_ME, -90.0);
    let (equator, _) = schuler_period_s(PlanetaryBody::MOON_ME, 0.0);
    assert!(
        (pole / expected_pole - 1.0).abs() < 0.01,
        "{pole} vs {expected_pole}"
    );
    assert!((pole / equator - 1.0).abs() < 0.01, "{pole} vs {equator}");
    let (north_pole, expected_north) = schuler_period_s(PlanetaryBody::EARTH_WGS84, 90.0);
    assert!(
        (north_pole / expected_north - 1.0).abs() < 0.01,
        "{north_pole} vs {expected_north}"
    );
}
