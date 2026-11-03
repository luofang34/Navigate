//! A lunar south pole landing in closed loop: a coast at 15 km
//! perilune, powered descent initiation (PDI), E-guidance to a soft
//! touchdown in the principal axis frame `MOON_PA`, and the fail-closed
//! refusal when the position uncertainty exceeds the landing bound.
//!
//! The simulation runs in the body-fixed frame with the apparent gravity
//! of `PlanetaryBody::MOON_PA`, about 1.62 m/s² at the surface, plus the
//! Coriolis acceleration that the guidance law does not model.

use navigate_contract::GeodeticPosition;
use navigate_geodesy::{BodyEllipsoid, geodetic_to_body_fixed};
use navigate_trajectory::GlideSlope;

use super::*;
use crate::config::{GuidanceConfig, SpaceGuidanceConfig};

const MOON: PlanetaryBody = PlanetaryBody::MOON_PA;
const MOON_RADIUS_M: f64 = 1_737_400.0;
const SITE_LATITUDE_DEG: f64 = -89.5;
const PDI_ALTITUDE_M: f64 = 15e3;
/// The perilune speed of a 15 km by 100 km orbit.
const PDI_SPEED_MPS: f64 = 1_692.7;
/// The arc from PDI to the site, about 503 km.
const PDI_UPRANGE_DEG: f64 = 16.6;
const PDI_MS: u64 = 10_000;
const DESCENT_MS: u64 = 660_000;
const STEP_MS: u64 = 200;
const LANDING_1SIGMA_BOUND_M: f64 = 10.0;

fn body_fixed(latitude_deg: f64, altitude_m: f64) -> Vector3<f64> {
    let position = GeodeticPosition::new(latitude_deg.to_radians(), 0.0, altitude_m);
    let p = geodetic_to_body_fixed(&position, BodyEllipsoid::MOON_PA);
    Vector3::new(p.x_m, p.y_m, p.z_m)
}

/// The vehicle in the coast before PDI, flying south along the prime
/// meridian toward the site.
fn coasting_vehicle() -> StateVector {
    let latitude = (SITE_LATITUDE_DEG + PDI_UPRANGE_DEG).to_radians();
    let position = body_fixed(SITE_LATITUDE_DEG + PDI_UPRANGE_DEG, PDI_ALTITUDE_M);
    let north = Vector3::new(-latitude.sin(), 0.0, latitude.cos());
    let transport = MOON.rotation_body_fixed().cross(&position);
    StateVector::new(position, -north * PDI_SPEED_MPS - transport)
}

fn landing_profile() -> MissionProfile<'static> {
    let target = PoweredDescentTarget::new(
        body_fixed(SITE_LATITUDE_DEG, 0.0),
        Vector3::zeros(),
        Vector3::zeros(),
        0.5,
        5.0,
    )
    .with_glide_slope(GlideSlope::new(5f64.to_radians(), 3_000.0));
    MissionProfile::Landing {
        descent: PoweredDescentPlan::new(
            MOON,
            target,
            at_millis(PDI_MS),
            at_millis(PDI_MS + DESCENT_MS),
        ),
    }
}

fn landing_config() -> MissionConfig {
    MissionConfig::new(
        SpaceGuidanceConfig::default().with_max_position_1sigma_m(LANDING_1SIGMA_BOUND_M),
        GuidanceConfig::default(),
    )
}

/// The body-fixed acceleration: apparent gravity, Coriolis, and thrust.
fn acceleration(p: &Vector3<f64>, v: &Vector3<f64>, thrust: &Vector3<f64>) -> Vector3<f64> {
    let (gravity, _) = MOON.apparent_gravity_body_fixed(p);
    gravity - MOON.rotation_body_fixed().cross(v) * 2.0 + thrust
}

/// What a closed-loop landing gives.
struct Landing {
    vehicle: StateVector,
    millis: u64,
    min_altitude_m: f64,
    max_specific_force_mps2: f64,
}

/// Flies the landing profile with the one-sigma uncertainty
/// `sigma_m(millis)` until touchdown, or until the first refusal. It
/// panics when the vehicle does not land before the time limit.
fn fly(sigma_m: impl Fn(u64) -> f64) -> Result<Landing, GuidanceRefusal> {
    let profile = landing_profile();
    let mut state = MissionState::START;
    let mut landing = Landing {
        vehicle: coasting_vehicle(),
        millis: 0,
        min_altitude_m: f64::INFINITY,
        max_specific_force_mps2: 0.0,
    };
    while landing.millis < PDI_MS + DESCENT_MS + 120_000 {
        let frame = profile.expected_frame(state.phase).unwrap();
        let now = at_millis(landing.millis);
        let solution = space_solution(frame, &landing.vehicle, now, sigma_m(landing.millis));
        let input = MissionInput::Space {
            solution: &solution,
            sensed_drag_mps2: 0.0,
        };
        let step = step_mission(state, &profile, input, now, CLOCK, &landing_config())?;
        state = step.state;
        if state.phase == MissionPhase::Landed {
            return Ok(landing);
        }
        let thrust = thrust_of(step.command).map_or(Vector3::zeros(), |(d, f, _)| d * f);
        landing.max_specific_force_mps2 = landing.max_specific_force_mps2.max(thrust.norm());
        landing.vehicle = rk4(&landing.vehicle, STEP_MS as f64 / 1e3, |p, v| {
            acceleration(p, v, &thrust)
        });
        let altitude = landing.vehicle.position_m.norm() - MOON_RADIUS_M;
        landing.min_altitude_m = landing.min_altitude_m.min(altitude);
        landing.millis += STEP_MS;
    }
    panic!(
        "no touchdown before the time limit, phase {:?}",
        state.phase
    );
}

#[test]
fn e_guidance_lands_softly_at_the_lunar_south_pole() {
    let landing = fly(|_| 3.0).unwrap();
    let site = body_fixed(SITE_LATITUDE_DEG, 0.0);
    let miss_m = (landing.vehicle.position_m - site).norm();
    let speed = landing.vehicle.velocity_mps.norm();
    assert!(miss_m < 1.0, "touchdown {miss_m} m from the site");
    assert!(speed < 0.5, "touchdown at {speed} m/s");
    assert!(
        landing.min_altitude_m > -1.0,
        "{} m",
        landing.min_altitude_m
    );
    assert!(landing.max_specific_force_mps2 <= 5.0 + 1e-9);
    let flown_s = (landing.millis - PDI_MS) as f64 / 1e3;
    let planned_s = DESCENT_MS as f64 / 1e3;
    assert!(
        (planned_s - 10.0..planned_s + 60.0).contains(&flown_s),
        "touchdown {flown_s} s after PDI, {miss_m} m, {speed} m/s, min altitude {} m",
        landing.min_altitude_m
    );
    let surface_gravity = MOON.surface_gravity_mps2(SITE_LATITUDE_DEG.to_radians());
    assert!(
        (surface_gravity - 1.62).abs() < 0.01,
        "{surface_gravity} m/s²"
    );
}

#[test]
fn a_position_uncertainty_over_the_landing_bound_is_refused() {
    assert_eq!(
        fly(|_| 50.0).err(),
        Some(GuidanceRefusal::UncertaintyExceeded {
            max_axis_1sigma_m: 50.0,
            bound_m: LANDING_1SIGMA_BOUND_M,
        })
    );
    let degraded_at = PDI_MS + 300_000;
    let sigma = |millis| if millis < degraded_at { 3.0 } else { 25.0 };
    assert_eq!(
        fly(sigma).err(),
        Some(GuidanceRefusal::UncertaintyExceeded {
            max_axis_1sigma_m: 25.0,
            bound_m: LANDING_1SIGMA_BOUND_M,
        })
    );
}
