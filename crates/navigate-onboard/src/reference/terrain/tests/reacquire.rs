//! The first fix after a long gap: a belief that dead reckoning left
//! kilometres uncertain and some hundreds of metres off still finds the
//! ground, because the search of the match widens with the prior up to
//! the reach of the plain matcher: 32 working pixels, about 230 m from
//! 1760 m with the demo camera.

use std::sync::Arc;

use nalgebra::UnitQuaternion;
use navigate_camera::CameraMount;
use navigate_contract::{AttitudeQuaternion, GeodeticPosition, NedVelocity, SymmetricCov3};

use super::fixes::terrain;
use super::flight::{CAMERA, CLOCK, Flight, RATE, at, show_events, suite};
use super::geometry::{camera, imager};
use super::world::{CENTRE, GROUND_M, TODAY, World, WorldSpec, frame_at};
use crate::{
    Alignment, Navigator, Outcome, ReferenceNavigator, SensorData, SensorInput, TerrainMapRequest,
};

/// The outcome and the horizontal error of one frame of the demo camera
/// from 1760 m above the ground, with the belief `off` metres (north,
/// east) from the truth and a one-sigma of `sigma_m` on each axis.
fn first_fix(world: &World, off: [f64; 2], sigma_m: f64) -> (Outcome, f64) {
    let hover = Flight {
        start: [CENTRE[0], CENTRE[1], GROUND_M + 1_760.0],
        speed_m_s: 0.0,
    };
    let mount = CameraMount::nadir();
    let mut navigator = ReferenceNavigator::new(suite(imager(), mount), CLOCK, None).unwrap();
    let [lat, lon, h] = hover.start;
    let variance = sigma_m * sigma_m;
    navigator
        .align(&Alignment {
            position: GeodeticPosition::new(
                lat.to_radians() + off[0] / 6_364_000.0,
                lon.to_radians() + off[1] / 4_760_000.0,
                h + 10.0,
            ),
            velocity: NedVelocity::new(0.0, 0.0, 0.0),
            attitude: AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0),
            position_cov: SymmetricCov3::from_diagonal(variance, variance, 400.0),
            velocity_cov: SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            attitude_cov: SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-5),
            at: at(0),
            imu_bias_prior: None,
        })
        .unwrap();
    let request = TerrainMapRequest::new(world.package, TODAY);
    navigator
        .load_terrain_map(Arc::clone(&world.store), &request, None)
        .unwrap();
    let mut sink = Vec::new();
    for step in 1..=RATE {
        navigator.ingest(&hover.increment(step), &mut sink).unwrap();
    }
    let level = UnitQuaternion::identity();
    let luma = frame_at(world, &camera(), hover.body(RATE), &level, &mount, RATE);
    let frame = SensorInput {
        sensor: CAMERA,
        acquired_at: at(RATE),
        delivered_at: at(RATE),
        data: SensorData::TerrainFrame {
            width: 400,
            height: 300,
            luma,
        },
    };
    navigator.ingest(&frame, &mut sink).unwrap();
    let outcome = terrain(&sink).first().map(|r| r.0.clone()).unwrap();
    let error = hover.error_m(&navigator.solution(at(RATE)).unwrap(), RATE);
    (outcome, error.xy().norm())
}

/// From 1760 m, a belief of 2 km one-sigma that is 200 m off is found
/// again in one frame; the tight search of a small prior would not reach
/// that far.
#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn a_belief_kilometres_uncertain_finds_the_ground_again() {
    show_events();
    let world = World::build(&WorldSpec {
        radius_m: 6_000.0,
        ..WorldSpec::default()
    });
    for off in [[140.0, -140.0], [-110.0, 165.0]] {
        let (outcome, error_m) = first_fix(&world, off, 2_000.0);
        tracing::info!(?off, ?outcome, error_m, "the first fix on a wide belief");
        assert_eq!(outcome, Outcome::Accepted, "{off:?}");
        assert!(error_m < 20.0, "{off:?}: {error_m}");
    }
}
