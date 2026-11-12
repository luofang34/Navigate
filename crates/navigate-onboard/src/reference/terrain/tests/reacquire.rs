//! The first fixes after a long gap: a belief that dead reckoning left
//! kilometres uncertain and off finds the ground again by acquisition,
//! and an acquisition gives no fix at a wrong place.

use std::sync::Arc;

use nalgebra::UnitQuaternion;
use navigate_camera::CameraMount;
use navigate_contract::{AttitudeQuaternion, GeodeticPosition, NedVelocity, SymmetricCov3};

use super::flight::{CAMERA, CLOCK, Flight, RATE, at, show_events, suite};
use super::geometry::{camera, imager};
use super::world::{CENTRE, GROUND_M, TODAY, Texture, World, WorldSpec, frame_at};
use crate::reference::terrain::names::ACQUISITION_UNCONFIRMED;
use crate::{
    Alignment, Navigator, Outcome, ReferenceNavigator, SensorData, SensorInput, TerrainMapRequest,
};

/// The outcome of each of `frames` frames, one a second, of the demo
/// camera hovering 1760 m above the ground, and the horizontal error of
/// the solution after each, with the belief `off` metres (north, east)
/// from the truth and a one-sigma of `sigma_m` on each axis.
fn first_frames(world: &World, off: [f64; 2], sigma_m: f64, frames: u64) -> Vec<(Outcome, f64)> {
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
    let level = UnitQuaternion::identity();
    let mut seen = Vec::new();
    for frame in 1..=frames {
        for step in (frame - 1) * RATE + 1..=frame * RATE {
            navigator.ingest(&hover.increment(step), &mut sink).unwrap();
        }
        let step = frame * RATE;
        let luma = frame_at(world, &camera(), hover.body(step), &level, &mount, step);
        let input = SensorInput {
            sensor: CAMERA,
            acquired_at: at(step),
            delivered_at: at(step),
            data: SensorData::TerrainFrame {
                width: 400,
                height: 300,
                luma,
            },
        };
        navigator.ingest(&input, &mut sink).unwrap();
        let outcome = super::fixes::terrain(&sink)
            .last()
            .map(|r| r.0.clone())
            .unwrap();
        let error = hover.error_m(&navigator.solution(at(step)).unwrap(), step);
        seen.push((outcome, error.xy().norm()));
    }
    seen
}

fn world(texture: Texture) -> World {
    World::build(&WorldSpec {
        radius_m: 6_000.0,
        texture,
        ..WorldSpec::default()
    })
}

/// North and east offsets of `magnitude` metres in `count` directions.
fn around(magnitude: f64, count: u32) -> Vec<[f64; 2]> {
    (0..count)
        .map(|k| {
            let angle = std::f64::consts::TAU * f64::from(k) / f64::from(count) + 0.3;
            [magnitude * angle.cos(), magnitude * angle.sin()]
        })
        .collect()
}

/// From 1760 m over farmland, a belief of 1 km one-sigma that is 0.5 to
/// 2 km off is found again: the first frame acquires and waits, the
/// second confirms it and gives the fix.
#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn a_belief_kilometres_off_finds_the_ground_again_in_two_frames() {
    show_events();
    let world = world(Texture::Farmland);
    for magnitude in [500.0, 1_000.0, 2_000.0] {
        for off in around(magnitude, 2) {
            let seen = first_frames(&world, off, 1_000.0, 2);
            tracing::info!(?off, ?seen, "the first frames on a wide belief");
            assert_eq!(
                seen[0].0,
                Outcome::Unused(ACQUISITION_UNCONFIRMED),
                "{off:?}"
            );
            assert_eq!(seen[1].0, Outcome::Accepted, "{off:?}");
            assert!(seen[1].1 < 20.0, "{off:?}: {}", seen[1].1);
        }
    }
}

/// Over farmland, over regular and irregular town blocks, and with the
/// truth past the acquisition radius and near the edge of the map, no
/// acquisition gives the filter a fix more than 30 m from the truth.
#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn an_acquisition_gives_no_fix_at_a_wrong_place() {
    show_events();
    for texture in [Texture::Farmland, Texture::Town, Texture::TownIrregular] {
        let world = world(texture);
        let (mut fixes, mut wrong, mut trials) = (0, 0, 0);
        for magnitude in [500.0, 2_000.0, 4_000.0] {
            for off in around(magnitude, 3) {
                for (outcome, error_m) in first_frames(&world, off, 1_000.0, 3) {
                    trials += 1;
                    if outcome == Outcome::Accepted {
                        fixes += 1;
                        if error_m > 30.0 {
                            wrong += 1;
                            tracing::warn!(?texture, ?off, error_m, "a wrong acquisition");
                        }
                    }
                }
            }
        }
        tracing::info!(?texture, trials, fixes, wrong, "acquisitions");
        assert_eq!(wrong, 0, "{texture:?}");
    }
}
