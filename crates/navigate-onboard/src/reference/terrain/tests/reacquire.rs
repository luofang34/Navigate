//! The first fixes after a long gap: a belief that dead reckoning left
//! kilometres uncertain and off finds the ground again by acquisition,
//! and an acquisition gives no fix at a wrong place.

use std::sync::Arc;

use nalgebra::UnitQuaternion;
use navigate_camera::CameraMount;
use navigate_contract::{AttitudeQuaternion, GeodeticPosition, NedVelocity, SymmetricCov3};

use super::flight::{CAMERA, CLOCK, Flight, RATE, at, show_events, suite};
use super::geometry::mismatch::Mismatch;
use super::geometry::{camera, imager};
use super::world::{CENTRE, GROUND_M, TODAY, Texture, World, WorldSpec, frame_at};
use crate::reference::terrain::names::ACQUISITION_UNCONFIRMED;
use crate::{
    Alignment, Navigator, Outcome, ReferenceNavigator, SensorData, SensorInput, TerrainMapRequest,
};

/// How the camera and the belief of a trial differ from the map and the
/// truth.
#[derive(Clone, Copy)]
struct Trial<'a> {
    /// The package the navigator loads.
    map: &'a World,
    /// The ground the camera sees.
    truth: &'a World,
    mismatch: Mismatch,
    /// Error of the believed heading and of the believed roll, radians;
    /// the belief states each as its one-sigma.
    heading_rad: f64,
    tilt_rad: f64,
}

impl<'a> Trial<'a> {
    fn exact(world: &'a World) -> Self {
        Self {
            map: world,
            truth: world,
            mismatch: Mismatch::EXACT,
            heading_rad: 0.0,
            tilt_rad: 0.0,
        }
    }
}

/// The outcome of each of `frames` frames, one a second, of the demo
/// camera hovering 1760 m above the ground, and the horizontal error of
/// the solution after each, with the belief `off` metres (north, east)
/// from the truth and a one-sigma of `sigma_m` on each axis.
fn first_frames(trial: Trial<'_>, off: [f64; 2], sigma_m: f64, frames: u64) -> Vec<(Outcome, f64)> {
    let hover = Flight {
        start: [CENTRE[0], CENTRE[1], GROUND_M + 1_760.0],
        speed_m_s: 0.0,
    };
    let mount = CameraMount::nadir();
    let mut navigator = ReferenceNavigator::new(suite(imager(), mount), CLOCK, None).unwrap();
    let [lat, lon, h] = hover.start;
    let variance = sigma_m * sigma_m;
    let believed = UnitQuaternion::from_euler_angles(trial.tilt_rad, 0.0, trial.heading_rad);
    let q = believed.quaternion();
    let tilt_var = trial.tilt_rad.abs().max(1e-3).powi(2);
    let heading_var = trial.heading_rad.abs().max(3e-3).powi(2);
    navigator
        .align(&Alignment {
            position: GeodeticPosition::new(
                lat.to_radians() + off[0] / 6_364_000.0,
                lon.to_radians() + off[1] / 4_760_000.0,
                h + 10.0,
            ),
            velocity: NedVelocity::new(0.0, 0.0, 0.0),
            attitude: AttitudeQuaternion::new(q.w, q.i, q.j, q.k),
            position_cov: SymmetricCov3::from_diagonal(variance, variance, 400.0),
            velocity_cov: SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            attitude_cov: SymmetricCov3::from_diagonal(tilt_var, tilt_var, heading_var),
            at: at(0),
            imu_bias_prior: None,
        })
        .unwrap();
    let request = TerrainMapRequest::new(trial.map.package, TODAY);
    navigator
        .load_terrain_map(Arc::clone(&trial.map.store), &request, None)
        .unwrap();
    let mut sink = Vec::new();
    let level = UnitQuaternion::identity();
    let mut seen = Vec::new();
    for frame in 1..=frames {
        for step in (frame - 1) * RATE + 1..=frame * RATE {
            navigator.ingest(&hover.increment(step), &mut sink).unwrap();
        }
        let step = frame * RATE;
        let rendered = frame_at(
            trial.truth,
            &camera(),
            hover.body(step),
            &level,
            &mount,
            step,
        );
        let luma = trial.mismatch.apply(&rendered, 400, 300, step);
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
            let seen = first_frames(Trial::exact(&world), off, 1_000.0, 2);
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
                for (outcome, error_m) in first_frames(Trial::exact(&world), off, 1_000.0, 3) {
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

/// Outcomes of the frames of a statistics run.
#[derive(Default, Debug)]
struct Tally {
    frames: u32,
    attempts: u32,
    fixes: u32,
    wrong: u32,
    refusals: std::collections::BTreeMap<String, u32>,
}

impl Tally {
    /// Counts the frames of one run: a frame is an acquisition attempt
    /// until the run's first accepted fix.
    fn add(&mut self, seen: &[(Outcome, f64)], label: &str) {
        let mut wide = true;
        for (outcome, error_m) in seen {
            self.frames += 1;
            if wide {
                self.attempts += 1;
            }
            match outcome {
                Outcome::Accepted => {
                    self.fixes += 1;
                    wide = false;
                    if *error_m > 30.0 {
                        self.wrong += 1;
                        tracing::warn!(label, error_m, "a wrong acquisition");
                    }
                }
                other => *self.refusals.entry(format!("{other:?}")).or_default() += 1,
            }
        }
    }
}

/// The acquisition under a believed heading off by 2° and 5° and a
/// believed roll off by 0.5° and 1°, and under the photometric, optical
/// and seasonal mismatch of the geometry grid, over farmland and regular
/// and irregular town blocks, with the truth 0.5 km, 2 km and 4 km from
/// the belief: no accepted fix is more than 30 m off.
#[test]
#[ignore = "statistics over some 500 frames; minutes in an optimized build"]
fn the_full_acquisition_grid_gives_no_wrong_fix() {
    show_events();
    let deg = std::f64::consts::PI / 180.0;
    let mut total = Tally::default();
    for texture in [Texture::Farmland, Texture::Town, Texture::TownIrregular] {
        let season = (texture == Texture::Farmland).then(|| world(Texture::FarmlandSeason));
        let world = world(texture);
        let mut trials: Vec<(String, Trial<'_>)> =
            [(2.0, 0.5), (-2.0, 1.0), (5.0, 0.5), (-5.0, 1.0)]
                .iter()
                .map(|&(heading, tilt)| {
                    let trial = Trial {
                        heading_rad: heading * deg,
                        tilt_rad: tilt * deg,
                        ..Trial::exact(&world)
                    };
                    (format!("heading {heading} tilt {tilt}"), trial)
                })
                .collect();
        for (name, mismatch) in [
            ("photometric", Mismatch::PHOTOMETRIC),
            ("all", Mismatch::ALL),
        ] {
            trials.push((
                name.into(),
                Trial {
                    mismatch,
                    ..Trial::exact(&world)
                },
            ));
        }
        if let Some(season) = &season {
            let trial = Trial {
                truth: season,
                mismatch: Mismatch::ALL,
                ..Trial::exact(&world)
            };
            trials.push(("season".into(), trial));
        }
        for (name, trial) in &trials {
            let mut tally = Tally::default();
            for magnitude in [500.0, 2_000.0, 4_000.0] {
                for off in around(magnitude, 3) {
                    let seen = first_frames(*trial, off, 1_000.0, 3);
                    tally.add(&seen, name);
                    total.add(&seen, name);
                }
            }
            tracing::info!(?texture, name, ?tally, "acquisition grid");
        }
    }
    tracing::info!(?total, "acquisition grid total");
    assert_eq!(total.wrong, 0, "{total:?}");
}
