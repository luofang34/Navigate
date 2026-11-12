//! The first fixes after a long gap: a belief that dead reckoning left
//! kilometres uncertain and off finds the ground again by acquisition,
//! and an acquisition gives no fix at a wrong place.

use std::sync::Arc;

use nalgebra::{Matrix2, UnitQuaternion, Vector3};
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
/// What one frame did: its outcome, and the horizontal error of the
/// solution after it and that error's squared Mahalanobis length under
/// the stated horizontal covariance.
#[derive(Debug, Clone, PartialEq)]
struct Seen {
    outcome: Outcome,
    error_m: f64,
    nees: f64,
}

/// The squared Mahalanobis length of the north and east `error` under the
/// north and east block of `cov`.
fn horizontal_nees(error: Vector3<f64>, cov: &SymmetricCov3) -> f64 {
    let [nn, ne, _, ee, _, _] = cov.upper_triangle();
    Matrix2::new(nn, ne, ne, ee)
        .try_inverse()
        .map_or(f64::INFINITY, |inv| {
            (error.xy().transpose() * inv * error.xy()).x
        })
}

fn first_frames(trial: Trial<'_>, off: [f64; 2], sigma_m: f64, frames: u64) -> Vec<Seen> {
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
        let solution = navigator.solution(at(step)).unwrap();
        let error = hover.error_m(&solution, step);
        seen.push(Seen {
            outcome,
            error_m: error.xy().norm(),
            nees: horizontal_nees(error, &solution.position_cov),
        });
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

/// Frames that an acquisition takes at most after a long gap, at the
/// default budget of a frame: the map raster is built a block a frame,
/// the search of a frame runs over two or three frames, and a second
/// search confirms the first.
const FRAMES: u64 = 16;

/// From 1760 m over farmland, a belief of 1 km one-sigma that is 0.5 to
/// 2 km off is found again within [`FRAMES`] frames: an acquisition first
/// waits for a second one to confirm it, and only the second gives the
/// filter its fix.
#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn a_belief_kilometres_off_finds_the_ground_again() {
    show_events();
    let world = world(Texture::Farmland);
    for magnitude in [500.0, 1_000.0, 2_000.0] {
        for off in around(magnitude, 2) {
            let seen = first_frames(Trial::exact(&world), off, 1_000.0, FRAMES);
            let outcomes: Vec<_> = seen.iter().map(|s| &s.outcome).collect();
            tracing::info!(?off, ?outcomes, "the first frames on a wide belief");
            let first = seen.iter().position(|s| s.outcome == Outcome::Accepted);
            let Some(first) = first else {
                panic!("{off:?}: {outcomes:?}");
            };
            let waited = Outcome::Unused(ACQUISITION_UNCONFIRMED);
            assert!(seen[..first].iter().any(|s| s.outcome == waited), "{off:?}");
            assert!(seen[first].error_m < 20.0, "{off:?}: {:?}", seen[first]);
        }
    }
}

/// Over farmland, over regular and irregular town blocks, and with the
/// truth past the acquisition radius and near the edge of the map, no
/// acquisition gives the filter a fix at a wrong place or one whose error
/// its covariance does not hold.
#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn an_acquisition_gives_no_fix_at_a_wrong_place() {
    show_events();
    let mut total = Tally::default();
    for texture in [Texture::Farmland, Texture::Town, Texture::TownIrregular] {
        let world = world(texture);
        for magnitude in [500.0, 4_000.0] {
            for off in around(magnitude, 2) {
                let seen = first_frames(Trial::exact(&world), off, 1_000.0, FRAMES);
                total.add(&seen, &format!("{texture:?}"));
            }
        }
    }
    tracing::info!(?total, "acquisitions");
    assert_eq!((total.wrong_place, total.inconsistent), (0, 0), "{total:?}");
}

/// Squared Mahalanobis length of a two-dimensional error that a
/// consistent covariance exceeds once in a thousand.
const NEES_999: f64 = 13.82;

/// Outcomes of the frames of a statistics run. An accepted fix is at a
/// wrong place when it is farther from the truth than the exclusion
/// distance of the acquisition, and inconsistent when its error is past
/// [`NEES_999`] under the covariance it stated; errors past 30 m are
/// counted as well.
#[derive(Default, Debug)]
struct Tally {
    frames: u32,
    attempts: u32,
    fixes: u32,
    wrong_place: u32,
    inconsistent: u32,
    past_30_m: u32,
    refusals: std::collections::BTreeMap<String, u32>,
}

impl Tally {
    /// Counts the frames of one run: a frame is an acquisition attempt
    /// until the run's first accepted fix.
    fn add(&mut self, seen: &[Seen], label: &str) {
        let mut wide = true;
        for frame in seen {
            self.frames += 1;
            if wide {
                self.attempts += 1;
            }
            if frame.outcome != Outcome::Accepted {
                let name = format!("{:?}", frame.outcome);
                *self.refusals.entry(name).or_default() += 1;
                continue;
            }
            self.fixes += 1;
            wide = false;
            let wrong = frame.error_m > 300.0;
            let inconsistent = frame.nees > NEES_999;
            self.wrong_place += u32::from(wrong);
            self.inconsistent += u32::from(inconsistent);
            self.past_30_m += u32::from(frame.error_m > 30.0);
            if wrong || inconsistent || frame.error_m > 30.0 {
                tracing::warn!(label, ?frame, "an acquired fix far off");
            }
        }
    }
}

/// The acquisition under a believed heading off by 2° and 5° and a
/// believed roll off by 0.5° and 1°, and under the photometric, optical
/// and seasonal mismatch of the geometry grid, over farmland and regular
/// and irregular town blocks, with the truth 0.5 km, 2 km and 4 km from
/// the belief: no accepted fix is at a wrong place or past what its
/// covariance holds.
#[test]
#[ignore = "statistics over some 2700 frames; minutes in an optimized build"]
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
                    let seen = first_frames(*trial, off, 1_000.0, FRAMES);
                    tally.add(&seen, name);
                    total.add(&seen, name);
                }
            }
            tracing::info!(?texture, name, ?tally, "acquisition grid");
        }
    }
    tracing::info!(?total, "acquisition grid total");
    assert_eq!((total.wrong_place, total.inconsistent), (0, 0), "{total:?}");
}
