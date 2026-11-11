//! Synthetic terrain anchors through the admission of the navigator: the
//! 3-D position NEES over many seeds of flights over a random elevation
//! error, with one fix for each map cell and with the map error of each
//! cell bound as a constant correlated error.
//!
//! The anchors are not matched images. Each anchor is the true position
//! plus the errors that a matched frame has: the registration of its
//! imagery cell; the tilt of the elevation error under the footprint
//! times the height above the ground, and its mean on the vertical, which
//! the real matcher follows (`tilt`); and white matcher noise.

mod ground;

use nalgebra::Vector3;
use navigate_camera::CameraMount;
use navigate_contract::{SensorClass, SourceComposition, SymmetricCov3};
use navigate_fusion::{FogmBias, Observation, ObservationValue};
use navigate_visual::session::{ContinuityEpoch, FrameKey, MapCell, StreamId};

use super::flight::{CAMERA, imager, show_events};
use crate::reference::terrain::budget::{MapSigmas, ViewGeometry, tilt_sigma_m};
use crate::reference::terrain::with_map_error;
use crate::reference::tests::truth::{
    CLOCK, Draws, IMU, Imu, STEP_S, Truth, at, imu_budget, mix, moved, ned_offset, nees, start,
};
use crate::sensor::SensorSpec;
use crate::{
    ErrorBudget, Navigator, Outcome, ReferenceNavigator, SensorInput, SensorKind, SensorSuite,
};
use ground::Field;

const SEEDS: u32 = 12;
const DURATION_S: f64 = 900.0;
const REPORTS_S: &[f64] = &[300.0, 600.0, 900.0];
/// The 0.1 % and 99.9 % points of χ²(3·SEEDS)/SEEDS.
const BAND: (f64, f64) = (1.28, 5.67);
/// Anchors at 2 Hz.
const ANCHOR_STEPS: u32 = 5;
/// Imagery cells of zoom 16 at the latitude of the tests.
const CELL_M: f64 = 470.0;
/// Registration of the imagery, each horizontal axis.
const REGISTRATION_M: f64 = 1.5;
/// The elevation error: one-sigma and correlation length.
const DEM_SIGMA_M: f64 = 5.0;
const DEM_LENGTH_M: f64 = 300.0;

fn suite() -> SensorSuite {
    let (accel, gyro, accel_bias, gyro_bias): (f64, f64, FogmBias, FogmBias) = imu_budget();
    let imu = SensorSpec {
        id: IMU,
        kind: SensorKind::Imu,
        imager: None,
        mount: None,
        rate_hz: 10.0,
        budget: ErrorBudget::Imu {
            accel_noise_mps2_sqrt_hz: accel,
            gyro_noise_rad_s_sqrt_hz: gyro,
            accel_bias,
            gyro_bias,
        },
    };
    let camera = SensorSpec {
        id: CAMERA,
        kind: SensorKind::TerrainCamera,
        imager: Some(imager()),
        mount: Some(CameraMount::nadir()),
        rate_hz: 2.0,
        budget: ErrorBudget::TerrainCamera {
            calibration_sigma_rad: 0.0,
        },
    };
    SensorSuite::new(vec![imu, camera]).unwrap()
}

/// The imagery cell of a NED offset from the start, and its column and row.
fn cell(offset: &Vector3<f64>) -> (MapCell, [i64; 2]) {
    let cell = MapCell::local("synthetic", offset.y + 50.0, offset.x + 250.0, CELL_M);
    let index = cell.local_index().unwrap();
    (cell, index)
}

/// The registration of a cell for one seed.
fn registration(seed: u32, [east, north]: [i64; 2]) -> Vector3<f64> {
    let key = mix(u64::from(seed) ^ (east as u64).wrapping_mul(0x9E37) ^ (north as u64) << 32);
    Draws {
        seed: key,
        index: 0,
    }
    .vec3([REGISTRATION_M, REGISTRATION_M, 0.0])
}

/// The white one-sigma of the matcher at `agl_m`, metres.
fn matcher_m(agl_m: f64) -> f64 {
    (0.0025 * agl_m).max(1.0)
}

/// The tilt term that the budget states for a nadir fix from `agl_m`.
fn tilt_m(agl_m: f64) -> f64 {
    let map = MapSigmas {
        horizontal_m: REGISTRATION_M,
        height_m: DEM_SIGMA_M,
        elevation_m: DEM_SIGMA_M,
        elevation_length_m: DEM_LENGTH_M,
    };
    let view = ViewGeometry {
        off_nadir_rad: 0.0,
        ground_direction: [0.0, 0.0],
        range_m: agl_m,
        field_ratio: 0.8,
    };
    tilt_sigma_m(&map, &view)
}

/// How the navigator takes the anchors.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Admission {
    /// One fix for each imagery cell, its map error declared white.
    OneForEachCell,
    /// Each fix, the map error of its cell bound as a correlated error.
    Keyed,
    /// Each fix, the map error and the tilt of its cell bound together.
    KeyedWithTilt,
}

/// A flight: its turn rate at each time and its height above the ground.
#[derive(Debug, Clone, Copy)]
struct Scenario {
    name: &'static str,
    turn: fn(f64) -> f64,
    agl_m: f64,
}

/// A cruise with two 90° turns.
fn cruise(t_s: f64) -> f64 {
    let rate = 0.114;
    let turn = core::f64::consts::FRAC_PI_2 / rate;
    [(200.0, 1.0), (500.0, -1.0)]
        .iter()
        .find(|(begin, _)| (*begin..begin + turn).contains(&t_s))
        .map_or(0.0, |(_, sign)| sign * rate)
}

/// A circle of 600 m at 40 m/s.
fn loiter(_: f64) -> f64 {
    40.0 / 600.0
}

/// A racetrack: legs of 60 s and turns of 180° in 60 s.
fn racetrack(t_s: f64) -> f64 {
    if t_s.rem_euclid(120.0) < 60.0 {
        0.0
    } else {
        core::f64::consts::PI / 60.0
    }
}

/// Straight and level.
fn straight(_: f64) -> f64 {
    0.0
}

/// One run: the NEES at each report time and the count of anchors that
/// the filter took.
fn run(seed: u32, scenario: Scenario, admission: Admission) -> (Vec<f64>, u32) {
    let mut draws = Draws {
        seed: u64::from(seed),
        index: 0,
    };
    let field = Field::new(u64::from(seed), DEM_SIGMA_M, DEM_LENGTH_M);
    let mut truth = Truth::new(scenario.turn);
    let mut navigator = Box::new(ReferenceNavigator::new(suite(), CLOCK, None).unwrap());
    let imu = Imu::align(&mut navigator, &mut draws);
    let agl_m = scenario.agl_m;
    let (matcher, tilt) = (matcher_m(agl_m), tilt_m(agl_m));
    let white = |extra: [f64; 3]| {
        let [n, e, d] = extra;
        let h = matcher * matcher + tilt * tilt;
        SymmetricCov3::from_diagonal(h + n * n, h + e * e, matcher * matcher + d * d)
    };
    let mut seen = std::collections::BTreeSet::new();
    let (mut out, mut taken) = (Vec::new(), 0_u32);
    let steps = (DURATION_S / STEP_S).round() as u32;
    for step in 1..=steps {
        imu.step(&mut truth, &mut navigator, &mut draws, step);
        if step % ANCHOR_STEPS == 0 {
            let offset = ned_offset(&start(), &truth.state.position);
            let (cell, index) = cell(&offset);
            let v = truth.state.velocity;
            let heading = v.east_mps.atan2(v.north_mps);
            let error = registration(seed, index)
                + field.fix_error([offset.x, offset.y], heading, agl_m)
                + draws.vec3([matcher; 3]);
            let position = moved(&truth.state.position, &error);
            let stamp = navigator.stamp(CAMERA, at(step));
            let composition = SourceComposition::of(SensorClass::VisualLandmark);
            let input = SensorInput {
                sensor: CAMERA,
                acquired_at: at(step),
                delivered_at: at(step),
                data: crate::SensorData::StarExposureMid,
            };
            let frame = FrameKey {
                stream: StreamId(CAMERA.get()),
                continuity: ContinuityEpoch(0),
                index: u64::from(step),
            };
            let accepted = match admission {
                Admission::Keyed | Admission::KeyedWithTilt => {
                    let bound = admission == Admission::KeyedWithTilt;
                    let in_block = if bound { tilt } else { 0.0 };
                    let h = matcher * matcher + (tilt * tilt - in_block * in_block);
                    let covariance = SymmetricCov3::from_diagonal(h, h, matcher * matcher);
                    let value = ObservationValue::PositionFix {
                        position,
                        covariance,
                    };
                    let observation = Observation::new(stamp, value, composition);
                    let horizontal = REGISTRATION_M.hypot(in_block);
                    let map = [horizontal, horizontal, DEM_SIGMA_M];
                    let observation = with_map_error(observation, &cell, map);
                    navigator.offer_terrain_fix(&observation, cell, frame, &input)
                        == Outcome::Accepted
                }
                Admission::OneForEachCell => {
                    if seen.insert(index) {
                        let value = ObservationValue::PositionFix {
                            position,
                            covariance: white([REGISTRATION_M, REGISTRATION_M, DEM_SIGMA_M]),
                        };
                        let observation = Observation::new(stamp, value, composition);
                        navigator.offer_terrain_fix(&observation, cell, frame, &input)
                            == Outcome::Accepted
                    } else {
                        false
                    }
                }
            };
            taken = taken.wrapping_add(u32::from(accepted));
        }
        if REPORTS_S
            .iter()
            .any(|r| ((r / STEP_S).round() as u32) == step)
        {
            let solution = navigator.solution(at(step)).unwrap();
            let error = ned_offset(&truth.state.position, &solution.position);
            out.push(nees(&error, &solution.position_cov));
        }
    }
    (out, taken)
}

/// The mean NEES at each report time over the seeds `first..first +
/// SEEDS`, and the mean count of anchors taken.
fn mean(first: u32, scenario: Scenario, admission: Admission) -> (Vec<f64>, f64) {
    let mut sum = vec![0.0; REPORTS_S.len()];
    let mut taken = 0.0;
    for seed in first..first + SEEDS {
        let (nees, count) = run(seed, scenario, admission);
        for (total, value) in sum.iter_mut().zip(nees) {
            *total += value;
        }
        taken += f64::from(count);
    }
    let n = f64::from(SEEDS);
    (sum.into_iter().map(|total| total / n).collect(), taken / n)
}

const SCENARIOS: [Scenario; 6] = [
    Scenario {
        name: "cruise 400 m",
        turn: cruise,
        agl_m: 400.0,
    },
    Scenario {
        name: "cruise 1760 m",
        turn: cruise,
        agl_m: 1_760.0,
    },
    Scenario {
        name: "loiter 400 m",
        turn: loiter,
        agl_m: 400.0,
    },
    Scenario {
        name: "loiter 1760 m",
        turn: loiter,
        agl_m: 1_760.0,
    },
    Scenario {
        name: "racetrack 400 m",
        turn: racetrack,
        agl_m: 400.0,
    },
    Scenario {
        name: "racetrack 1760 m",
        turn: racetrack,
        agl_m: 1_760.0,
    },
];

#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn explore_anchor_baselines() {
    show_events();
    let only = std::env::var("ONLY").unwrap_or_default();
    for scenario in SCENARIOS {
        if !only.is_empty() && !scenario.name.contains(&only) {
            continue;
        }
        for admission in [
            Admission::OneForEachCell,
            Admission::Keyed,
            Admission::KeyedWithTilt,
        ] {
            if std::env::var("ADM").is_ok_and(|a| !format!("{admission:?}").contains(&a)) {
                continue;
            }
            for first in [0, 12] {
                let (nees, taken) = mean(first, scenario, admission);
                let inside = nees.iter().all(|v| (BAND.0..=BAND.1).contains(v));
                tracing::warn!(
                    "ANCHORS {} {admission:?} seeds {first}-{} nees={nees:.2?} taken={taken:.0} in_band={inside}",
                    scenario.name,
                    first + SEEDS - 1
                );
            }
        }
    }
}

#[test]
fn a_cell_whose_block_a_newer_cell_took_is_withheld_before_the_filter() {
    let mut draws = Draws { seed: 1, index: 0 };
    let mut truth = Truth::new(straight);
    let mut navigator = Box::new(ReferenceNavigator::new(suite(), CLOCK, None).unwrap());
    let imu = Imu::align(&mut navigator, &mut draws);
    let cells = [[0, 0], [0, 1], [0, 1], [0, 0]];
    let mut outcomes = Vec::new();
    for (k, [east, north]) in cells.into_iter().enumerate() {
        let step = u32::try_from(k).unwrap() + 1;
        imu.step(&mut truth, &mut navigator, &mut draws, step);
        let cell = MapCell::Local {
            map_manifest_sha256: "synthetic".into(),
            east,
            north,
        };
        let stamp = navigator.stamp(CAMERA, at(step));
        let observation = Observation::new(
            stamp,
            ObservationValue::PositionFix {
                position: truth.state.position,
                covariance: SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
            },
            SourceComposition::of(SensorClass::VisualLandmark),
        );
        let observation = with_map_error(
            observation,
            &cell,
            [REGISTRATION_M, REGISTRATION_M, DEM_SIGMA_M],
        );
        let frame = FrameKey {
            stream: StreamId(CAMERA.get()),
            continuity: ContinuityEpoch(0),
            index: u64::from(step),
        };
        let input = SensorInput {
            sensor: CAMERA,
            acquired_at: at(step),
            delivered_at: at(step),
            data: crate::SensorData::StarExposureMid,
        };
        outcomes.push(navigator.offer_terrain_fix(&observation, cell, frame, &input));
    }
    // The second cell takes the one map block; a later fix of the second
    // cell reads it, and a fix of the first cell, whose error the filter
    // no longer holds, is withheld before the filter.
    assert_eq!(
        outcomes,
        [
            Outcome::Accepted,
            Outcome::Accepted,
            Outcome::Accepted,
            Outcome::Unused("shared_map_error"),
        ]
    );
}
