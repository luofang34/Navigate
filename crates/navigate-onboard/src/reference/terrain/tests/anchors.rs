//! Synthetic terrain anchors through the admission of the navigator: the
//! 3-D position NEES over many seeds of a cruise across map cells and of
//! a loiter in one cell, with the map error of each cell bound as a
//! constant correlated error.
//!
//! The anchors are not matched images. Each anchor is the true position
//! plus the errors that a matched frame has: the registration and the
//! terrain height of its map cell, constant in the cell; the relief term
//! of the same height error seen at the off-nadir angle along the view;
//! a constant calibration angle of the camera times the range, which
//! turns with the heading; and white matcher noise. The real matcher is
//! tested in `fixes`.

use nalgebra::Vector3;
use navigate_camera::CameraMount;
use navigate_contract::{SensorClass, SourceComposition, SymmetricCov3};
use navigate_fusion::{FogmBias, Observation, ObservationValue};
use navigate_visual::session::{ContinuityEpoch, FrameKey, MapCell, StreamId};

use super::flight::{CAMERA, imager};
use crate::reference::terrain::with_map_error;
use crate::reference::tests::truth::{
    CLOCK, Draws, IMU, Imu, STEP_S, Truth, at, imu_budget, mix, moved, ned_offset, nees, start,
};
use crate::sensor::SensorSpec;
use crate::{
    ErrorBudget, Navigator, Outcome, ReferenceNavigator, SensorInput, SensorKind, SensorSuite,
};

const SEEDS: u32 = if cfg!(debug_assertions) { 1 } else { 12 };
const DURATION_S: f64 = if cfg!(debug_assertions) { 60.0 } else { 900.0 };
const REPORTS_S: &[f64] = if cfg!(debug_assertions) {
    &[30.0, 60.0]
} else {
    &[300.0, 600.0, 900.0]
};
/// The 0.1 % and 99.9 % points of χ²(3·SEEDS)/SEEDS.
const BAND: (f64, f64) = if cfg!(debug_assertions) {
    (0.024, 16.3)
} else {
    (1.28, 5.67)
};
/// Anchors at 2 Hz.
const ANCHOR_STEPS: u32 = 5;
const CELL_M: f64 = 500.0;
/// The map: registration on each horizontal axis and terrain height.
const MAP_SIGMA_M: [f64; 3] = [2.5, 2.5, 4.0];
const OFF_NADIR_RAD: f64 = 0.436;
const RANGE_M: f64 = 330.0;
const CALIBRATION_RAD: f64 = 1e-3;
const MATCHER_M: f64 = 1.0;

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
        mount: Some(CameraMount::zenith()),
        rate_hz: 2.0,
        budget: ErrorBudget::TerrainCamera {
            calibration_sigma_rad: CALIBRATION_RAD,
        },
    };
    SensorSuite::new(vec![imu, camera]).unwrap()
}

/// The cell of a NED offset from the start, and its column and row.
fn cell(offset: &Vector3<f64>) -> (MapCell, [i64; 2]) {
    let cell = MapCell::local("synthetic", offset.y + 50.0, offset.x + 250.0, CELL_M);
    let index = cell.local_index().unwrap();
    (cell, index)
}

/// The map error of a cell for one seed: registration and height.
fn map_error(seed: u32, [east, north]: [i64; 2]) -> Vector3<f64> {
    let key = mix(u64::from(seed) ^ (east as u64).wrapping_mul(0x9E37) ^ (north as u64) << 32);
    Draws {
        seed: key,
        index: 0,
    }
    .vec3(MAP_SIGMA_M)
}

/// The white covariance of an anchor seen along `heading`: the matcher,
/// the calibration, and the relief term of the terrain height.
fn white_covariance(heading: f64) -> SymmetricCov3 {
    let relief = MAP_SIGMA_M[2] * OFF_NADIR_RAD.tan();
    let along = Vector3::new(heading.cos(), heading.sin(), 0.0) * relief;
    let m = nalgebra::Matrix3::identity()
        * (MATCHER_M.powi(2) + (CALIBRATION_RAD * RANGE_M).powi(2))
        + along * along.transpose();
    SymmetricCov3::from_upper_triangle([
        m[(0, 0)],
        m[(0, 1)],
        m[(0, 2)],
        m[(1, 1)],
        m[(1, 2)],
        m[(2, 2)],
    ])
}

/// One run: the NEES at each report time and the count of anchors that
/// the filter took.
fn run(seed: u32, turn: fn(f64) -> f64) -> (Vec<f64>, u32) {
    let mut draws = Draws {
        seed: u64::from(seed),
        index: 0,
    };
    let mut truth = Truth::new(turn);
    let mut navigator = Box::new(ReferenceNavigator::new(suite(), CLOCK, None).unwrap());
    let imu = Imu::align(&mut navigator, &mut draws);
    let calibration = draws.vec3([CALIBRATION_RAD, CALIBRATION_RAD, 0.0]) * RANGE_M;
    let (mut out, mut taken) = (Vec::new(), 0_u32);
    let steps = (DURATION_S / STEP_S).round() as u32;
    for step in 1..=steps {
        imu.step(&mut truth, &mut navigator, &mut draws, step);
        if step % ANCHOR_STEPS == 0 {
            let offset = ned_offset(&start(), &truth.state.position);
            let (cell, index) = cell(&offset);
            let v = truth.state.velocity;
            let heading = v.east_mps.atan2(v.north_mps);
            let (sin, cos) = heading.sin_cos();
            let map = map_error(seed, index);
            let relief = Vector3::new(cos, sin, 0.0) * map.z * OFF_NADIR_RAD.tan();
            let turned = Vector3::new(
                cos * calibration.x - sin * calibration.y,
                sin * calibration.x + cos * calibration.y,
                0.0,
            );
            let error = map + relief + turned + draws.vec3([MATCHER_M; 3]);
            let input = SensorInput {
                sensor: CAMERA,
                acquired_at: at(step),
                delivered_at: at(step),
                data: crate::SensorData::StarExposureMid,
            };
            let stamp = navigator.stamp(CAMERA, at(step));
            let value = |covariance| ObservationValue::PositionFix {
                position: moved(&truth.state.position, &error),
                covariance,
            };
            let composition = SourceComposition::of(SensorClass::VisualLandmark);
            let frame = FrameKey {
                stream: StreamId(CAMERA.get()),
                continuity: ContinuityEpoch(0),
                index: u64::from(step),
            };
            let observation =
                Observation::new(stamp, value(white_covariance(heading)), composition);
            let observation = with_map_error(observation, &cell, MAP_SIGMA_M);
            let accepted =
                navigator.offer_terrain_fix(&observation, cell, frame, &input) == Outcome::Accepted;
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

/// The mean NEES at each report time and the mean count of anchors taken.
fn mean(turn: fn(f64) -> f64) -> (Vec<f64>, f64) {
    let mut sum = vec![0.0; REPORTS_S.len()];
    let mut taken = 0.0;
    for seed in 0..SEEDS {
        let (nees, count) = run(seed, turn);
        for (total, value) in sum.iter_mut().zip(nees) {
            *total += value;
        }
        taken += f64::from(count);
    }
    let n = f64::from(SEEDS);
    (sum.into_iter().map(|total| total / n).collect(), taken / n)
}

/// A cruise with two 90° turns: the anchors cross many cells.
fn cruise(t_s: f64) -> f64 {
    let rate = 0.114;
    let turn = core::f64::consts::FRAC_PI_2 / rate;
    [(200.0, 1.0), (500.0, -1.0)]
        .iter()
        .find(|(begin, _)| (*begin..begin + turn).contains(&t_s))
        .map_or(0.0, |(_, sign)| sign * rate)
}

/// A loiter on a circle of 200 m: the anchors stay in one cell.
fn loiter(_: f64) -> f64 {
    0.2
}

fn in_band(nees: &[f64]) -> bool {
    nees.iter().all(|v| (BAND.0..=BAND.1).contains(v))
}

#[test]
fn monte_carlo_a_cruise_across_cells_with_bound_map_errors_keeps_the_position_nees_in_band() {
    let (nees, taken) = mean(cruise);
    assert!(in_band(&nees), "NEES {nees:?}");
    // Each anchor of the flight is taken.
    let anchors = (DURATION_S / STEP_S).round() / f64::from(ANCHOR_STEPS);
    assert!(taken >= 0.99 * anchors, "{taken} of {anchors}");
}

#[test]
fn monte_carlo_a_loiter_in_one_cell_with_its_bound_map_error_keeps_the_position_nees_in_band() {
    // The one-fix-for-each-cell admission takes one anchor here; the
    // bound map error lets the filter take each of them.
    let (nees, taken) = mean(loiter);
    assert!(in_band(&nees), "NEES {nees:?}");
    let anchors = (DURATION_S / STEP_S).round() / f64::from(ANCHOR_STEPS);
    assert!(taken >= 0.99 * anchors, "{taken} of {anchors}");
}

#[test]
fn a_cell_whose_block_a_newer_cell_took_is_withheld_before_the_filter() {
    let mut draws = Draws { seed: 1, index: 0 };
    let mut truth = Truth::new(cruise);
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
                covariance: white_covariance(0.0),
            },
            SourceComposition::of(SensorClass::VisualLandmark),
        );
        let observation = with_map_error(observation, &cell, MAP_SIGMA_M);
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
    // cell reads it, and the first cell is withheld as before.
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
