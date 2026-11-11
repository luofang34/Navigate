//! Terrain fixes end to end: the position against the truth, the path of
//! each reference, and what the filter is told about each map cell.

use nalgebra::Vector3;
use navigate_camera::CameraMount;
use navigate_contract::MonotonicNanos;
use navigate_fusion::RejectionReason;

use super::flight::{CAMERA, Flight, RATE, at, fly, load, navigator, request, show_events};
use super::world::{CENTRE, GROUND_M, World, WorldSpec, camera_model, correlation_cell};
use crate::reference::terrain::MEASUREMENT;
use crate::{AcceptanceRecord, BeliefSource, Navigator, Outcome, TerrainConfig, TerrainMapRequest};

/// 800 m above the mean ground.
pub(crate) const HEIGHT_M: f64 = GROUND_M + 800.0;

/// The outcome and the path of each terrain frame.
pub(crate) fn terrain(records: &[AcceptanceRecord]) -> Vec<(Outcome, Option<&'static str>)> {
    records
        .iter()
        .filter(|r| r.sensor == CAMERA && r.measurement == MEASUREMENT)
        .map(|r| (r.outcome.clone(), r.context))
        .collect()
}

/// A nadir camera one metre behind and half a metre below the reference
/// point.
pub(crate) fn belly() -> CameraMount {
    CameraMount::nadir().with_lever_arm(Vector3::new(-1.0, 0.0, 0.5))
}

/// A flight to the north at a third of a correlation cell for each half
/// second, so that the frames at each 25th step see the ground at one
/// sixth, one half, and five sixths of each cell: no frame is near a cell
/// boundary.
pub(crate) fn across_cells() -> Flight {
    // The belly camera is one metre behind the reference point.
    north_through_cells(super::world::CORRELATION_ZOOM, 3.0, HEIGHT_M, -1.0)
}

/// A flight to the north at `height_m` from just south of a cell of
/// `zoom`, at one cell for `frames` frames of each half second. The
/// ground that the camera sees is `ahead_m` north of the reference point,
/// and it is half a frame step into the cell at the first frame.
pub(crate) fn north_through_cells(zoom: u32, frames: f64, height_m: f64, ahead_m: f64) -> Flight {
    let ([lat, lon], cell_m) = correlation_cell(CENTRE, zoom);
    let south_edge = lat - (cell_m / 2.0 / 6_364_000.0_f64).to_degrees();
    let step_m = cell_m / frames;
    let start = south_edge - ((ahead_m + step_m / 2.0) / 6_364_000.0_f64).to_degrees();
    Flight {
        start: [start, lon, height_m],
        speed_m_s: step_m / 0.5,
    }
}

const SHARED: Outcome = Outcome::Unused("shared_map_error");

/// Bound on the error of the solution after a fix, on each of the
/// horizontal and the vertical, metres: three pixels of the test camera,
/// which are 5.3 m on the ground from 800 m. Over the poses of
/// [`the_declared_sigma_holds_the_error_and_is_of_its_order`] the
/// horizontal error is 3.7 m root mean square and 8.5 m at most.
const FIX_BOUND_M: f64 = 16.0;

#[test]
fn one_fix_pulls_a_hovering_vehicle_to_the_truth_through_an_oblique_lever_armed_camera() {
    show_events();
    let world = World::build(&WorldSpec::default());
    let hover = Flight {
        start: [CENTRE[0], CENTRE[1], HEIGHT_M],
        speed_m_s: 0.0,
    };
    // The camera looks 20° off the nadir, to the right of the nose and
    // rolled, from 2 m ahead of and 0.9 m below the reference point.
    let mount = CameraMount::rotated_forward(0.1, -70.0_f64.to_radians(), 0.4)
        .with_lever_arm(Vector3::new(2.0, -0.7, 0.9));
    // Aligned 35 m north, 25 m west, and 12 m above the truth.
    let mut navigator = navigator(&hover, mount, [35.0, -25.0, 12.0], 40.0);
    let report = load(&mut navigator, &world);
    let mut sink = Vec::new();
    let each_second = |step: u64| step.is_multiple_of(RATE);
    fly(
        &mut navigator,
        &hover,
        &world,
        &mount,
        1..=3 * RATE,
        each_second,
        &mut sink,
    )
    .unwrap();
    // The first frame is the first fix of its map cell. The vehicle does
    // not move, so each later frame shares the map error of that fix and
    // is not offered to the filter as a second independent fix.
    assert_eq!(
        terrain(&sink),
        [
            (Outcome::Accepted, Some("render")),
            (SHARED, Some("render")),
            (SHARED, Some("render"))
        ]
    );
    let solution = navigator.solution(at(3 * RATE)).unwrap();
    let error = hover.error_m(&solution, 3 * RATE);
    let (nn, ee, dd) = solution.position_cov.diagonal();
    let (sigma_h, sigma_v) = ((nn + ee).sqrt(), dd.sqrt());
    let (north_m, east_m, up_m) = (error.x, error.y, error.z);
    tracing::info!(
        north_m,
        east_m,
        up_m,
        sigma_h,
        sigma_v,
        "a hovering vehicle after one fix"
    );
    // The alignment error was 43 m horizontal and 12 m vertical.
    assert!(error.xy().norm() < FIX_BOUND_M, "{error}");
    assert!(error.z.abs() < FIX_BOUND_M, "{error}");
    // The declared one-sigma holds the error and is of its order, and it
    // is not less than the stated error of the map.
    assert!(error.xy().norm() < 3.0 * sigma_h && error.z.abs() < 3.0 * sigma_v);
    assert!(sigma_h < 25.0, "{sigma_h}");
    assert!(nn.sqrt() >= report.imagery.sigma_m && dd.sqrt() >= report.elevation.sigma_m);
}

#[test]
fn a_flight_takes_one_independent_fix_in_each_map_cell() {
    show_events();
    let world = World::build(&WorldSpec::default());
    let flight = across_cells();
    let mut navigator = navigator(&flight, belly(), [-20.0, 30.0, -8.0], 40.0);
    load(&mut navigator, &world);
    let mut sink = Vec::new();
    let each_half_second = |step: u64| step.is_multiple_of(25);
    fly(
        &mut navigator,
        &flight,
        &world,
        &belly(),
        1..=250,
        each_half_second,
        &mut sink,
    )
    .unwrap();
    // Ten frames over three cells and the first frame of a fourth: the
    // first frame of each cell is fused, and the two others share its map
    // error.
    let fused = (Outcome::Accepted, Some("render"));
    let shared = (SHARED, Some("render"));
    let expected: Vec<_> = (0..10)
        .map(|frame| {
            if frame % 3 == 0 {
                fused.clone()
            } else {
                shared.clone()
            }
        })
        .collect();
    assert_eq!(terrain(&sink), expected);
    let solution = navigator.solution(at(250)).unwrap();
    let error = flight.error_m(&solution, 250);
    let (nn, ee, _) = solution.position_cov.diagonal();
    let (north_m, east_m, sigma_h) = (error.x, error.y, (nn + ee).sqrt());
    tracing::info!(north_m, east_m, sigma_h, "a flight after four fixes");
    assert!(error.xy().norm() < FIX_BOUND_M, "{error}");
    assert!(error.xy().norm() < 3.0 * (nn + ee).sqrt());
    // Four fixes with independent map error: the filter may go below the
    // error of one fix, but not below the map error over the square root
    // of their count.
    assert!(nn.sqrt() > 1.5 / 2.0, "{}", nn.sqrt());
}

#[test]
fn a_fix_that_the_filter_refuses_does_not_close_its_map_cell() {
    let world = World::build(&WorldSpec::default());
    let hover = Flight {
        start: [CENTRE[0], CENTRE[1], HEIGHT_M],
        speed_m_s: 0.0,
    };
    // The navigator is 90 m off and states a one-sigma of 1 m, so the
    // filter refuses the fix at its innovation gate. The search is wide
    // enough that the match itself succeeds.
    let mut navigator = navigator(&hover, belly(), [90.0, 0.0, 0.0], 1.0);
    let wide = TerrainMapRequest {
        config: TerrainConfig {
            min_search_radius_m: 150.0,
            ..request(&world).config
        },
        ..request(&world)
    };
    navigator
        .load_terrain_map(std::sync::Arc::clone(&world.store), &wide, None)
        .unwrap();
    let mut sink = Vec::new();
    let each_second = |step: u64| step.is_multiple_of(RATE);
    fly(
        &mut navigator,
        &hover,
        &world,
        &belly(),
        1..=2 * RATE,
        each_second,
        &mut sink,
    )
    .unwrap();
    let records = terrain(&sink);
    assert_eq!(records.len(), 2);
    for (outcome, path) in &records {
        let gated = matches!(
            outcome,
            Outcome::Rejected(RejectionReason::InnovationGate { .. })
        );
        assert!(
            gated && *path == Some("render"),
            "a refused fix is not an earlier fix of its cell: {records:?}"
        );
    }
}

#[test]
fn a_late_frame_is_matched_at_its_exposure_and_an_old_frame_is_not_matched() {
    let world = World::build(&WorldSpec::default());
    let flight = across_cells();
    let mut navigator = navigator(&flight, belly(), [15.0, -20.0, 5.0], 30.0);
    load(&mut navigator, &world);
    let mut sink = Vec::new();
    // The frame of step 25 comes 60 ms later, after three more increments.
    for step in 1..=28 {
        navigator
            .ingest(&flight.increment(step), &mut sink)
            .unwrap();
    }
    navigator
        .ingest(&flight.frame(&world, &belly(), 25, 3), &mut sink)
        .unwrap();
    for step in 29..=100 {
        navigator
            .ingest(&flight.increment(step), &mut sink)
            .unwrap();
    }
    // The filter gave its own state at the exposure.
    assert_eq!(
        navigator.terrain_belief_source(CAMERA),
        Some(BeliefSource::FilterAtExposure)
    );
    // The vehicle moved 2.3 m in the 60 ms. The fix is for the exposure.
    let error = flight.error_m(&navigator.solution(at(100)).unwrap(), 100);
    assert!(error.xy().norm() < FIX_BOUND_M, "{error}");
    // A frame that comes 0.8 s after its exposure is older than the bound.
    navigator
        .ingest(&flight.frame(&world, &belly(), 60, 40), &mut sink)
        .unwrap();
    assert_eq!(
        terrain(&sink),
        [
            (Outcome::Accepted, Some("render")),
            (Outcome::Unused("capture_lag"), None)
        ]
    );
}

#[test]
fn a_frame_older_than_the_history_of_the_filter_takes_the_moved_back_belief() {
    let world = World::build(&WorldSpec::default());
    let flight = across_cells();
    let mut navigator = navigator(&flight, belly(), [15.0, -20.0, 5.0], 30.0);
    load(&mut navigator, &world);
    let mut sink = Vec::new();
    for step in 1..=5 {
        navigator
            .ingest(&flight.increment(step), &mut sink)
            .unwrap();
    }
    // The exposure is 100 ms before the first state of the filter, so the
    // history of the filter does not hold it, whatever its depth.
    let mut frame = flight.frame(&world, &belly(), 0, 5);
    frame.acquired_at = MonotonicNanos::from_nanos(at(0).as_nanos() - 100_000_000);
    assert!(navigator.filter().state_at(frame.acquired_at).is_err());
    navigator.ingest(&frame, &mut sink).unwrap();
    assert_eq!(
        navigator.terrain_belief_source(CAMERA),
        Some(BeliefSource::MovedBackFromDelivery)
    );
    // The frame is matched from the moved-back belief: the vehicle went
    // 8 m in the 0.2 s. The filter then judges the time of the fix.
    let records = terrain(&sink);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].1, Some("render"), "{records:?}");
}

#[test]
fn keyframes_serve_inside_their_envelope_and_the_package_serves_outside() {
    let world = World::build(&WorldSpec::default());
    let keyframes = world.keyframes(camera_model(), HEIGHT_M);
    let with_products = TerrainMapRequest {
        products: Some(keyframes.list),
        ..request(&world)
    };
    // 20 m north, 15 m west, and 10 m above the second keyframe: inside
    // its envelope of 60 m and 40 m. And 250 m east of it: outside.
    let [lat, lon, height] = keyframes.poses[1];
    let north = (20.0 / 6_364_000.0_f64).to_degrees();
    let east = |m: f64| (m / (6_378_137.0 * lat.to_radians().cos())).to_degrees();
    let near = [lat + north, lon + east(-15.0), height + 10.0];
    let far = [lat + north, lon + east(250.0), height + 10.0];
    for (start, path) in [(near, "keyframe"), (far, "render")] {
        let hover = Flight {
            start,
            speed_m_s: 0.0,
        };
        let mut navigator = navigator(&hover, belly(), [10.0, 12.0, -6.0], 20.0);
        let report = navigator
            .load_terrain_map(std::sync::Arc::clone(&world.store), &with_products, None)
            .unwrap();
        assert_eq!(report.keyframes, keyframes.poses.len());
        let mut sink = Vec::new();
        let once = |step: u64| step == RATE;
        fly(
            &mut navigator,
            &hover,
            &world,
            &belly(),
            1..=RATE,
            once,
            &mut sink,
        )
        .unwrap();
        assert_eq!(terrain(&sink), [(Outcome::Accepted, Some(path))]);
        let error = hover.error_m(&navigator.solution(at(RATE)).unwrap(), RATE);
        assert!(
            error.xy().norm() < FIX_BOUND_M && error.z.abs() < FIX_BOUND_M,
            "{path}: {error}"
        );
    }
}

/// The fix of each of eight poses, alternately through the belly camera
/// and an oblique camera, against a navigator that knows little (a
/// one-sigma of 1 km), so that the solution after the fix is the fix.
#[test]
fn the_declared_sigma_holds_the_error_and_is_of_its_order() {
    show_events();
    let world = World::build(&WorldSpec::default());
    let wide = TerrainMapRequest {
        config: TerrainConfig {
            max_search_radius_m: 80.0,
            ..request(&world).config
        },
        ..request(&world)
    };
    let (mut squared_m2, mut normalized) = (0.0, 0.0);
    const POSES: u32 = 8;
    for i in 0..POSES {
        let a = f64::from(i);
        let hover = Flight {
            start: [
                CENTRE[0] + 0.004 * (a * 1.7).sin(),
                CENTRE[1] + 0.006 * (a * 2.3).cos(),
                HEIGHT_M + 60.0 * (a * 0.9).sin(),
            ],
            speed_m_s: 0.0,
        };
        let off = [
            30.0 * (a * 1.1).sin(),
            30.0 * (a * 0.7).cos(),
            10.0 * (a * 1.9).sin(),
        ];
        let mount = if i % 2 == 0 {
            belly()
        } else {
            CameraMount::rotated_forward(0.1, -70.0_f64.to_radians(), 0.4 + a)
                .with_lever_arm(Vector3::new(2.0, -0.7, 0.9))
        };
        let mut navigator = navigator(&hover, mount, off, 1_000.0);
        navigator
            .load_terrain_map(std::sync::Arc::clone(&world.store), &wide, None)
            .unwrap();
        let mut sink = Vec::new();
        fly(
            &mut navigator,
            &hover,
            &world,
            &mount,
            1..=1,
            |_| true,
            &mut sink,
        )
        .unwrap();
        assert_eq!(terrain(&sink), [(Outcome::Accepted, Some("render"))]);
        let solution = navigator.solution(at(1)).unwrap();
        let error = hover.error_m(&solution, 1);
        let (nn, ee, dd) = solution.position_cov.diagonal();
        assert!(error.x.abs() < 3.0 * nn.sqrt() && error.y.abs() < 3.0 * ee.sqrt());
        assert!(error.z.abs() < 3.0 * dd.sqrt());
        squared_m2 += error.xy().norm_squared();
        normalized += error.x * error.x / nn + error.y * error.y / ee;
    }
    let rms_m = (squared_m2 / f64::from(POSES)).sqrt();
    let nees = normalized / f64::from(POSES);
    tracing::info!(
        rms_m,
        nees,
        "the horizontal error of eight fixes and its normalized square"
    );
    assert!(rms_m < 2.0 * 5.3, "{rms_m}");
    // The normalized squared error of two axes is 2 for a covariance that
    // is exact. Above 2 the fix is overconfident. Below 0.02 the declared
    // one-sigma is more than ten times the error.
    assert!(nees < 2.0 && nees > 0.02, "{nees}");
}
