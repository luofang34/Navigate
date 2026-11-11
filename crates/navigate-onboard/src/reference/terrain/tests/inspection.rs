//! The record of the last terrain frame, for a host that shows the match:
//! what it holds for each kind of frame, and that keeping it changes no
//! value that the navigator gives.

use super::content::dark;
use super::fixes::{HEIGHT_M, belly};
use super::flight::{CAMERA, Flight, at, load, navigator};
use super::world::{World, WorldSpec};
use navigate_contract::NavigationSolution;

use crate::{
    AcceptanceRecord, Navigator, Outcome, ReferenceNavigator, SensorData, SensorInput, TerrainMatch,
};

/// Correlation cells of 228 m, as in the test of the check before the
/// match: the flight gives a fused frame, frames that share its map error
/// after a match, and frames that are not matched.
const ZOOM: u32 = 17;

/// The inputs of the flight: each inertial increment, a frame each half
/// second, and a frame of noise at step 60.
fn inputs(flight: &Flight, world: &World) -> Vec<SensorInput> {
    let mut inputs = Vec::new();
    for step in 1..=175 {
        inputs.push(flight.increment(step));
        if step.is_multiple_of(25) {
            inputs.push(flight.frame(world, &belly(), step, 0));
        }
        if step == 60 {
            inputs.push(SensorInput {
                sensor: CAMERA,
                acquired_at: at(step),
                delivered_at: at(step),
                data: SensorData::TerrainFrame {
                    width: 160,
                    height: 120,
                    luma: dark(3.0, 1),
                },
            });
        }
    }
    inputs
}

/// Runs `inputs` on `navigator`: the acceptance records, the solution
/// after each input, and the record of the last terrain frame after each
/// frame.
fn run(
    navigator: &mut ReferenceNavigator,
    inputs: &[SensorInput],
) -> (
    Vec<AcceptanceRecord>,
    Vec<Option<NavigationSolution>>,
    Vec<Option<TerrainMatch>>,
) {
    let mut sink = Vec::new();
    let mut solutions = Vec::new();
    let mut matches = Vec::new();
    for input in inputs {
        navigator.ingest(input, &mut sink).unwrap();
        solutions.push(navigator.solution(input.delivered_at));
        if matches!(input.data, SensorData::TerrainFrame { .. }) {
            matches.push(navigator.last_terrain_match(CAMERA).cloned());
        }
    }
    (sink, solutions, matches)
}

/// The record of a fused frame holds the two images at the working size,
/// the pairs with as many inliers as the solved pose states, and the fix;
/// a frame that shares a map error after a match holds the images and the
/// pose and no fix; a frame not matched and a frame of noise hold only
/// their outcome.
fn check_records(matches: &[Option<TerrainMatch>]) {
    let records: Vec<&TerrainMatch> = matches.iter().map(|m| m.as_ref().unwrap()).collect();
    let fused = records[0];
    assert_eq!(fused.outcome(), &Outcome::Accepted);
    assert_eq!(fused.path(), Some("render"));
    let working = fused.working_camera().unwrap().intrinsics;
    let (reference, frame) = (fused.reference().unwrap(), fused.frame().unwrap());
    for luma in [reference, frame] {
        assert_eq!((luma.width, luma.height), (working.width, working.height));
        assert_eq!(luma.pixels.len(), (working.width * working.height) as usize);
    }
    let solved = fused.solved().unwrap();
    let flagged = fused.pairs().iter().filter(|p| p.inlier).count();
    assert_eq!(flagged, solved.inliers);
    assert!(fused.pairs().len() > solved.inliers / 2 && solved.inliers >= 20);
    assert!(solved.reprojection_rms_px < 2.5);
    let gate = fused.prior_gate().unwrap();
    assert!(
        gate.position_m <= gate.position_radius_m && gate.attitude_rad <= gate.attitude_radius_rad
    );
    assert!(
        gate.position_m > 1.0,
        "the belief starts 37 m off: {gate:?}"
    );
    let fix = fused.fix().unwrap();
    let (nn, ee, _) = fix.covariance_ned.diagonal();
    assert!(nn > 0.0 && ee > 0.0);
    // The solved camera is near the believed one: the view was rendered
    // within the search radius of the belief.
    let believed = fused.reference_pose().unwrap().geodetic;
    assert!((solved.camera.geodetic[2] - believed[2]).abs() < 100.0);
    assert!((solved.camera.geodetic[2] - HEIGHT_M).abs() < 50.0);

    let shared = Outcome::Unused("shared_map_error");
    let matched_shared = records[1];
    assert_eq!(matched_shared.outcome(), &shared);
    assert!(matched_shared.reference().is_some() && matched_shared.solved().is_some());
    assert!(matched_shared.fix().is_none());

    // The frame of noise follows the second frame of the flight.
    let noise = records[2];
    assert_eq!(noise.outcome(), &Outcome::Unused("featureless_frame"));
    assert_eq!(noise.acquired_at(), at(60));
    assert!(noise.reference().is_none() && noise.pairs().is_empty());

    let skipped = records.last().unwrap();
    assert_eq!(skipped.outcome(), &shared);
    assert_eq!(skipped.path(), None);
    assert!(skipped.reference().is_none() && skipped.solved().is_none());
    assert!(skipped.prior_gate().is_none() && noise.prior_gate().is_none());
}

#[test]
fn the_record_of_the_last_frame_changes_no_value_of_the_navigator() {
    let world = World::build(&WorldSpec {
        correlation_zoom: ZOOM,
        ..WorldSpec::default()
    });
    let flight = super::fixes::north_through_cells(ZOOM, 12.0, HEIGHT_M, -1.0);
    let inputs = inputs(&flight, &world);
    let start = || {
        let mut navigator = navigator(&flight, belly(), [-20.0, 30.0, -8.0], 40.0);
        load(&mut navigator, &world);
        navigator
    };
    let mut plain = start();
    let mut shown = start();
    shown.set_terrain_match_inspection(true);
    let (records, solutions, none) = run(&mut plain, &inputs);
    let (shown_records, shown_solutions, matches) = run(&mut shown, &inputs);
    assert_eq!(shown_records, records);
    assert_eq!(shown_solutions, solutions);
    assert!(none.iter().all(Option::is_none));
    assert_eq!(matches.len(), 8);
    check_records(&matches);

    // A fork of the inspected navigator half way gives the values of the
    // plain one, and a map load drops the record and keeps the switch.
    let (head, tail) = inputs.split_at(inputs.len() / 2);
    let mut halfway = start();
    halfway.set_terrain_match_inspection(true);
    run(&mut halfway, head);
    let mut fork = halfway.snapshot();
    let mut sink = Vec::new();
    let mut fork_solutions = Vec::new();
    for input in tail {
        fork.ingest(input, &mut sink).unwrap();
        fork_solutions.push(fork.solution(input.delivered_at));
    }
    assert_eq!(sink, records[records.len() - sink.len()..]);
    assert_eq!(fork_solutions, solutions[head.len()..]);
    load(&mut halfway, &world);
    assert!(halfway.last_terrain_match(CAMERA).is_none());
    let (_, _, after_load) = run(&mut halfway, tail);
    assert!(after_load.iter().all(Option::is_some));
}
