//! The record of the last terrain frame, for a host that shows the match:
//! what it holds for each kind of frame, and that keeping it changes no
//! value that the navigator gives.

use super::content::dark;
use super::fixes::{HEIGHT_M, belly};
use super::flight::{CAMERA, CLOCK, Flight, at, imager, load, navigator, suite};
use super::precheck::{FORWARD, aligned_state, forward};
use super::world::{World, WorldSpec};
use navigate_contract::NavigationSolution;

use crate::{
    AcceptanceRecord, Navigator, Outcome, ReferenceNavigator, SensorData, SensorInput, SensorSuite,
    TerrainMatch,
};

/// Correlation cells of 228 m, as in the test of the check before the
/// match. A forward camera reaches each cell first and takes the one map
/// block, so the belly camera gives a fused frame, frames withheld after
/// a match, and frames that are not matched.
const ZOOM: u32 = 17;
const STEPS: u64 = 300;

/// The inputs of the flight: each inertial increment, a frame of each
/// camera each half second, and a belly frame of noise at step 60.
fn inputs(flight: &Flight, world: &World) -> Vec<SensorInput> {
    let mut inputs = Vec::new();
    for step in 1..=STEPS {
        inputs.push(flight.increment(step));
        if step.is_multiple_of(25) {
            inputs.push(flight.frame(world, &belly(), step, 0));
            let mut ahead = flight.frame(world, &forward(), step, 0);
            ahead.sensor = FORWARD;
            inputs.push(ahead);
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

/// The navigator of the flight with the belly and the forward camera,
/// aligned off the truth, with the map loaded.
fn two_cameras(flight: &Flight, world: &World) -> ReferenceNavigator {
    let mut sensors = suite(imager(), belly()).sensors().to_vec();
    let mut second = sensors.last().cloned().unwrap();
    second.id = FORWARD;
    second.mount = Some(forward());
    sensors.push(second);
    let aligned = navigator(flight, belly(), [-20.0, 30.0, -8.0], 40.0);
    let suite = SensorSuite::new(sensors).unwrap();
    let mut navigator = ReferenceNavigator::new(suite, CLOCK, None).unwrap();
    navigator.align(&aligned_state(&aligned)).unwrap();
    load(&mut navigator, world);
    navigator
}

/// Runs `inputs` on `navigator`: the acceptance records, the solution
/// after each input, and the record of the last belly frame after each
/// belly frame.
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
        if input.sensor == CAMERA && matches!(input.data, SensorData::TerrainFrame { .. }) {
            matches.push(navigator.last_terrain_match(CAMERA).cloned());
        }
    }
    (sink, solutions, matches)
}

/// The first record that `pick` selects.
fn first<'a>(
    records: &[&'a TerrainMatch],
    pick: impl Fn(&TerrainMatch) -> bool,
) -> &'a TerrainMatch {
    records.iter().copied().find(|m| pick(m)).unwrap()
}

/// The record of a fused frame holds the two images at the working size,
/// the pairs with as many inliers as the solved pose states, and the fix
/// with the declared map error of its cell; a frame withheld after a
/// match holds the images and the pose and no fix; a frame not matched
/// and a frame of noise hold only their outcome.
fn check_records(matches: &[Option<TerrainMatch>]) {
    let records: Vec<&TerrainMatch> = matches.iter().map(|m| m.as_ref().unwrap()).collect();
    let fused = first(&records, |m| m.outcome() == &Outcome::Accepted);
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
    let fix = fused.fix().unwrap();
    let (nn, ee, _) = fix.covariance_ned.diagonal();
    assert!(nn > 0.0 && ee > 0.0);
    let map = fix.map_sigma_m.unwrap();
    assert!(map.iter().all(|sigma| *sigma > 0.0), "{map:?}");
    // The solved camera is near the believed one: the view was rendered
    // within the search radius of the belief.
    let believed = fused.reference_pose().unwrap().geodetic;
    assert!((solved.camera.geodetic[2] - believed[2]).abs() < 100.0);
    assert!((solved.camera.geodetic[2] - HEIGHT_M).abs() < 50.0);

    let shared = Outcome::Unused("shared_map_error");
    let withheld = first(&records, |m| m.outcome() == &shared && m.path().is_some());
    assert!(withheld.reference().is_some() && withheld.solved().is_some());
    assert!(withheld.fix().is_none());

    let noise = first(&records, |m| m.acquired_at() == at(60));
    assert_eq!(noise.outcome(), &Outcome::Unused("featureless_frame"));
    assert!(noise.reference().is_none() && noise.pairs().is_empty());

    let skipped = first(&records, |m| m.outcome() == &shared && m.path().is_none());
    assert!(skipped.reference().is_none() && skipped.solved().is_none());
}

#[test]
fn the_record_of_the_last_frame_changes_no_value_of_the_navigator() {
    let world = World::build(&WorldSpec {
        correlation_zoom: ZOOM,
        ..WorldSpec::default()
    });
    let flight = super::fixes::north_through_cells(ZOOM, 12.0, HEIGHT_M, -1.0);
    let inputs = inputs(&flight, &world);
    let (head, tail) = inputs.split_at(inputs.len() / 2);
    let mut plain = two_cameras(&flight, &world);
    let (records, solutions, none) = run(&mut plain, &inputs);
    assert!(none.iter().all(Option::is_none));

    // The inspected navigator, and a fork of it half way, give the values
    // of the plain one.
    let mut shown = two_cameras(&flight, &world);
    shown.set_terrain_match_inspection(true);
    let (mut shown_records, mut shown_solutions, mut matches) = run(&mut shown, head);
    let mut fork = shown.snapshot();
    let (rest_records, rest_solutions, rest_matches) = run(&mut shown, tail);
    shown_records.extend(rest_records);
    shown_solutions.extend(rest_solutions);
    matches.extend(rest_matches);
    assert_eq!(shown_records, records);
    assert_eq!(shown_solutions, solutions);
    assert_eq!(matches.len(), 13);
    check_records(&matches);
    let mut sink = Vec::new();
    let mut fork_solutions = Vec::new();
    for input in tail {
        fork.ingest(input, &mut sink).unwrap();
        fork_solutions.push(fork.solution(input.delivered_at));
    }
    assert_eq!(sink, records[records.len() - sink.len()..]);
    assert_eq!(fork_solutions, solutions[head.len()..]);

    // A map load drops the record and keeps the switch.
    load(&mut shown, &world);
    assert!(shown.last_terrain_match(CAMERA).is_none());
    let next = STEPS + 25;
    let later: Vec<_> = (STEPS + 1..=next)
        .map(|step| flight.increment(step))
        .chain([flight.frame(&world, &belly(), next, 0)])
        .collect();
    let (_, _, after_load) = run(&mut shown, &later);
    assert!(after_load.iter().all(Option::is_some));
}
