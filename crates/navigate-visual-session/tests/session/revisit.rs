//! Out-and-back flight: revisits close the loop through real geometric checks.

use crate::support::*;
use nalgebra::Vector3;
use navigate_visual_session::{
    MapPose, RevisionCause, RevisitDecision, RevisitEvidence, SessionEvent, VisualSession,
};

const STEP_M: f64 = 6.0;
const LEG: u64 = 60;

fn out_and_back() -> Flight {
    let mut flight = Flight::standard();
    flight.heading_bias_rad = 0.004;
    flight.scale_error = 0.02;
    for i in 0..LEG {
        flight.fly_to(nadir(Vector3::new(STEP_M * i as f64, 0.0, 100.0), 0.0));
    }
    for i in (0..LEG).rev() {
        flight.fly_to(nadir(
            Vector3::new(STEP_M * i as f64 + 1.5, 3.0, 102.0),
            std::f64::consts::PI,
        ));
    }
    let first = flight.truth[0].0;
    let start = anchor(&flight, &first, Vector3::zeros());
    flight.session.submit_anchor(start).unwrap();
    flight
}

fn map_position(session: &VisualSession, k: &navigate_visual_session::FrameKey) -> Vector3<f64> {
    match session.pose_at(k).unwrap().map {
        MapPose::Located { pose, .. } => pose.translation.vector,
        MapPose::Unlocated(reason) => panic!("frame {} is not located: {reason:?}", k.index),
    }
}

/// Largest disagreement between outbound and return map positions of the
/// same ground place, after removing the true offset between the passes.
fn revisit_inconsistency(flight: &Flight) -> f64 {
    (0..LEG as usize)
        .map(|i| {
            let (out_key, out_truth, _) = &flight.truth[i];
            let (back_key, back_truth, _) = &flight.truth[2 * LEG as usize - 1 - i];
            let measured =
                map_position(&flight.session, back_key) - map_position(&flight.session, out_key);
            let truth = back_truth.translation.vector - out_truth.translation.vector;
            (measured - truth).xy().norm()
        })
        .fold(0.0, f64::max)
}

fn close_revisits(flight: &mut Flight) -> usize {
    let mut accepted = 0;
    let returning: Vec<_> = flight.truth[LEG as usize..]
        .iter()
        .map(|(k, _, _)| *k)
        .collect();
    for later in returning.iter().step_by(4) {
        // The host tries candidates in order. A candidate without overlap fails the check.
        for candidate in flight.session.revisit_candidates(later, None).unwrap() {
            let (_, later_truth, later_frame) = flight.find(later).clone();
            let (_, earlier_truth, earlier_frame) = flight.find(&candidate.earlier).clone();
            let surface = ground_view(&pose_of(&candidate.earlier_pose), 0.0);
            let matches = ground_matches(&earlier_truth, &later_truth, 0.0);
            let evidence = RevisitEvidence {
                later: &later_frame,
                earlier: &earlier_frame,
                earlier_surface: &surface,
                matches: &matches,
                matcher_identity: "synthetic-ground",
            };
            let Ok(constraint) = flight.session.verify_revisit(&candidate, evidence) else {
                continue;
            };
            if flight.session.submit_revisit(constraint).unwrap() == RevisitDecision::Accepted {
                accepted += 1;
                break;
            }
        }
    }
    accepted
}

#[test]
fn revisits_make_the_return_pass_agree_with_the_outbound_pass() {
    let mut flight = out_and_back();
    let before = revisit_inconsistency(&flight);
    assert!(
        before > 15.0,
        "heading bias separates the passes before closure: {before} m"
    );
    flight.session.drain_events();
    let accepted = close_revisits(&mut flight);
    assert!(accepted >= 5, "verified closures: {accepted}");
    let after = revisit_inconsistency(&flight);
    assert!(
        after < before / 4.0 && after < 5.0,
        "revisit inconsistency {before} m -> {after} m"
    );
    let events = flight.session.drain_events().events;
    assert!(events.iter().any(|e| matches!(e, SessionEvent::TrajectoryRevised(r) if r.cause == RevisionCause::Closure && r.ranges.iter().all(|range| range.from.index < LEG))));
}

#[test]
fn temporal_neighbours_are_not_revisits_and_wrong_geometry_is_refused() {
    let mut flight = out_and_back();
    let k = flight.truth[30].0;
    let candidates = flight.session.revisit_candidates(&k, None).unwrap();
    assert!(
        candidates
            .iter()
            .all(|c| c.earlier.index + 20 <= k.index || c.earlier.index >= k.index + 20),
        "no adjacent keyframes: {candidates:?}"
    );

    // Anchors near both ends make the predicted closure tight.
    for index in [LEG as usize - 1, 2 * LEG as usize - 1] {
        let k = flight.truth[index].0;
        flight
            .session
            .submit_anchor(anchor(&flight, &k, Vector3::zeros()))
            .unwrap();
    }
    // A repeated pattern: the matches describe a camera 70 m away from the later frame.
    let later = flight.truth[2 * LEG as usize - 5].0;
    let candidate = flight
        .session
        .revisit_candidates(&later, None)
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let (_, later_truth, later_frame) = flight.find(&later).clone();
    let (_, earlier_truth, earlier_frame) = flight.find(&candidate.earlier).clone();
    let moved = later_truth * navigate_visual_session::Pose::translation(70.0, 0.0, 0.0);
    let surface = ground_view(&pose_of(&candidate.earlier_pose), 0.0);
    let matches = ground_matches(&earlier_truth, &moved, 0.0);
    let evidence = RevisitEvidence {
        later: &later_frame,
        earlier: &earlier_frame,
        earlier_surface: &surface,
        matches: &matches,
        matcher_identity: "synthetic-ground",
    };
    let result = flight.session.verify_revisit(&candidate, evidence);
    let refused = match result {
        Err(_) => true,
        Ok(constraint) => matches!(
            flight.session.submit_revisit(constraint).unwrap(),
            RevisitDecision::Inconsistent { .. }
        ),
    };
    assert!(
        refused,
        "a closure 70 m from the prediction does not enter the graph"
    );
}
