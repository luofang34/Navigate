use super::{AnchorArrival, AnchorOutcome};
use crate::{
    corridor::{latency::StageLatencies, motion::TrackState},
    pose::{PoseCandidate, PoseScore},
};
use nalgebra::Matrix3;
use navigate_match::ActiveTrackSet;
use serde_json::Value;

fn candidate(eye: [f64; 3]) -> PoseCandidate {
    PoseCandidate {
        eye,
        rotation: Matrix3::identity(),
        k: Matrix3::identity(),
        size: [800, 600],
        points: Vec::new(),
        pixels: Vec::new(),
        inliers: Vec::new(),
        score: PoseScore {
            focal_px: 600.0,
            geometry_accepted: true,
            holdout_count: 0,
            holdout_inlier_fraction: 1.0,
            holdout_median_px: 0.0,
            image_support_fraction: 1.0,
            inliers: 100,
            pairs: 100,
            positive_depth_fraction: 1.0,
            train_inliers: 100,
            train_rms_px: 0.5,
        },
    }
}

fn outcome(case_id: &str, eye: [f64; 3], accepted: bool) -> AnchorOutcome {
    AnchorOutcome {
        case_id: case_id.into(),
        best: Some(candidate(eye)),
        seeded: accepted.then(ActiveTrackSet::empty),
        attempts: 3,
        elapsed_ms: 12.5,
    }
}

#[test]
fn arrivals_count_fixes_for_the_sent_frame_and_keep_tracks_on_rejection() {
    let mut state = TrackState::new();
    let mut fixes = 0;
    let mut latencies = StageLatencies::default();
    let mut results = Vec::new();
    let eye = [100.0, 100.0, 900.0];
    let mut arrival = AnchorArrival {
        state: &mut state,
        independent_fixes: &mut fixes,
        latencies: &mut latencies,
        results: &mut results,
        catalog_sha256: "catalog",
        cell_m: 500.0,
    };

    arrival.state.frames_since_anchor = 7;
    arrival.apply(outcome("f-0003", eye, false), Some("f-0005"));
    assert_eq!(arrival.state.frames_since_anchor, 7);
    assert_eq!(*arrival.independent_fixes, 0);

    arrival.apply(outcome("f-0006", eye, true), Some("f-0009"));
    assert_eq!(arrival.state.frames_since_anchor, 0);
    assert_eq!(*arrival.independent_fixes, 1);

    arrival.apply(outcome("f-0010", eye, true), None);
    assert_eq!(
        *arrival.independent_fixes, 1,
        "same map cell is not independent"
    );

    assert_eq!(latencies.anchor_ms, [12.5, 12.5, 12.5]);
    assert_eq!(results[0]["accepted"], false);
    assert_eq!(results[1]["anchor_result_for"], "f-0006");
    assert_eq!(results[1]["arrived_at"], "f-0009");
    assert_eq!(results[1]["independent_ekf_fix"], true);
    assert_eq!(results[2]["arrived_at"], Value::Null);
    assert_eq!(results[2]["independent_ekf_fix"], false);
    assert_eq!(state.visited_cells.len(), 1);
}
