use super::{AnchorArrival, AnchorOutcome};
use crate::{
    corridor::{
        kinematics::VehicleDynamicsProfile,
        latency::StageLatencies,
        motion::{OdometryStep, TrackState},
    },
    pose::{PoseCandidate, PoseScore},
};
use nalgebra::{Matrix3, Vector3};
use navigate_match::{ActiveTrackSet, HadamardProjector, PreparedKashinQuery, SparseFeatures};
use serde_json::{Value, json};

fn k_test() -> Matrix3<f64> {
    Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0)
}

fn candidate(eye: [f64; 3]) -> PoseCandidate {
    PoseCandidate {
        eye,
        rotation: Matrix3::identity(),
        k: k_test(),
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

fn outcome(seq: usize, case_id: &str, eye: [f64; 3], accepted: bool) -> AnchorOutcome {
    AnchorOutcome {
        sent_frame_seq: seq,
        case_id: case_id.into(),
        k_render: k_test(),
        render_size: [800, 600],
        best: Some(candidate(eye)),
        seeded: accepted.then(ActiveTrackSet::empty),
        locked_focal_px: None,
        attempts: vec![json!({"seed": 0}), json!({"seed": 1}), json!({"seed": 2})],
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
    state.frame_seq = 5;
    let mut arrival = AnchorArrival {
        state: &mut state,
        independent_fixes: &mut fixes,
        latencies: &mut latencies,
        results: &mut results,
        catalog_sha256: "catalog",
        cell_m: 500.0,
        local_frame: None,
        focal_estimator: None,
        dynamics_profile: VehicleDynamicsProfile::rotary_survey(),
    };

    arrival.state.frames_since_anchor = 7;
    assert!(
        arrival
            .apply(outcome(3, "f-0003", eye, false), Some("f-0005"))
            .is_none()
    );
    assert_eq!(arrival.state.frames_since_anchor, 7);
    assert_eq!(*arrival.independent_fixes, 0);

    arrival.state.frame_seq = 9;
    assert!(
        arrival
            .apply(outcome(6, "f-0006", eye, true), Some("f-0009"))
            .is_some()
    );
    assert_eq!(arrival.state.frames_since_anchor, 0);
    assert_eq!(*arrival.independent_fixes, 1);

    let _ = arrival.apply(outcome(10, "f-0010", eye, true), None);
    assert_eq!(
        *arrival.independent_fixes, 1,
        "same map cell is not independent"
    );

    assert_eq!(latencies.anchor_ms, [12.5, 12.5, 12.5]);
    assert_eq!(results[0]["accepted"], false);
    assert_eq!(results[1]["anchor_result_for"], "f-0006");
    assert_eq!(results[1]["arrived_at"], "f-0009");
    assert_eq!(results[1]["propagated_frames"], 3);
    assert_eq!(results[1]["independent_ekf_fix"], true);
    assert_eq!(results[2]["arrived_at"], Value::Null);
    assert_eq!(results[2]["independent_ekf_fix"], false);
    assert_eq!(state.visited_cells.len(), 1);
}

#[test]
fn delayed_anchor_arrival_propagates_odometry_and_prunes_frustum_tracks() {
    let mut state = TrackState::new();
    state.frame_seq = 3;
    state.pose_steps.push_back(OdometryStep {
        frame_seq: 2,
        rot_step: Matrix3::identity(),
        eye_delta: [6.0, 0.0, 0.0],
    });
    state.pose_steps.push_back(OdometryStep {
        frame_seq: 3,
        rot_step: Matrix3::identity(),
        eye_delta: [4.0, 0.0, 0.0],
    });

    // Camera looks along +Z in world (identity world_to_cam).
    // At t_now eye = [10, 0, 0]: point [10, 0, 200] projects to center, while [10, 0, -50] is behind camera.
    let tracks = ActiveTrackSet {
        world_points_enu: vec![[10.0, 0.0, 200.0], [10.0, 0.0, -50.0]],
        sign_bits: vec![1, 2],
        kashin_i8: vec![[1_u8; 64], [2_u8; 64]],
    };
    let delayed = AnchorOutcome {
        sent_frame_seq: 1,
        case_id: "f-0001".into(),
        k_render: k_test(),
        render_size: [800, 600],
        best: Some(candidate([0.0, 0.0, 0.0])),
        seeded: Some(tracks),
        locked_focal_px: None,
        attempts: vec![json!({"seed": 0})],
        elapsed_ms: 8.0,
    };

    let mut fixes = 0;
    let mut latencies = StageLatencies::default();
    let mut results = Vec::new();
    let prop = AnchorArrival {
        state: &mut state,
        independent_fixes: &mut fixes,
        latencies: &mut latencies,
        results: &mut results,
        catalog_sha256: "catalog",
        cell_m: 500.0,
        local_frame: None,
        focal_estimator: None,
        dynamics_profile: VehicleDynamicsProfile::rotary_survey(),
    }
    .apply(delayed, Some("f-0003"))
    .expect("propagated pose");

    assert!((prop.pose.0[0] - 10.0).abs() < 1e-9);
    assert_eq!(state.active_tracks.len(), 1);
    assert_eq!(state.active_tracks.world_points_enu[0], [10.0, 0.0, 200.0]);
    assert_eq!(state.prev_eye, Some([10.0, 0.0, 0.0]));
}

#[test]
fn test_anchor_worker_state_transitions() {
    use super::AnchorWorkerState;

    let mut state = AnchorWorkerState::default();
    assert!(state.prev_eye.is_none());
    assert!(state.prev_rot.is_none());
    assert_eq!(state.vel_enu_mps, [0.0, 0.0, 0.0]);

    // Cold start
    state.on_anchor_accepted([10.0, 20.0, 100.0], Matrix3::identity(), 0.1);
    assert_eq!(state.prev_eye, Some([10.0, 20.0, 100.0]));
    assert_eq!(state.vel_enu_mps, [0.0, 0.0, 0.0]);

    // Follow-up anchor
    state.on_anchor_accepted([13.0, 20.0, 100.0], Matrix3::identity(), 0.1);
    assert!((state.vel_enu_mps[0] - 30.0).abs() < 1e-4);

    // Relocation
    state.on_relocation([500.0, 20.0, 100.0], 1.0);
    assert_eq!(state.prev_eye, Some([500.0, 20.0, 100.0]));
    assert!((state.vel_enu_mps[0] - 487.0).abs() < 1e-4);

    // Track loss
    state.on_track_loss();
    assert!(state.prev_eye.is_none());
    assert!(state.prev_rot.is_none());
    assert_eq!(state.vel_enu_mps, [0.0, 0.0, 0.0]);
}

#[test]
fn sequential_anchor_arrivals_with_5m_snap_keep_velocity_bounded() {
    let mut state = TrackState::new();
    let mut fixes = 0;
    let mut latencies = StageLatencies::default();
    let mut results = Vec::new();
    let profile = VehicleDynamicsProfile::rotary_survey();

    state.frame_seq = 2;
    {
        let mut arrival = AnchorArrival {
            state: &mut state,
            independent_fixes: &mut fixes,
            latencies: &mut latencies,
            results: &mut results,
            catalog_sha256: "catalog",
            cell_m: 500.0,
            local_frame: None,
            focal_estimator: None,
            dynamics_profile: profile,
        };
        let first = arrival
            .apply(
                outcome(1, "f-0001", [100.0, 200.0, 350.0], true),
                Some("f-0002"),
            )
            .expect("first anchor");
        assert_eq!(first.pose.0, [100.0, 200.0, 350.0]);
    }
    assert_eq!(state.prev_vel, [0.0, 0.0, 0.0]);

    // Simulate steady warm-tracked velocity (12 m/s East) followed by a 5 m map-anchor snap.
    state.prev_vel = [12.0, 0.0, 0.0];
    state.time_since_accepted_pose_s = 0.033;
    state.frame_seq = 12;
    {
        let mut arrival = AnchorArrival {
            state: &mut state,
            independent_fixes: &mut fixes,
            latencies: &mut latencies,
            results: &mut results,
            catalog_sha256: "catalog",
            cell_m: 500.0,
            local_frame: None,
            focal_estimator: None,
            dynamics_profile: profile,
        };
        let second = arrival
            .apply(
                outcome(10, "f-0010", [105.0, 200.0, 350.0], true),
                Some("f-0012"),
            )
            .expect("second anchor");
        assert_eq!(second.pose.0, [105.0, 200.0, 350.0]);
    }
    let speed = Vector3::from(state.prev_vel).norm();
    assert!(
        speed <= profile.max_speed_mps,
        "velocity exploded to {speed} m/s (> {})",
        profile.max_speed_mps
    );
    assert!((state.prev_vel[0] - 12.0).abs() < 1e-9);
    let kin = state.kinematic.as_ref().expect("kinematic initialized");
    assert!(Vector3::from(kin.vel_enu_mps).norm() <= profile.max_speed_mps);

    // Same-frame warm-track solve on the arrival frame must snap pose without adding (8 m / dt_s) velocity.
    let proj = HadamardProjector::new(0xBEEF);
    let empty_q = PreparedKashinQuery::from_features(&proj, &SparseFeatures::empty()).expect("q");
    let warm_on_arrival = candidate([113.0, 196.0, 352.0]);
    state.snap_pose_preserving_velocity(&warm_on_arrival, empty_q);
    assert_eq!(state.prev_eye, Some([113.0, 196.0, 352.0]));
    assert!((state.prev_vel[0] - 12.0).abs() < 1e-9);
    assert_eq!(
        state.kinematic.as_ref().expect("kinematic").vel_enu_mps,
        [12.0, 0.0, 0.0]
    );
}
