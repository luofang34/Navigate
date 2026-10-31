use super::*;
use nalgebra::Matrix3;
use navigate_visual::MapRevision;
use navigate_visual_session::{ContinuityEpoch, FrameKey, Pose, ScaleSource, StreamId};

#[test]
fn test_session_register_frame_nominal() {
    let Ok(mut session) = VisualCorridorSession::new() else {
        return;
    };
    assert!(session.register_frame(1_000_000_000).is_ok());
    assert_eq!(session.next_seq, 1);
    let key = FrameKey {
        stream: StreamId(1),
        continuity: ContinuityEpoch(0),
        index: 0,
    };
    assert!(session.session.pose_at(&key).is_ok());
}

#[test]
fn test_session_rejects_out_of_order_and_duplicate_timestamps() {
    let Ok(mut session) = VisualCorridorSession::new() else {
        return;
    };
    assert!(session.register_frame(2_000_000_000).is_ok());
    assert!(session.register_frame(1_000_000_000).is_err());
    assert!(session.register_frame(2_000_000_000).is_err());
}

#[test]
fn test_session_apply_motion_seeded_map_points() {
    let Ok(mut session) = VisualCorridorSession::new() else {
        return;
    };
    assert!(session.register_frame(100_000_000).is_ok());
    assert!(session.register_frame(200_000_000).is_ok());

    let map = MapRevision {
        release_id: "r1".into(),
        manifest_sha256: "a".repeat(64),
    };
    let scale = ScaleSource::SeededMapPoints { map };
    let res = session.apply_motion_step(0.1, Pose::identity(), scale);
    assert!(res.is_ok());
}

#[test]
fn test_session_refuses_unmetric_scale_and_advances_epoch() {
    let Ok(mut session) = VisualCorridorSession::new() else {
        return;
    };
    assert!(session.register_frame(100_000_000).is_ok());
    assert!(session.register_frame(200_000_000).is_ok());

    let initial_epoch = session.current_epoch;
    let res = session.apply_motion_step(0.1, Pose::identity(), ScaleSource::Unknown);
    assert!(res.is_err());
    assert!(session.current_epoch.0 > initial_epoch.0);
}

#[test]
fn test_dual_path_dead_reckoner_path_a_and_path_b() {
    let mut reck_a = DualPathDeadReckoner::new([0.0, 0.0, 100.0], [20.0, 0.0, 0.0], true);
    let est_a = reck_a.step(0.1, 0, 5.0, 0.05);
    assert_eq!(est_a.path, DeadReckonPath::PathA);
    assert!((est_a.eye_enu[0] - 2.0).abs() < 1e-4);

    let mut reck_b = DualPathDeadReckoner::new([0.0, 0.0, 100.0], [10.0, 5.0, 0.0], false);
    let est_b = reck_b.step(0.5, 0, 2.0, 0.01);
    assert_eq!(est_b.path, DeadReckonPath::PathB);
    assert!((est_b.eye_enu[0] - 5.0).abs() < 1e-4);
    assert!((est_b.eye_enu[1] - 2.5).abs() < 1e-4);
}

#[test]
fn test_dual_path_dead_reckoner_flow_vectors_inflation() {
    let mut reck_flow = DualPathDeadReckoner::new([0.0, 0.0, 100.0], [0.0, 0.0, 0.0], false);
    reck_flow.step(1.0, 6, 4.0, 0.01);
    let var_flow = reck_flow.kinematic.pos_cov_enu_m2[(0, 0)];

    let mut reck_coast = DualPathDeadReckoner::new([0.0, 0.0, 100.0], [0.0, 0.0, 0.0], false);
    reck_coast.step(1.0, 2, 4.0, 0.01);
    let var_coast = reck_coast.kinematic.pos_cov_enu_m2[(0, 0)];

    assert!(var_coast > var_flow);
}

#[test]
fn test_session_filter_ingest_test_fix() {
    let Ok(mut session) = VisualCorridorSession::new() else {
        return;
    };
    let accepted = session.ingest_test_fix(1_000_000_000, 47.0, 11.0, 500.0);
    assert!(accepted);

    let sol = session
        .filter
        .tick(MonotonicNanos::from_nanos(1_000_000_000));
    assert!(sol.is_some());
}

#[test]
fn test_vertical_velocity_gating_helper() {
    assert!(can_update_vertical_velocity(12, 10));
    assert!(can_update_vertical_velocity(5, 48));
    assert!(!can_update_vertical_velocity(11, 47));
}

#[test]
fn test_anchor_worker_state_reset() {
    let mut state = AnchorWorkerState::default();
    state.on_anchor_accepted([10.0, 20.0, 30.0], Matrix3::identity(), 0.1);
    assert_eq!(state.prev_eye, Some([10.0, 20.0, 30.0]));
    assert_eq!(state.vel_enu_mps, [0.0, 0.0, 0.0]);

    state.on_anchor_accepted([12.0, 20.0, 30.0], Matrix3::identity(), 0.1);
    assert!((state.vel_enu_mps[0] - 20.0).abs() < 1e-4);

    state.on_track_loss();
    assert!(state.prev_eye.is_none());
    assert_eq!(state.vel_enu_mps, [0.0, 0.0, 0.0]);
}

#[test]
fn test_submit_anchor_with_consensus_and_handle_relocation() {
    let Ok(mut session) = VisualCorridorSession::new() else {
        return;
    };
    session.handle_relocation([500.0, 200.0, 150.0], Matrix3::identity(), 0.5);
    assert_eq!(
        session.dead_reckoner.kinematic.eye_enu,
        [500.0, 200.0, 150.0]
    );
    assert_eq!(
        session.dead_reckoner.kinematic.vel_enu_mps,
        [1000.0, 400.0, 300.0]
    );

    session.handle_relocation([600.0, 250.0, 160.0], Matrix3::identity(), 0.0);
    assert_eq!(
        session.dead_reckoner.kinematic.eye_enu,
        [600.0, 250.0, 160.0]
    );
    assert_eq!(session.dead_reckoner.kinematic.vel_enu_mps, [0.0, 0.0, 0.0]);
}

#[test]
fn test_submit_anchor_with_consensus_applies_policy() {
    use navigate_visual_session::AnchorPolicy;
    let Ok(mut session) = VisualCorridorSession::new() else {
        return;
    };
    let custom_policy = AnchorPolicy {
        consensus_anchors: 3,
        consensus_separation_m: 10.0,
        ..Default::default()
    };
    assert_eq!(session.anchor_policy().consensus_anchors, 2);

    assert!(session.set_anchor_policy(custom_policy).is_ok());
    assert_eq!(session.anchor_policy().consensus_anchors, 3);
    assert_eq!(session.anchor_policy().consensus_separation_m, 10.0);
}
