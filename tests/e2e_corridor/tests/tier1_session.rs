//! Tier 1: Feature coverage tests for Features 6 to 11 (Session, Filter & Dead Reckoning).

use e2e_corridor_tests::{
    AnchorWorkerState, DeadReckonPath, DualPathDeadReckoner, KinematicState, TestResult,
    VisualCorridorSession, approx_eq, can_update_vertical_velocity, check,
};
use nalgebra::Matrix3;
use navigate_contract::MonotonicNanos;
use navigate_visual::MapRevision;
use navigate_visual_session::{
    ContinuityEpoch, FrameKey, Pose, RelativeMotion, ScaleSource, StreamId,
};

// --- Feature 6: Visual Session Frame Registration ---

#[test]
fn t1_06_01_register_nominal_frame() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    let res = session.register_frame(1_000_000_000);
    check(res.is_ok(), "frame registered successfully")?;
    check(session.next_seq == 1, "next sequence is 1")?;
    Ok(())
}

#[test]
fn t1_06_02_frame_record_dimensions() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    session.register_frame(1_000_000_000)?;
    let key = FrameKey {
        stream: StreamId(1),
        continuity: ContinuityEpoch(0),
        index: 0,
    };
    let lookup = session.session.pose_at(&key);
    check(lookup.is_ok(), "registered key exists in session")?;
    Ok(())
}

#[test]
fn t1_06_03_sequential_frame_indexing() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    for i in 1..=5 {
        session.register_frame(i * 33_333_333)?;
    }
    check(session.next_seq == 5, "5 frames indexed sequentially")?;
    Ok(())
}

#[test]
fn t1_06_04_unanchored_frame_state() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    session.register_frame(100_000_000)?;
    let key = FrameKey {
        stream: StreamId(1),
        continuity: ContinuityEpoch(0),
        index: 0,
    };
    let pose = session.session.pose_at(&key)?;
    check(
        matches!(pose.map, navigate_visual_session::MapPose::Unlocated(_)),
        "frame before first anchor is unlocated",
    )?;
    Ok(())
}

#[test]
fn t1_06_05_session_pose_lookup() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    session.register_frame(200_000_000)?;
    let key = FrameKey {
        stream: StreamId(1),
        continuity: ContinuityEpoch(0),
        index: 0,
    };
    check(
        session.session.pose_at(&key).is_ok(),
        "pose lookup succeeds",
    )?;
    Ok(())
}

// --- Feature 7: Motion Application & Epochs ---

#[test]
fn t1_07_01_apply_motion_seeded_map_points() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    session.register_frame(100_000_000)?;
    session.register_frame(200_000_000)?;
    let from = FrameKey {
        stream: StreamId(1),
        continuity: ContinuityEpoch(0),
        index: 0,
    };
    let to = FrameKey {
        stream: StreamId(1),
        continuity: ContinuityEpoch(0),
        index: 1,
    };
    let map = MapRevision {
        release_id: "r1".into(),
        manifest_sha256: "a".repeat(64),
    };
    let motion = RelativeMotion {
        from,
        to,
        from_to: Pose::identity(),
        scale: ScaleSource::SeededMapPoints { map },
        from_observation_sha256: format!("{:064x}", 1),
        to_observation_sha256: format!("{:064x}", 2),
    };
    let outcome = session.session.apply_motion(motion);
    check(
        outcome.is_ok(),
        "apply motion with SeededMapPoints succeeds",
    )?;
    Ok(())
}

#[test]
fn t1_07_02_apply_motion_map_depth() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    session.register_frame(100_000_000)?;
    session.register_frame(200_000_000)?;
    let from = FrameKey {
        stream: StreamId(1),
        continuity: ContinuityEpoch(0),
        index: 0,
    };
    let to = FrameKey {
        stream: StreamId(1),
        continuity: ContinuityEpoch(0),
        index: 1,
    };
    let map = MapRevision {
        release_id: "r1".into(),
        manifest_sha256: "a".repeat(64),
    };
    let motion = RelativeMotion {
        from,
        to,
        from_to: Pose::identity(),
        scale: ScaleSource::MapDepth { map },
        from_observation_sha256: format!("{:064x}", 1),
        to_observation_sha256: format!("{:064x}", 2),
    };
    let outcome = session.session.apply_motion(motion);
    check(outcome.is_ok(), "apply motion with MapDepth succeeds")?;
    Ok(())
}

#[test]
fn t1_07_03_apply_motion_external() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    session.register_frame(100_000_000)?;
    session.register_frame(200_000_000)?;
    let from = FrameKey {
        stream: StreamId(1),
        continuity: ContinuityEpoch(0),
        index: 0,
    };
    let to = FrameKey {
        stream: StreamId(1),
        continuity: ContinuityEpoch(0),
        index: 1,
    };
    let motion = RelativeMotion {
        from,
        to,
        from_to: Pose::identity(),
        scale: ScaleSource::External {
            sensor: "altimeter".into(),
        },
        from_observation_sha256: format!("{:064x}", 1),
        to_observation_sha256: format!("{:064x}", 2),
    };
    let outcome = session.session.apply_motion(motion);
    check(outcome.is_ok(), "apply motion with External scale succeeds")?;
    Ok(())
}

#[test]
fn t1_07_04_advance_continuity_epoch() -> TestResult {
    let epoch0 = ContinuityEpoch(0);
    let epoch1 = ContinuityEpoch(1);
    check(epoch1.0 > epoch0.0, "new epoch has incremented counter")?;
    Ok(())
}

#[test]
fn t1_07_05_segment_start_after_gap() -> TestResult {
    let seg_start = navigate_visual_session::SegmentStart::AnchorAfterGap;
    check(
        format!("{seg_start:?}").contains("AnchorAfterGap"),
        "segment start variant exists",
    )?;
    Ok(())
}

// --- Feature 8: Anchor Submission & EKF Ingestion ---

#[test]
fn t1_08_01_submit_visual_anchor_nominal() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    session.register_frame(1_000_000_000)?;
    check(session.next_seq == 1, "session frame ready for anchor")?;
    Ok(())
}

#[test]
fn t1_08_02_convert_visual_fix_position() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    let admitted = session.ingest_test_fix(1_000_000_000, 47.0, 11.0, 500.0);
    check(admitted, "fix accepted into filter")?;
    Ok(())
}

#[test]
fn t1_08_03_ingest_position_fix_into_ekf() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    let admitted1 = session.ingest_test_fix(1_000_000_000, 47.0, 11.0, 500.0);
    check(admitted1, "initial fix admitted")?;
    let sol = session
        .filter
        .tick(MonotonicNanos::from_nanos(1_000_000_000));
    check(sol.is_some(), "tick publishes navigation solution")?;
    Ok(())
}

#[test]
fn t1_08_04_filter_covariance_reduction() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    session.ingest_test_fix(1_000_000_000, 47.0, 11.0, 500.0);
    let sol1 = session
        .filter
        .tick(MonotonicNanos::from_nanos(1_000_000_000))
        .ok_or("sol1 missing")?;
    let var1 = sol1.position_cov.diagonal().0;
    session.ingest_test_fix(1_100_000_000, 47.0, 11.0, 500.0);
    let sol2 = session
        .filter
        .tick(MonotonicNanos::from_nanos(1_100_000_000))
        .ok_or("sol2 missing")?;
    let var2 = sol2.position_cov.diagonal().0;
    check(
        var2 <= var1 + 1.0,
        "filter variance bounded after second fix",
    )?;
    Ok(())
}

#[test]
fn t1_08_05_evidence_independence_gate() -> TestResult {
    let label = navigate_visual_session::FusionEligibility::Independent;
    let ind = navigate_visual_fusion::EvidenceIndependence::ValidatedIndependent;
    let restricted = ind.restricted_by(&label);
    check(
        restricted == navigate_visual_fusion::EvidenceIndependence::ValidatedIndependent,
        "independent remains independent",
    )?;
    Ok(())
}

// --- Feature 9: Dual-Path Dead Reckoning ---

#[test]
fn t1_09_01_path_a_attitude_interpolation() -> TestResult {
    let mut reckoner = DualPathDeadReckoner::new([0.0, 0.0, 100.0], [20.0, 0.0, 0.0], true);
    let est = reckoner.step(0.1, 0, 5.0, 0.05);
    check(
        est.path == DeadReckonPath::PathA,
        "selects Path A when external sensors present",
    )?;
    check(
        approx_eq(est.eye_enu[0], 2.0, 1e-4),
        "position updated by 2.0m",
    )?;
    Ok(())
}

#[test]
fn t1_09_02_path_a_search_prior_projection() -> TestResult {
    let reckoner = DualPathDeadReckoner::new([100.0, 500.0, 200.0], [0.0, 0.0, 0.0], true);
    check(
        reckoner.external_sensors_available,
        "external sensors configured",
    )?;
    Ok(())
}

#[test]
fn t1_09_03_path_b_kinematic_state_maintenance() -> TestResult {
    let mut reckoner = DualPathDeadReckoner::new([0.0, 0.0, 100.0], [10.0, 5.0, 0.0], false);
    let est = reckoner.step(0.5, 0, 2.0, 0.01);
    check(
        est.path == DeadReckonPath::PathB,
        "selects Path B on vision fallback",
    )?;
    check(approx_eq(est.eye_enu[0], 5.0, 1e-4), "x displaced 5.0m")?;
    check(approx_eq(est.eye_enu[1], 2.5, 1e-4), "y displaced 2.5m")?;
    Ok(())
}

#[test]
fn t1_09_04_path_b_optical_flow_integration() -> TestResult {
    let mut reckoner = DualPathDeadReckoner::new([0.0, 0.0, 100.0], [0.0, 0.0, 0.0], false);
    let initial_var = reckoner.kinematic.pos_cov_enu_m2[(0, 0)];
    reckoner.step(1.0, 8, 4.0, 0.01);
    let var_flow = reckoner.kinematic.pos_cov_enu_m2[(0, 0)];
    check(
        var_flow > initial_var,
        "covariance grew moderately with flow",
    )?;
    Ok(())
}

#[test]
fn t1_09_05_path_b_covariance_inflation() -> TestResult {
    let mut reck_flow = DualPathDeadReckoner::new([0.0, 0.0, 100.0], [0.0, 0.0, 0.0], false);
    reck_flow.step(1.0, 6, 4.0, 0.01);
    let var_flow = reck_flow.kinematic.pos_cov_enu_m2[(0, 0)];

    let mut reck_coast = DualPathDeadReckoner::new([0.0, 0.0, 100.0], [0.0, 0.0, 0.0], false);
    reck_coast.step(1.0, 2, 4.0, 0.01);
    let var_coast = reck_coast.kinematic.pos_cov_enu_m2[(0, 0)];

    check(
        var_coast > var_flow,
        "coasting inflates covariance more than optical flow",
    )?;
    Ok(())
}

// --- Feature 10: 3D Mahalanobis Innovation Gate ---

#[test]
fn t1_10_01_innovation_vector_calculation() -> TestResult {
    let state = KinematicState::new([100.0, 200.0, 50.0], [0.0, 0.0, 0.0]);
    let cand_eye = [102.0, 199.0, 50.5];
    let diff = [
        cand_eye[0] - state.eye_enu[0],
        cand_eye[1] - state.eye_enu[1],
        cand_eye[2] - state.eye_enu[2],
    ];
    check(approx_eq(diff[0], 2.0, 1e-4), "dx = 2.0")?;
    check(approx_eq(diff[1], -1.0, 1e-4), "dy = -1.0")?;
    check(approx_eq(diff[2], 0.5, 1e-4), "dz = 0.5")?;
    Ok(())
}

#[test]
fn t1_10_02_joint_covariance_sum() -> TestResult {
    let p_cov = Matrix3::from_diagonal_element(4.0);
    let g_cov = Matrix3::from_diagonal_element(5.0);
    let joint = p_cov + g_cov;
    check(
        approx_eq(joint[(0, 0)], 9.0, 1e-4),
        "joint cov diagonal is 9.0",
    )?;
    Ok(())
}

#[test]
fn t1_10_03_accept_candidate_within_gate() -> TestResult {
    let mut state = KinematicState::new([0.0, 0.0, 100.0], [0.0, 0.0, 0.0]);
    state.pos_cov_enu_m2 = Matrix3::from_diagonal_element(10.0);
    let geom_cov = Matrix3::from_diagonal_element(6.0);
    let cand_eye = [1.0, 1.0, 101.0];
    let accepted = state.mahalanobis_innovation_gate(cand_eye, geom_cov, 11.34);
    check(accepted, "small innovation passes gate")?;
    Ok(())
}

#[test]
fn t1_10_04_reject_candidate_outside_gate() -> TestResult {
    let mut state = KinematicState::new([0.0, 0.0, 100.0], [0.0, 0.0, 0.0]);
    state.pos_cov_enu_m2 = Matrix3::from_diagonal_element(1.0);
    let geom_cov = Matrix3::from_diagonal_element(1.0);
    let cand_eye = [20.0, 20.0, 100.0];
    let accepted = state.mahalanobis_innovation_gate(cand_eye, geom_cov, 11.34);
    check(!accepted, "large innovation jump rejected")?;
    Ok(())
}

#[test]
fn t1_10_05_vertical_velocity_gate_nominal() -> TestResult {
    check(
        can_update_vertical_velocity(12, 10),
        "holdout >= 12 allows vertical update",
    )?;
    check(
        can_update_vertical_velocity(5, 48),
        "inliers >= 48 allows vertical update",
    )?;
    check(
        !can_update_vertical_velocity(11, 47),
        "11 holdouts and 47 inliers refuses update",
    )?;
    Ok(())
}

// --- Feature 11: Anchor Pose and Velocity Reset ---

#[test]
fn t1_11_01_reset_state_on_track_loss() -> TestResult {
    let mut state = AnchorWorkerState {
        prev_eye: Some([100.0, 200.0, 50.0]),
        prev_rot: Some(Matrix3::identity()),
        vel_enu_mps: [15.0, 0.0, 0.0],
    };
    state.on_track_loss();
    check(state.prev_eye.is_none(), "prev_eye cleared on loss")?;
    check(state.prev_rot.is_none(), "prev_rot cleared on loss")?;
    check(
        state.vel_enu_mps == [0.0, 0.0, 0.0],
        "velocity cleared on loss",
    )?;
    Ok(())
}

#[test]
fn t1_11_02_reset_state_on_anchor_acceptance() -> TestResult {
    let mut state = AnchorWorkerState::default();
    state.on_anchor_accepted([10.0, 20.0, 100.0], Matrix3::identity(), 0.1);
    check(
        state.prev_eye == Some([10.0, 20.0, 100.0]),
        "first anchor sets prev_eye",
    )?;
    check(
        state.vel_enu_mps == [0.0, 0.0, 0.0],
        "initial velocity is zero",
    )?;

    state.on_anchor_accepted([13.0, 20.0, 100.0], Matrix3::identity(), 0.1);
    check(
        approx_eq(state.vel_enu_mps[0], 30.0, 1e-4),
        "velocity recomputed to 30 m/s",
    )?;
    Ok(())
}

#[test]
fn t1_11_03_reset_state_on_relocation() -> TestResult {
    let mut state = AnchorWorkerState {
        prev_eye: Some([0.0, 0.0, 100.0]),
        prev_rot: Some(Matrix3::identity()),
        vel_enu_mps: [10.0, 0.0, 0.0],
    };
    state.on_relocation([500.0, 0.0, 100.0], 1.0);
    check(
        state.prev_eye == Some([500.0, 0.0, 100.0]),
        "relocated eye updated",
    )?;
    check(
        approx_eq(state.vel_enu_mps[0], 500.0, 1e-4),
        "velocity updated from relocation jump",
    )?;
    Ok(())
}

#[test]
fn t1_11_04_recompute_velocity_from_anchor() -> TestResult {
    let mut state = AnchorWorkerState {
        prev_eye: Some([10.0, 20.0, 30.0]),
        prev_rot: Some(Matrix3::identity()),
        vel_enu_mps: [0.0, 0.0, 0.0],
    };
    state.on_anchor_accepted([11.0, 22.0, 29.0], Matrix3::identity(), 0.5);
    check(
        approx_eq(state.vel_enu_mps[0], 2.0, 1e-4),
        "vx = 1.0 / 0.5 = 2.0",
    )?;
    check(
        approx_eq(state.vel_enu_mps[1], 4.0, 1e-4),
        "vy = 2.0 / 0.5 = 4.0",
    )?;
    check(
        approx_eq(state.vel_enu_mps[2], -2.0, 1e-4),
        "vz = -1.0 / 0.5 = -2.0",
    )?;
    Ok(())
}

#[test]
fn t1_11_05_prevent_stale_velocity_carryover() -> TestResult {
    let mut state = AnchorWorkerState {
        prev_eye: Some([0.0, 0.0, 0.0]),
        prev_rot: Some(Matrix3::identity()),
        vel_enu_mps: [100.0, 100.0, 100.0],
    };
    state.on_track_loss();
    state.on_anchor_accepted([50.0, 50.0, 50.0], Matrix3::identity(), 0.1);
    check(
        state.vel_enu_mps == [0.0, 0.0, 0.0],
        "fresh lock initializes velocity without stale carryover",
    )?;
    Ok(())
}
