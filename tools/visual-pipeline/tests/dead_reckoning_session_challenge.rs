//! Empirical stress tests for dead-reckoning and session management.
//!
//! This test suite challenges:
//! - DualPathDeadReckoner path selection and kinematic coasting.
//! - Covariance inflation differences under zero versus six or more ground flow vectors.
//! - VisualCorridorSession frame registration with monotonic timestamps.
//! - Out-of-order and duplicate timestamp rejection.
//! - Continuity epoch advancement upon unobservable metric scale.
//! - SegmentStart::AnchorAfterGap re-acquisition following tracking gaps.
//! - Mahalanobis innovation gating and vertical velocity update criteria.

use nalgebra::{Matrix3, Matrix6, Translation3, UnitQuaternion};
use navigate_visual::{LocalFrame, MapRevision};
use navigate_visual_pipeline::corridor::{
    AnchorWorkerState, DeadReckonPath, DualPathDeadReckoner, KinematicState, VisualCorridorSession,
    can_update_vertical_velocity,
    kinematics::{CHI2_GATE_3DOF_99, MAX_DELTA_T_S, MIN_DELTA_T_S},
};
use navigate_visual_session::{
    AnchorDecision, AnchorObservation, ContinuityEpoch, FrameKey, MapReliability, Pose,
    RegionChange, ScaleSource, SegmentStart, SessionError, SessionEvent, SurfaceModel,
};
use std::error::Error;

type TestResult = Result<(), Box<dyn Error>>;

fn make_test_anchor(frame: FrameKey, pos: [f64; 3]) -> Result<AnchorObservation, Box<dyn Error>> {
    let local_frame = LocalFrame::anchor_mercator(47.0, 11.0).map_err(|e| format!("{e:?}"))?;
    let map = MapRevision {
        release_id: "test-map".into(),
        manifest_sha256: "b".repeat(64),
    };
    let pose = Pose::from_parts(
        Translation3::new(pos[0], pos[1], pos[2]),
        UnitQuaternion::identity(),
    );
    let geom = Matrix6::identity() * 0.1;
    Ok(AnchorObservation {
        frame,
        observation_sha256: format!("{:064x}", frame.index + 1),
        map,
        local_frame,
        pose,
        geometry_covariance: geom,
        backend: "test-matcher".into(),
        reliability: MapReliability {
            horizontal_m: 1.0,
            vertical_m: 2.0,
            imagery_age_years: Some(1.0),
            age_growth_m_per_year: 0.1,
            unknown_age_years: 5.0,
            surface: SurfaceModel::Surface,
            change: RegionChange::Unchanged,
        },
    })
}

#[test]
fn test_dual_path_selection_with_and_without_external_sensors() -> TestResult {
    // With external sensors, dead reckoner must select Path A.
    let mut reck_a = DualPathDeadReckoner::new([100.0, 200.0, 50.0], [15.0, -5.0, 0.0], true);
    for flow_count in [0, 1, 5, 6, 12, 100] {
        let est = reck_a.step(0.1, flow_count, 4.0, 0.02);
        assert_eq!(est.path, DeadReckonPath::PathA);
    }

    // Without external sensors, dead reckoner must select Path B.
    let mut reck_b = DualPathDeadReckoner::new([100.0, 200.0, 50.0], [15.0, -5.0, 0.0], false);
    for flow_count in [0, 1, 5, 6, 12, 100] {
        let est = reck_b.step(0.1, flow_count, 4.0, 0.02);
        assert_eq!(est.path, DeadReckonPath::PathB);
    }

    Ok(())
}

#[test]
fn test_covariance_inflation_under_zero_vs_six_flow_vectors() -> TestResult {
    let dt = 1.0;
    let a_rms = 5.0;
    let omega_rms = 0.1;

    // Test zero flow vectors: full noise inflation.
    let mut reck_zero = DualPathDeadReckoner::new([0.0, 0.0, 100.0], [0.0, 0.0, 0.0], false);
    let init_var = reck_zero.kinematic.pos_cov_enu_m2[(0, 0)];
    let init_att = reck_zero.kinematic.att_sigma_rad;
    reck_zero.step(dt, 0, a_rms, omega_rms);
    let var_zero_delta = reck_zero.kinematic.pos_cov_enu_m2[(0, 0)] - init_var;
    let att_zero_delta = reck_zero.kinematic.att_sigma_rad - init_att;

    // Expected zero flow inflation: 0.5 * a_rms * dt^2 = 0.5 * 5.0 * 1.0 = 2.5
    assert!((var_zero_delta - 2.5).abs() < 1e-6);
    assert!((att_zero_delta - 0.1).abs() < 1e-6);

    // Test five flow vectors: still below threshold of six, uses full noise.
    let mut reck_five = DualPathDeadReckoner::new([0.0, 0.0, 100.0], [0.0, 0.0, 0.0], false);
    reck_five.step(dt, 5, a_rms, omega_rms);
    let var_five_delta = reck_five.kinematic.pos_cov_enu_m2[(0, 0)] - init_var;
    assert!((var_five_delta - 2.5).abs() < 1e-6);

    // Test six flow vectors: meets threshold, noise is scaled by 0.8.
    let mut reck_six = DualPathDeadReckoner::new([0.0, 0.0, 100.0], [0.0, 0.0, 0.0], false);
    reck_six.step(dt, 6, a_rms, omega_rms);
    let var_six_delta = reck_six.kinematic.pos_cov_enu_m2[(0, 0)] - init_var;
    let att_six_delta = reck_six.kinematic.att_sigma_rad - init_att;

    // Expected six flow inflation: 0.5 * (a_rms * 0.8) * dt^2 = 0.5 * 4.0 * 1.0 = 2.0
    assert!((var_six_delta - 2.0).abs() < 1e-6);
    assert!((att_six_delta - 0.08).abs() < 1e-6);

    // Test ten flow vectors: remains at reduced noise.
    let mut reck_ten = DualPathDeadReckoner::new([0.0, 0.0, 100.0], [0.0, 0.0, 0.0], false);
    reck_ten.step(dt, 10, a_rms, omega_rms);
    let var_ten_delta = reck_ten.kinematic.pos_cov_enu_m2[(0, 0)] - init_var;
    assert!((var_ten_delta - 2.0).abs() < 1e-6);

    // Verify inflation ratio between zero flow and six flow vectors is exactly 1.25.
    let ratio = var_zero_delta / var_six_delta;
    assert!((ratio - 1.25).abs() < 1e-6);

    Ok(())
}

#[test]
fn test_dual_path_dead_reckoner_interval_clamping_and_coasting() -> TestResult {
    let mut reck = DualPathDeadReckoner::new([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], false);

    // Interval below MIN_DELTA_T_S must clamp to 1e-4 s.
    let est_micro = reck.step(1e-8, 0, 1.0, 0.01);
    let disp_micro = est_micro.eye_enu[0];
    assert!((disp_micro - (10.0 * MIN_DELTA_T_S)).abs() < 1e-9);

    // Interval above MAX_DELTA_T_S must clamp to 60.0 s.
    let mut reck_large = DualPathDeadReckoner::new([0.0, 0.0, 0.0], [10.0, 0.0, 0.0], false);
    let est_large = reck_large.step(500.0, 0, 1.0, 0.01);
    let disp_large = est_large.eye_enu[0];
    assert!((disp_large - (10.0 * MAX_DELTA_T_S)).abs() < 1e-6);

    // Covariance matrix must remain symmetric and positive definite across multiple steps.
    for _ in 0..50 {
        reck.step(0.1, 8, 2.0, 0.01);
    }
    let cov = reck.kinematic.pos_cov_enu_m2;
    assert!((cov[(0, 1)] - cov[(1, 0)]).abs() < 1e-12);
    assert!(cov[(0, 0)] > 0.0 && cov[(1, 1)] > 0.0 && cov[(2, 2)] > 0.0);

    Ok(())
}

#[test]
fn test_session_frame_registration_nominal_and_out_of_order() -> TestResult {
    let mut session = VisualCorridorSession::new()?;

    // Register nominal frames with monotonic timestamps.
    session.register_frame(1_000_000_000)?;
    assert_eq!(session.next_seq, 1);
    session.register_frame(1_033_333_333)?;
    assert_eq!(session.next_seq, 2);

    // Out-of-order timestamp must return error.
    let res_backward = session.register_frame(900_000_000);
    assert!(res_backward.is_err());

    // Duplicate timestamp must return error.
    let res_duplicate = session.register_frame(1_033_333_333);
    assert!(res_duplicate.is_err());

    // Registering a subsequent valid monotonic timestamp must succeed without corruption.
    let res_valid = session.register_frame(1_066_666_666);
    assert!(res_valid.is_ok());

    Ok(())
}

#[test]
fn test_session_epoch_advancement_on_unobservable_scale() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    session.register_frame(100_000_000)?;
    session.register_frame(200_000_000)?;

    assert_eq!(session.current_epoch, ContinuityEpoch(0));

    // Motion with unmetric scale must fail and advance continuity epoch.
    let res_unknown = session.apply_motion_step(0.1, Pose::identity(), ScaleSource::Unknown);
    assert!(matches!(
        res_unknown,
        Err(SessionError::ScaleUnobservable { .. })
    ));
    assert_eq!(session.current_epoch, ContinuityEpoch(1));

    // Registering frames in the new epoch must use updated epoch key.
    session.register_frame(300_000_000)?;
    session.register_frame(400_000_000)?;
    let key = session
        .last_registered_key
        .ok_or("missing registered key")?;
    assert_eq!(key.continuity, ContinuityEpoch(1));

    // Another unobservable scale motion advances epoch again.
    let res_unknown2 = session.apply_motion_step(0.1, Pose::identity(), ScaleSource::Unknown);
    assert!(res_unknown2.is_err());
    assert_eq!(session.current_epoch, ContinuityEpoch(2));

    Ok(())
}

#[test]
fn test_session_anchor_after_gap_reacquisition() -> TestResult {
    let mut session = VisualCorridorSession::new()?;

    // Register frame 0 and submit initial anchor.
    session.register_frame(100_000_000)?;
    let key0 = session.last_registered_key.ok_or("missing frame 0 key")?;
    let anchor0 = make_test_anchor(key0, [0.0, 0.0, 100.0])?;
    let dec0 = session.submit_anchor(anchor0)?;
    assert!(matches!(dec0, AnchorDecision::Accepted { .. }));

    // Register frame 1 and link via motion.
    session.register_frame(200_000_000)?;
    let map = MapRevision {
        release_id: "test-map".into(),
        manifest_sha256: "b".repeat(64),
    };
    let step_outcome =
        session.apply_motion_step(0.1, Pose::identity(), ScaleSource::MapDepth { map });
    assert!(step_outcome.is_ok());

    // Register frame 2 without odometry motion (simulating a gap in tracking).
    session.register_frame(500_000_000)?;
    let key2 = session.last_registered_key.ok_or("missing frame 2 key")?;

    // Submit anchor for frame 2 after gap.
    let anchor2 = make_test_anchor(key2, [10.0, 0.0, 100.0])?;
    let dec2 = session.submit_anchor(anchor2)?;
    assert!(matches!(dec2, AnchorDecision::Accepted { .. }));

    // Drain events and verify SegmentStart::AnchorAfterGap was recorded.
    let events = session.session.drain_events().events;
    let found_gap_event = events.iter().any(|e| {
        matches!(
            e,
            SessionEvent::SegmentStarted {
                first,
                reason: SegmentStart::AnchorAfterGap,
                ..
            } if *first == key2
        )
    });
    assert!(
        found_gap_event,
        "AnchorAfterGap event must be emitted for gap frame"
    );

    Ok(())
}

#[test]
fn test_mahalanobis_innovation_gate_boundary_behavior() -> TestResult {
    let state = KinematicState::new([100.0, 200.0, 50.0], Matrix3::identity());
    let geom_cov = Matrix3::identity() * 2.0;

    // Small displacement must pass gate.
    let close_eye = [100.5, 200.2, 50.1];
    assert!(state.mahalanobis_innovation_gate(close_eye, geom_cov, CHI2_GATE_3DOF_99));

    // Large outlier displacement must be rejected by gate.
    let far_eye = [150.0, 250.0, 80.0];
    assert!(!state.mahalanobis_innovation_gate(far_eye, geom_cov, CHI2_GATE_3DOF_99));

    // Non-finite coordinates must return false without panic.
    let nan_eye = [f64::NAN, 200.0, 50.0];
    assert!(!state.mahalanobis_innovation_gate(nan_eye, geom_cov, CHI2_GATE_3DOF_99));

    let inf_eye = [f64::INFINITY, 200.0, 50.0];
    assert!(!state.mahalanobis_innovation_gate(inf_eye, geom_cov, CHI2_GATE_3DOF_99));

    Ok(())
}

#[test]
fn test_vertical_velocity_gating_truth_table() -> TestResult {
    // Holdout count threshold is 12; inlier count threshold is 48.
    for holdout in 0..=15 {
        for inliers in 0..=55 {
            let expected = holdout >= 12 || inliers >= 48;
            assert_eq!(can_update_vertical_velocity(holdout, inliers), expected);
        }
    }
    Ok(())
}

#[test]
fn test_anchor_worker_state_transitions_and_velocity_recomputation() -> TestResult {
    let mut state = AnchorWorkerState::default();
    assert_eq!(state.prev_eye, None);
    assert_eq!(state.vel_enu_mps, [0.0; 3]);

    // First anchor accepted: stores previous eye and keeps zero velocity.
    state.on_anchor_accepted([50.0, 100.0, 200.0], Matrix3::identity(), 0.1);
    assert_eq!(state.prev_eye, Some([50.0, 100.0, 200.0]));
    assert_eq!(state.vel_enu_mps, [0.0; 3]);

    // Second anchor accepted across dt = 0.5 s: computes velocity.
    state.on_anchor_accepted([60.0, 100.0, 200.0], Matrix3::identity(), 0.5);
    assert_eq!(state.prev_eye, Some([60.0, 100.0, 200.0]));
    assert!((state.vel_enu_mps[0] - 20.0).abs() < 1e-6);
    assert!((state.vel_enu_mps[1] - 0.0).abs() < 1e-6);

    // Track loss: clears previous eye and resets velocity.
    state.on_track_loss();
    assert_eq!(state.prev_eye, None);
    assert_eq!(state.vel_enu_mps, [0.0; 3]);

    // Acceptance after track loss re-initializes pose cleanly.
    state.on_anchor_accepted([300.0, 400.0, 50.0], Matrix3::identity(), 0.1);
    assert_eq!(state.prev_eye, Some([300.0, 400.0, 50.0]));
    assert_eq!(state.vel_enu_mps, [0.0; 3]);

    // Relocation: clears state and recomputes velocity without stale carryover.
    state.on_relocation([500.0, 400.0, 50.0], 2.0);
    assert_eq!(state.prev_eye, Some([500.0, 400.0, 50.0]));
    assert!((state.vel_enu_mps[0] - 100.0).abs() < 1e-6);
    assert!((state.vel_enu_mps[1] - 0.0).abs() < 1e-6);

    Ok(())
}
