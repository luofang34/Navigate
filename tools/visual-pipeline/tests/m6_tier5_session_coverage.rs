//! Milestone 6 Tier 5 dead-reckoning, session relocation, and focal-sweep coverage tests.

use nalgebra::Matrix3;
use navigate_visual_pipeline::{
    corridor::{
        AnchorWorkerState, DeadReckonPath, DualPathDeadReckoner, can_update_vertical_velocity,
        session::VisualCorridorSession,
    },
    stages::focal::{DEFAULT_FOCAL_MM, FOCAL_SWEEP_MULTIPLIERS, FocalLengthEstimator},
};
use std::error::Error;

type TestResult = Result<(), Box<dyn Error>>;

#[test]
fn test_tier5_dead_reckoning_dynamic_path_transitions() -> TestResult {
    let mut reck = DualPathDeadReckoner::new([100.0, 200.0, 50.0], [10.0, 0.0, 0.0], true);

    // Initial step with external sensors must use Path A.
    let est_a1 = reck.step(0.1, 0, 4.0, 0.05);
    assert_eq!(est_a1.path, DeadReckonPath::PathA);
    assert!((est_a1.eye_enu[0] - 101.0).abs() < 1e-6);

    // Sensor loss transitions dead reckoner to Path B.
    reck.external_sensors_available = false;
    let var_before = reck.kinematic.pos_cov_enu_m2[(0, 0)];

    // Path B with 3 flow vectors (< 6) applies full noise inflation.
    let est_b_low = reck.step(1.0, 3, 4.0, 0.05);
    assert_eq!(est_b_low.path, DeadReckonPath::PathB);
    let var_delta_low = reck.kinematic.pos_cov_enu_m2[(0, 0)] - var_before;
    assert!((var_delta_low - 2.0).abs() < 1e-6);

    // Path B with 8 flow vectors (>= 6) applies reduced 0.8x noise inflation.
    let var_before_high = reck.kinematic.pos_cov_enu_m2[(0, 0)];
    let est_b_high = reck.step(1.0, 8, 4.0, 0.05);
    assert_eq!(est_b_high.path, DeadReckonPath::PathB);
    let var_delta_high = reck.kinematic.pos_cov_enu_m2[(0, 0)] - var_before_high;
    assert!((var_delta_high - 1.6).abs() < 1e-6);

    // Sensor restoration returns to Path A.
    reck.external_sensors_available = true;
    let est_a2 = reck.step(0.1, 0, 4.0, 0.05);
    assert_eq!(est_a2.path, DeadReckonPath::PathA);
    Ok(())
}

#[test]
fn test_tier5_session_relocation_and_vertical_velocity_gating() -> TestResult {
    let mut session = VisualCorridorSession::new()?;

    // Relocation with positive dt_s recomputes velocity.
    let eye1 = [500.0, 600.0, 100.0];
    session.handle_relocation(eye1, Matrix3::identity(), 0.5);
    assert_eq!(session.dead_reckoner.kinematic.eye_enu, eye1);
    assert!((session.dead_reckoner.kinematic.vel_enu_mps[0] - 1000.0).abs() < 1e-6);

    // Relocation with zero dt_s sets velocity to zero without zero division.
    let eye2 = [700.0, 800.0, 120.0];
    session.handle_relocation(eye2, Matrix3::identity(), 0.0);
    assert_eq!(session.dead_reckoner.kinematic.eye_enu, eye2);
    assert_eq!(session.dead_reckoner.kinematic.vel_enu_mps, [0.0, 0.0, 0.0]);

    // Truth table boundaries for vertical velocity update gating.
    assert!(!can_update_vertical_velocity(11, 47));
    assert!(can_update_vertical_velocity(12, 0));
    assert!(can_update_vertical_velocity(0, 48));
    assert!(can_update_vertical_velocity(12, 48));

    // Anchor worker state transitions:
    let mut worker_state = AnchorWorkerState::default();
    worker_state.on_anchor_accepted([100.0, 100.0, 50.0], Matrix3::identity(), 0.0);
    assert_eq!(worker_state.vel_enu_mps, [0.0, 0.0, 0.0]);
    worker_state.on_anchor_accepted([110.0, 100.0, 50.0], Matrix3::identity(), 0.5);
    assert!((worker_state.vel_enu_mps[0] - 20.0).abs() < 1e-6);

    worker_state.on_track_loss();
    assert_eq!(worker_state.prev_eye, None);
    assert_eq!(worker_state.vel_enu_mps, [0.0, 0.0, 0.0]);
    Ok(())
}

#[test]
fn test_tier5_focal_length_sweep_and_locking() -> TestResult {
    let mut estimator = FocalLengthEstimator::default();
    assert_eq!(estimator.default_focal_mm, DEFAULT_FOCAL_MM);
    assert_eq!(estimator.sweep_multipliers, FOCAL_SWEEP_MULTIPLIERS);
    assert_eq!(estimator.locked_focal_px, None);

    let baseline = estimator.baseline_focal_px(1920);
    let expected_baseline = (DEFAULT_FOCAL_MM / 36.0) * 1920.0;
    assert!((baseline - expected_baseline).abs() < 1e-6);

    // Lowest RMSE score at index 1 selects 0.55 multiplier and locks focal length.
    let scores = [3.5, 0.9, 2.8, 4.2];
    let chosen = estimator.select_multiplier(&scores, baseline);
    let expected_chosen = baseline * 0.55;
    assert!((chosen - expected_chosen).abs() < 1e-6);
    assert_eq!(estimator.locked_focal_px, Some(expected_chosen));
    Ok(())
}
