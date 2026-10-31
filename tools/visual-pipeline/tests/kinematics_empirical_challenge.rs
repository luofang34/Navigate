//! Empirical stress test harness challenging KinematicState and VehicleDynamicsProfile.
//!
//! Evaluates boundary conditions:
//! - Extreme delta-t (1e-4 s to 60.0 s).
//! - High velocities (600 m/s to 800 m/s) and high accelerations (100 m/s^2).
//! - Monotonic covariance growth.
//! - Mahalanobis gate rejection of degenerate outlier jumps.
//! - Matrix singularity and non-finite inputs.

use nalgebra::{Matrix3, Vector3};
use navigate_visual_pipeline::corridor::{
    KinematicState, TrackingSearchStrategy, VehicleDynamicsProfile, compute_dt_s,
    kinematics::{MAX_DELTA_T_S, MIN_DELTA_T_S, determine_search_strategy},
};
use std::error::Error;

type TestResult = Result<(), Box<dyn Error>>;

const CHI2_GATE_3DOF_99: f64 = 11.345;

#[test]
fn test_extreme_dt_lower_bound_microsecond() -> TestResult {
    // 1e-4 s lower bound check
    let dt_min = compute_dt_s(Some(10.0001), Some(10.0), 30.0);
    assert!((dt_min - MIN_DELTA_T_S).abs() < 1e-9);

    // Below 1e-4 s clamps to MIN_DELTA_T_S
    let dt_sub = compute_dt_s(Some(10.000001), Some(10.0), 30.0);
    assert!((dt_sub - MIN_DELTA_T_S).abs() < 1e-9);

    // Negative dt or inverted timestamps fall back to default frame rate
    let dt_neg = compute_dt_s(Some(9.0), Some(10.0), 30.0);
    assert!((dt_neg - (1.0 / 30.0)).abs() < 1e-6);

    // Identical timestamps fall back to default frame rate
    let dt_zero = compute_dt_s(Some(10.0), Some(10.0), 60.0);
    assert!((dt_zero - (1.0 / 60.0)).abs() < 1e-6);

    // Position prediction at 1e-4 s with high velocity (800 m/s)
    let mut state = KinematicState::new([1000.0, 2000.0, 300.0], Matrix3::identity());
    state.vel_enu_mps = [800.0, 0.0, 0.0];
    let pred = state.predict_position(MIN_DELTA_T_S);
    let expected_x = 1000.0 + 800.0 * 1e-4; // 1000.08 m
    assert!((pred[0] - expected_x).abs() < 1e-9);
    assert!((pred[1] - 2000.0).abs() < 1e-9);

    // Covariance propagation at microstep dt
    let p_before = state.pos_cov_enu_m2[(0, 0)];
    let att_before = state.att_sigma_rad;
    state.propagate_covariance(MIN_DELTA_T_S, 100.0, 6.0);
    let p_after = state.pos_cov_enu_m2[(0, 0)];
    let att_after = state.att_sigma_rad;
    assert!(p_after > p_before);
    assert!(att_after > att_before);
    let expected_q = 0.5 * 100.0 * MIN_DELTA_T_S * MIN_DELTA_T_S;
    assert!((p_after - (p_before + expected_q)).abs() < 1e-12);

    Ok(())
}

#[test]
fn test_extreme_dt_upper_bound_blackout() -> TestResult {
    // 60.0 s upper bound check
    let dt_max = compute_dt_s(Some(65.0), Some(5.0), 30.0);
    assert!((dt_max - MAX_DELTA_T_S).abs() < 1e-9);

    // Beyond 60.0 s clamps to MAX_DELTA_T_S
    let dt_huge = compute_dt_s(Some(1000.0), Some(0.0), 30.0);
    assert!((dt_huge - MAX_DELTA_T_S).abs() < 1e-9);

    // Position prediction across 60.0 s gap at 600 m/s
    let mut state = KinematicState::new([0.0, 0.0, 500.0], Matrix3::identity());
    state.vel_enu_mps = [600.0, -200.0, 10.0];
    let pred = state.predict_position(MAX_DELTA_T_S);
    assert!((pred[0] - 36_000.0).abs() < 1e-6);
    assert!((pred[1] - (-12_000.0)).abs() < 1e-6);
    assert!((pred[2] - 1_100.0).abs() < 1e-6);

    // Covariance propagation across 60.0 s at 100 m/s^2 acceleration noise
    let p_init = state.pos_cov_enu_m2[(0, 0)];
    state.propagate_covariance(MAX_DELTA_T_S, 100.0, 6.0);
    let expected_pos_growth = 0.5 * 100.0 * 60.0 * 60.0; // 180,000 m^2
    assert!((state.pos_cov_enu_m2[(0, 0)] - (p_init + expected_pos_growth)).abs() < 1e-6);
    assert!((state.att_sigma_rad - (0.01 + 6.0 * 60.0)).abs() < 1e-6);

    Ok(())
}

#[test]
fn test_monotonic_covariance_inflation_over_time() -> TestResult {
    let mut state = KinematicState::new([500.0, 500.0, 100.0], Matrix3::identity());
    let dt_steps = [0.033, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0];
    let mut prev_cov_trace = state.pos_cov_enu_m2.trace();
    let mut prev_att = state.att_sigma_rad;

    for &dt in &dt_steps {
        state.propagate_covariance(dt, 15.0, 0.5);
        let cur_cov_trace = state.pos_cov_enu_m2.trace();
        let cur_att = state.att_sigma_rad;

        // Strictly monotonic increase
        assert!(cur_cov_trace > prev_cov_trace);
        assert!(cur_att > prev_att);

        // Covariance matrix must remain symmetric positive-definite
        let det = state.pos_cov_enu_m2.determinant();
        assert!(det > 0.0);
        assert!(state.pos_cov_enu_m2[(0, 0)] > 0.0);
        assert!(state.pos_cov_enu_m2[(1, 1)] > 0.0);
        assert!(state.pos_cov_enu_m2[(2, 2)] > 0.0);

        prev_cov_trace = cur_cov_trace;
        prev_att = cur_att;
    }

    Ok(())
}

#[test]
fn test_high_speed_and_acceleration_dynamics_profiles() -> TestResult {
    let hp = VehicleDynamicsProfile::high_performance();
    assert_eq!(hp.max_speed_mps, 800.0);
    assert_eq!(hp.max_vertical_speed_mps, 100.0);
    assert_eq!(hp.rms_accel_mps2, 100.0);
    assert_eq!(hp.max_angular_rate_rad_s, 6.0);

    // Max displacement formula: v * dt + 0.5 * a * dt^2
    let dt = 0.5;
    let expected_disp = 800.0 * 0.5 + 0.5 * 100.0 * 0.25; // 400 + 12.5 = 412.5 m
    assert!((hp.max_displacement(dt) - expected_disp).abs() < 1e-6);

    let expected_vdisp = 100.0 * 0.5 + 0.5 * 100.0 * 0.25; // 50 + 12.5 = 62.5 m
    assert!((hp.max_vertical_displacement(dt) - expected_vdisp).abs() < 1e-6);

    // Ground VO bounds scale with dynamics
    let (filter_gate, clamp_gate) = hp.ground_vo_bounds(dt);
    assert!(filter_gate >= clamp_gate);
    assert!((5.0..=1000.0).contains(&clamp_gate));

    // Velocity update gating under high performance envelope
    let mut state = KinematicState::new([0.0, 0.0, 100.0], Matrix3::identity());
    let reasonable_new_eye = [700.0 * 0.1, 0.0, 100.0];
    state.update_from_pose(reasonable_new_eye, Matrix3::identity(), 0.1, &hp);
    assert!((state.vel_enu_mps[0] - 700.0).abs() < 1e-6);

    // Extreme teleportation (> 1.5 * max_speed) is filtered from velocity absorption
    let teleport_eye = [reasonable_new_eye[0] + 5000.0, 0.0, 100.0];
    state.update_from_pose(teleport_eye, Matrix3::identity(), 0.1, &hp);
    // Velocity remains 700 m/s because 50,000 m/s candidate was filtered
    assert!((state.vel_enu_mps[0] - 700.0).abs() < 1e-6);

    Ok(())
}

#[test]
fn test_mahalanobis_gate_accepts_consistent_and_rejects_outliers() -> TestResult {
    let mut state = KinematicState::new([100.0, 200.0, 50.0], Matrix3::identity());
    state.pos_cov_enu_m2 = Matrix3::from_diagonal(&Vector3::new(4.0, 4.0, 2.0));
    let geom_cov = Matrix3::from_diagonal(&Vector3::new(2.0, 2.0, 1.0));

    // Candidate 1: Close to prediction (within 1-sigma)
    let cand_consistent = [101.0, 201.0, 50.5];
    assert!(state.mahalanobis_innovation_gate(cand_consistent, geom_cov, CHI2_GATE_3DOF_99));

    // Candidate 2: Boundary test
    // Total var = [6, 6, 3]. Mahalanobis d^2 = dx^2/6 + dy^2/6 + dz^2/3.
    // If dx = sqrt(6 * 11.0) = 8.124, d^2 = 11.0 <= 11.345 -> accept
    let cand_boundary = [100.0 + (6.0 * 11.0_f64).sqrt(), 200.0, 50.0];
    assert!(state.mahalanobis_innovation_gate(cand_boundary, geom_cov, CHI2_GATE_3DOF_99));

    // Candidate 3: Just outside boundary
    // If dx = sqrt(6 * 12.0) = 8.485, d^2 = 12.0 > 11.345 -> reject
    let cand_outside = [100.0 + (6.0 * 12.0_f64).sqrt(), 200.0, 50.0];
    assert!(!state.mahalanobis_innovation_gate(cand_outside, geom_cov, CHI2_GATE_3DOF_99));

    // Candidate 4: Massive horizontal jump outlier (150m away)
    let cand_horiz_outlier = [250.0, 200.0, 50.0];
    assert!(!state.mahalanobis_innovation_gate(cand_horiz_outlier, geom_cov, CHI2_GATE_3DOF_99));

    // Candidate 5: Vertical outlier jump (50m jump in z)
    let cand_vert_outlier = [100.0, 200.0, 100.0];
    assert!(!state.mahalanobis_innovation_gate(cand_vert_outlier, geom_cov, CHI2_GATE_3DOF_99));

    Ok(())
}

#[test]
fn test_mahalanobis_gate_matrix_degeneracy_and_robustness() -> TestResult {
    let state = KinematicState::new([0.0, 0.0, 100.0], Matrix3::identity());

    // Degenerate geometry covariance: zero matrix (pure prior check)
    let zero_geom_cov = Matrix3::zeros();
    let valid_cand = [0.5, 0.5, 100.5];
    assert!(state.mahalanobis_innovation_gate(valid_cand, zero_geom_cov, CHI2_GATE_3DOF_99));

    // Singular rank-1 geometry covariance
    let mut singular_geom = Matrix3::zeros();
    singular_geom[(0, 0)] = 5.0; // only x variance is defined
    assert!(state.mahalanobis_innovation_gate(valid_cand, singular_geom, CHI2_GATE_3DOF_99));

    // Degenerate joint covariance: if both state and geom covariance are zero
    let mut degenerate_state = state.clone();
    degenerate_state.pos_cov_enu_m2 = Matrix3::zeros();
    let all_zero_geom = Matrix3::zeros();
    // Inversion fails safely: returns false without panic
    assert!(!degenerate_state.mahalanobis_innovation_gate(
        valid_cand,
        all_zero_geom,
        CHI2_GATE_3DOF_99
    ));

    // Non-finite candidate coordinates (NaN / Inf)
    let nan_cand = [f64::NAN, 0.0, 100.0];
    assert!(!state.mahalanobis_innovation_gate(nan_cand, zero_geom_cov, CHI2_GATE_3DOF_99));

    let inf_cand = [f64::INFINITY, 0.0, 100.0];
    assert!(!state.mahalanobis_innovation_gate(inf_cand, zero_geom_cov, CHI2_GATE_3DOF_99));

    Ok(())
}

#[test]
fn test_adaptive_search_strategy_threshold_boundaries() -> TestResult {
    // 3 * sigma_px <= 64 -> RadiusSearch clamped to [24.0, 96.0]
    // Case 1: Very low uncertainty (sigma_px = 5.0 -> 3 * 5 = 15 px -> clamped to 24.0 px)
    let strat_low = determine_search_strategy(5.0);
    match strat_low {
        TrackingSearchStrategy::RadiusSearch { radius_px } => {
            assert!((radius_px - 24.0).abs() < 1e-6);
        }
        _ => return Err("Expected RadiusSearch for low uncertainty".into()),
    }

    // Case 2: Mid uncertainty (sigma_px = 15.0 -> 3 * 15 = 45.0 px)
    let strat_mid = determine_search_strategy(15.0);
    match strat_mid {
        TrackingSearchStrategy::RadiusSearch { radius_px } => {
            assert!((radius_px - 45.0).abs() < 1e-6);
        }
        _ => return Err("Expected RadiusSearch for mid uncertainty".into()),
    }

    // Case 3: Exact threshold boundary (3 * sigma_px = 64.0 -> sigma_px = 64.0 / 3.0)
    let strat_boundary = determine_search_strategy(64.0 / 3.0);
    match strat_boundary {
        TrackingSearchStrategy::RadiusSearch { radius_px } => {
            assert!((radius_px - 64.0).abs() < 1e-6);
        }
        _ => return Err("Expected RadiusSearch at exact 64px boundary".into()),
    }

    // Case 4: Above threshold (sigma_px = 22.0 -> 3 * 22 = 66.0 px > 64 px)
    let strat_high = determine_search_strategy(22.0);
    assert_eq!(strat_high, TrackingSearchStrategy::GlobalFrustumMatch);

    Ok(())
}

#[test]
fn test_pixel_uncertainty_projection_depth_and_rotation_geometry() -> TestResult {
    let mut state = KinematicState::new([0.0, 0.0, 200.0], Matrix3::identity());
    state.pos_cov_enu_m2 = Matrix3::from_diagonal(&Vector3::new(4.0, 4.0, 1.0));
    state.att_sigma_rad = 0.005;

    let fx = 800.0;
    let fy = 800.0;

    // Projected uncertainty decreases as depth increases
    let sigma_near = state.project_pixel_uncertainty(fx, fy, 50.0);
    let sigma_far = state.project_pixel_uncertainty(fx, fy, 500.0);
    assert!(sigma_near > sigma_far);

    // Negative and zero depths are safely guarded by max(1.0)
    let sigma_zero = state.project_pixel_uncertainty(fx, fy, 0.0);
    let sigma_neg = state.project_pixel_uncertainty(fx, fy, -100.0);
    assert!(sigma_zero.is_finite() && sigma_zero > 0.0);
    assert_eq!(sigma_zero, sigma_neg);

    // Rotated camera: 90-degree yaw swap
    let yaw_90 = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    state.rot_world_to_cam = yaw_90;
    let sigma_rotated = state.project_pixel_uncertainty(fx, fy, 200.0);
    assert!(sigma_rotated.is_finite() && sigma_rotated > 0.0);

    Ok(())
}
