//! Empirical challenge test suite for Milestone 2 features:
//! 1. 3D Mahalanobis innovation gating with extreme jumps, ill-conditioned matrices, and singular geometry covariance.
//! 2. Vertical velocity update gating (can_update_vertical_velocity) and velocity preservation.
//! 3. AnchorWorkerState lifecycle transitions (on_track_loss, on_anchor_accepted, on_relocation).

use nalgebra::{Matrix3, Rotation3, Vector3};
use navigate_visual_pipeline::corridor::{
    AnchorWorkerState, KinematicState, VehicleDynamicsProfile, can_update_vertical_velocity,
    kinematics::CHI2_GATE_3DOF_99,
};
use std::error::Error;

type TestResult = Result<(), Box<dyn Error>>;

#[test]
fn test_mahalanobis_gate_extreme_jumps() -> TestResult {
    let state = KinematicState::new([1000.0, 2000.0, 300.0], Matrix3::identity());
    let geom_cov = Matrix3::identity() * 2.0;

    // Sub-millimeter position jump is accepted
    let micro_jump = [1000.0001, 2000.0001, 300.0001];
    assert!(state.mahalanobis_innovation_gate(micro_jump, geom_cov, CHI2_GATE_3DOF_99));

    // Moderate jump within gate is accepted (total variance per axis = 1 + 2 = 3 m^2)
    // 3 * 3.0 m^2 = 9.0 <= 11.345
    let moderate_jump = [1001.0, 2001.0, 3001.0 - 2700.0];
    assert!(state.mahalanobis_innovation_gate(moderate_jump, geom_cov, CHI2_GATE_3DOF_99));

    // Extreme coordinate jump of 10 km is rejected
    let extreme_jump = [11000.0, 2000.0, 300.0];
    assert!(!state.mahalanobis_innovation_gate(extreme_jump, geom_cov, CHI2_GATE_3DOF_99));

    // Megameter jump (10^6 m) is rejected
    let mega_jump = [1_000_000.0, 2000.0, 300.0];
    assert!(!state.mahalanobis_innovation_gate(mega_jump, geom_cov, CHI2_GATE_3DOF_99));

    // Non-finite candidate coordinates return false without panicking
    let nan_x = [f64::NAN, 2000.0, 300.0];
    assert!(!state.mahalanobis_innovation_gate(nan_x, geom_cov, CHI2_GATE_3DOF_99));

    let inf_y = [1000.0, f64::INFINITY, 300.0];
    assert!(!state.mahalanobis_innovation_gate(inf_y, geom_cov, CHI2_GATE_3DOF_99));

    let neg_inf_z = [1000.0, 2000.0, f64::NEG_INFINITY];
    assert!(!state.mahalanobis_innovation_gate(neg_inf_z, geom_cov, CHI2_GATE_3DOF_99));

    Ok(())
}

#[test]
fn test_mahalanobis_gate_ill_conditioned_covariance() -> TestResult {
    let mut state = KinematicState::new([0.0, 0.0, 0.0], Matrix3::identity());

    // Highly anisotropic covariance: condition number 10^14
    // Variance in X is very small (1e-7 m^2), Y is 1.0 m^2, Z is large (1e7 m^2)
    state.pos_cov_enu_m2 = Matrix3::from_diagonal(&Vector3::new(1e-7, 1.0, 1e7));
    let zero_geom = Matrix3::zeros();

    // In X axis: variance is 1e-7. A 0.01 m jump yields d^2 = 1e-4 / 1e-7 = 1000 > 11.345 -> reject
    let x_stiff_jump = [0.01, 0.0, 0.0];
    assert!(!state.mahalanobis_innovation_gate(x_stiff_jump, zero_geom, CHI2_GATE_3DOF_99));

    // In X axis: a 0.0005 m jump yields d^2 = 2.5e-7 / 1e-7 = 2.5 <= 11.345 -> accept
    let x_tiny_jump = [0.0005, 0.0, 0.0];
    assert!(state.mahalanobis_innovation_gate(x_tiny_jump, zero_geom, CHI2_GATE_3DOF_99));

    // In Z axis: variance is 1e7. A 2000 m jump yields d^2 = 4e6 / 1e7 = 0.4 <= 11.345 -> accept
    let z_soft_jump = [0.0, 0.0, 2000.0];
    assert!(state.mahalanobis_innovation_gate(z_soft_jump, zero_geom, CHI2_GATE_3DOF_99));

    // Ultra-small covariance: 1e-12 * I
    state.pos_cov_enu_m2 = Matrix3::identity() * 1e-12;
    let ultra_small_geom = Matrix3::identity() * 1e-12;
    // Total variance = 2e-12
    let jump_inside = [1e-6, 0.0, 0.0]; // d^2 = 1e-12 / 2e-12 = 0.5 <= 11.345
    assert!(state.mahalanobis_innovation_gate(jump_inside, ultra_small_geom, CHI2_GATE_3DOF_99));

    let jump_outside = [1e-5, 0.0, 0.0]; // d^2 = 1e-10 / 2e-12 = 50 > 11.345
    assert!(!state.mahalanobis_innovation_gate(jump_outside, ultra_small_geom, CHI2_GATE_3DOF_99));

    // Ultra-large covariance: 1e12 * I
    state.pos_cov_enu_m2 = Matrix3::identity() * 1e12;
    let large_jump = [500_000.0, 500_000.0, 0.0]; // d^2 = 5e11 / 1e12 = 0.5 <= 11.345
    assert!(state.mahalanobis_innovation_gate(large_jump, zero_geom, CHI2_GATE_3DOF_99));

    Ok(())
}

#[test]
fn test_mahalanobis_gate_singular_geometry_covariance() -> TestResult {
    let state = KinematicState::new([100.0, 100.0, 500.0], Matrix3::identity());

    // Case 1: Zero geometry covariance (rank 0)
    let rank0_geom = Matrix3::zeros();
    let cand_close = [100.5, 100.5, 500.5];
    assert!(state.mahalanobis_innovation_gate(cand_close, rank0_geom, CHI2_GATE_3DOF_99));

    // Case 2: Rank-1 geometry covariance along X direction (collinear rays)
    let mut rank1_geom = Matrix3::zeros();
    rank1_geom[(0, 0)] = 25.0;
    // Joint variance in X is 1.0 + 25.0 = 26.0, Y is 1.0, Z is 1.0
    // Jump in X of 10m: d^2 = 100 / 26 = 3.84 <= 11.345 -> accept
    let cand_x_disp = [110.0, 100.0, 500.0];
    assert!(state.mahalanobis_innovation_gate(cand_x_disp, rank1_geom, CHI2_GATE_3DOF_99));
    // Jump in Y of 10m: d^2 = 100 / 1 = 100 > 11.345 -> reject
    let cand_y_disp = [100.0, 110.0, 500.0];
    assert!(!state.mahalanobis_innovation_gate(cand_y_disp, rank1_geom, CHI2_GATE_3DOF_99));

    // Case 3: Rank-2 geometry covariance (coplanar rays, unobservable depth / Z)
    let mut rank2_geom = Matrix3::zeros();
    rank2_geom[(0, 0)] = 16.0;
    rank2_geom[(1, 1)] = 16.0;
    // Z variance is purely from state prior (1.0)
    let cand_z_outside = [100.0, 100.0, 505.0]; // d^2 = 25 / 1 = 25 > 11.345
    assert!(!state.mahalanobis_innovation_gate(cand_z_outside, rank2_geom, CHI2_GATE_3DOF_99));

    // Case 4: Completely singular joint covariance (both state and geom are zero)
    let mut singular_state = state.clone();
    singular_state.pos_cov_enu_m2 = Matrix3::zeros();
    assert!(!singular_state.mahalanobis_innovation_gate(cand_close, rank0_geom, CHI2_GATE_3DOF_99));

    // Case 5: Non-finite entries in geometry covariance
    let mut nan_geom = Matrix3::identity();
    nan_geom[(0, 0)] = f64::NAN;
    assert!(!state.mahalanobis_innovation_gate(cand_close, nan_geom, CHI2_GATE_3DOF_99));

    Ok(())
}

#[test]
fn test_mahalanobis_gate_exact_boundary_and_rotation_invariance() -> TestResult {
    let state = KinematicState::new([0.0, 0.0, 0.0], Matrix3::identity());
    let sigmas = Vector3::new(2.0, 3.0, 4.0);
    let diag_cov = Matrix3::from_diagonal(&sigmas.map(|s| s * s));
    let zero_geom = Matrix3::zeros();

    // Point on gate ellipsoid boundary: (dx/sx)^2 + (dy/sy)^2 + (dz/sz)^2 = chi2_gate
    let gate = CHI2_GATE_3DOF_99;
    // Let alpha = 0.5, beta = 0.3, gamma = 0.2 (sum = 1.0)
    let dx = sigmas.x * (gate * 0.5).sqrt();
    let dy = sigmas.y * (gate * 0.3).sqrt();
    let dz = sigmas.z * (gate * 0.2).sqrt();

    let mut boundary_state = state.clone();
    boundary_state.pos_cov_enu_m2 = diag_cov;

    // Just inside boundary: scale = 0.999
    let cand_inside = [dx * 0.999, dy * 0.999, dz * 0.999];
    assert!(boundary_state.mahalanobis_innovation_gate(cand_inside, zero_geom, gate));

    // Exact boundary: scale = 1.0000000000000002
    let cand_exact = [dx, dy, dz];
    assert!(boundary_state.mahalanobis_innovation_gate(cand_exact, zero_geom, gate));

    // Just outside boundary: scale = 1.001
    let cand_outside = [dx * 1.001, dy * 1.001, dz * 1.001];
    assert!(!boundary_state.mahalanobis_innovation_gate(cand_outside, zero_geom, gate));

    // Rotated covariance test: rotate by 37 degrees around axis [1, 2, 3]
    let axis = Vector3::new(1.0, 2.0, 3.0).normalize();
    let rot = Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(axis), 0.65).into_inner();
    let rot_cov = rot * diag_cov * rot.transpose();
    let cand_inside_rot = rot * Vector3::from(cand_inside);
    let cand_outside_rot = rot * Vector3::from(cand_outside);

    let mut rot_state = state.clone();
    rot_state.pos_cov_enu_m2 = rot_cov;

    assert!(rot_state.mahalanobis_innovation_gate(
        [cand_inside_rot.x, cand_inside_rot.y, cand_inside_rot.z],
        zero_geom,
        gate,
    ));
    assert!(!rot_state.mahalanobis_innovation_gate(
        [cand_outside_rot.x, cand_outside_rot.y, cand_outside_rot.z],
        zero_geom,
        gate,
    ));

    Ok(())
}

#[test]
fn test_vertical_velocity_gating_truth_table() -> TestResult {
    let holdout_cases = [0, 5, 11, 12, 13, 50];
    let inlier_cases = [0, 20, 47, 48, 49, 200];

    for &h in &holdout_cases {
        for &i in &inlier_cases {
            let actual = can_update_vertical_velocity(h, i);
            let expected = h >= 12 || i >= 48;
            assert_eq!(
                actual, expected,
                "Failed at holdout={h}, inliers={i}: expected {expected}, got {actual}"
            );
        }
    }

    // Precise boundary transitions
    assert!(!can_update_vertical_velocity(11, 47));
    assert!(can_update_vertical_velocity(12, 47));
    assert!(can_update_vertical_velocity(11, 48));
    assert!(can_update_vertical_velocity(12, 48));

    Ok(())
}

#[test]
fn test_vertical_velocity_preserved_when_gated_and_updated_when_admitted() -> TestResult {
    let mut state = KinematicState::new([0.0, 0.0, 100.0], Matrix3::identity());
    state.vel_enu_mps = [10.0, 5.0, -3.5];
    let profile = VehicleDynamicsProfile::rotary_survey();
    let dt = 0.2; // seconds

    // Candidate pose implies:
    // delta_x = 2.0 -> vx = 10.0
    // delta_y = 1.0 -> vy = 5.0
    // delta_z = 3.0 -> vz = 15.0
    let cand_eye = [2.0, 1.0, 103.0];
    let cand_rot = Matrix3::identity();

    // Step 1: Gate is active (allow_vertical = false)
    // vx and vy update to 10.0 and 5.0. vz MUST remain locked at -3.5!
    state.update_from_pose_gated(cand_eye, cand_rot, dt, &profile, false);
    assert!((state.vel_enu_mps[0] - 10.0).abs() < 1e-6);
    assert!((state.vel_enu_mps[1] - 5.0).abs() < 1e-6);
    assert!(
        (state.vel_enu_mps[2] - (-3.5)).abs() < 1e-9,
        "Vertical velocity must be preserved when allow_vertical is false"
    );
    assert_eq!(state.eye_enu, cand_eye);

    // Step 2: Gate is released (allow_vertical = true)
    // Next candidate eye implies vz = -5.0 / 0.2 = -25.0
    let cand_eye_2 = [4.0, 2.0, 98.0];
    state.update_from_pose_gated(cand_eye_2, cand_rot, dt, &profile, true);
    assert!((state.vel_enu_mps[0] - 10.0).abs() < 1e-6);
    assert!((state.vel_enu_mps[1] - 5.0).abs() < 1e-6);
    assert!(
        (state.vel_enu_mps[2] - (-25.0)).abs() < 1e-6,
        "Vertical velocity must update when allow_vertical is true"
    );

    Ok(())
}

#[test]
fn test_vertical_velocity_multi_step_coplanar_trajectory() -> TestResult {
    let mut state = KinematicState::new([0.0, 0.0, 200.0], Matrix3::identity());
    state.vel_enu_mps = [12.0, -8.0, 1.25];
    let profile = VehicleDynamicsProfile::fixed_wing();
    let dt = 0.1;

    // Simulate 5 steps with coplanar/shallow rays (holdout=8 < 12, inliers=30 < 48)
    for step in 1..=5 {
        let allow_vertical = can_update_vertical_velocity(8, 30);
        assert!(!allow_vertical);
        let new_eye = [
            step as f64 * 1.2,
            step as f64 * -0.8,
            200.0 + step as f64 * 5.0, // candidate altitude jumps 50 m/s
        ];
        state.update_from_pose_gated(new_eye, Matrix3::identity(), dt, &profile, allow_vertical);
        // Throughout coplanar sequence, vertical velocity remains 1.25 m/s
        assert!(
            (state.vel_enu_mps[2] - 1.25).abs() < 1e-9,
            "Step {step}: vz drifted from 1.25"
        );
    }

    // Step 6: Holdout count reaches 15 >= 12 -> gate opens
    let allow_vertical = can_update_vertical_velocity(15, 30);
    assert!(allow_vertical);
    let new_eye_open = [6.0 * 1.2, 6.0 * -0.8, 200.0 + 5.0 * 5.0 + 0.5]; // dz = 0.5 over 0.1s = 5.0 m/s
    state.update_from_pose_gated(
        new_eye_open,
        Matrix3::identity(),
        dt,
        &profile,
        allow_vertical,
    );
    assert!(
        (state.vel_enu_mps[2] - 5.0).abs() < 1e-6,
        "vz did not update when gate opened"
    );

    Ok(())
}

#[test]
fn test_anchor_worker_state_transitions_and_loss_recovery() -> TestResult {
    let mut worker_state = AnchorWorkerState::default();

    // Cold start
    assert!(worker_state.prev_eye.is_none());
    assert!(worker_state.prev_rot.is_none());
    assert_eq!(worker_state.vel_enu_mps, [0.0, 0.0, 0.0]);

    // Frame 1: First anchor accepted
    worker_state.on_anchor_accepted([100.0, 200.0, 50.0], Matrix3::identity(), 0.1);
    assert_eq!(worker_state.prev_eye, Some([100.0, 200.0, 50.0]));
    assert_eq!(worker_state.prev_rot, Some(Matrix3::identity()));
    assert_eq!(worker_state.vel_enu_mps, [0.0, 0.0, 0.0]);

    // Frame 2: Second anchor accepted after 0.5s with displacement [10.0, -5.0, 1.0]
    worker_state.on_anchor_accepted([110.0, 195.0, 51.0], Matrix3::identity(), 0.5);
    assert_eq!(worker_state.prev_eye, Some([110.0, 195.0, 51.0]));
    assert!((worker_state.vel_enu_mps[0] - 20.0).abs() < 1e-6);
    assert!((worker_state.vel_enu_mps[1] - (-10.0)).abs() < 1e-6);
    assert!((worker_state.vel_enu_mps[2] - 2.0).abs() < 1e-6);

    // Frame 3: Relocation onto new consensus anchor (e.g. 500m jump)
    worker_state.on_relocation([610.0, 195.0, 51.0], 1.0);
    assert_eq!(worker_state.prev_eye, Some([610.0, 195.0, 51.0]));
    assert!((worker_state.vel_enu_mps[0] - 500.0).abs() < 1e-6);

    // Frame 4: Track loss occurs
    worker_state.on_track_loss();
    assert!(worker_state.prev_eye.is_none());
    assert!(worker_state.prev_rot.is_none());
    assert_eq!(worker_state.vel_enu_mps, [0.0, 0.0, 0.0]);

    // Frame 5: Re-acquisition following track loss
    // Must NOT carry over old velocity (500 m/s) from before loss!
    worker_state.on_anchor_accepted([900.0, 400.0, 60.0], Matrix3::identity(), 0.1);
    assert_eq!(worker_state.prev_eye, Some([900.0, 400.0, 60.0]));
    assert_eq!(
        worker_state.vel_enu_mps,
        [0.0, 0.0, 0.0],
        "Velocity must reset to zero on re-acquisition after track loss"
    );

    Ok(())
}

#[test]
fn test_anchor_worker_state_dt_extremes_and_clamping() -> TestResult {
    let mut worker_state = AnchorWorkerState::default();
    worker_state.on_anchor_accepted([0.0, 0.0, 0.0], Matrix3::identity(), 0.1);

    // Extreme dt = 0.0 s: clamped to MIN_DELTA_T_S (1e-4) -> prevents divide-by-zero
    worker_state.on_anchor_accepted([1.0, 0.0, 0.0], Matrix3::identity(), 0.0);
    assert!(worker_state.vel_enu_mps[0].is_finite());
    assert!((worker_state.vel_enu_mps[0] - 10_000.0).abs() < 1e-3);

    // Negative dt: clamped to MIN_DELTA_T_S (1e-4)
    worker_state.on_anchor_accepted([2.0, 0.0, 0.0], Matrix3::identity(), -5.0);
    assert!(worker_state.vel_enu_mps[0].is_finite());
    assert!((worker_state.vel_enu_mps[0] - 10_000.0).abs() < 1e-3);

    // Huge dt: clamped to MAX_DELTA_T_S (60.0)
    worker_state.on_anchor_accepted([62.0, 0.0, 0.0], Matrix3::identity(), 1000.0);
    assert!((worker_state.vel_enu_mps[0] - 1.0).abs() < 1e-6);

    // Relocation with dt = 0.0
    worker_state.on_relocation([63.0, 0.0, 0.0], 0.0);
    assert!(worker_state.vel_enu_mps[0].is_finite());
    assert!((worker_state.vel_enu_mps[0] - 10_000.0).abs() < 1e-3);

    Ok(())
}

#[test]
fn test_monte_carlo_property_stress() -> TestResult {
    let mut state = KinematicState::new([500.0, 500.0, 100.0], Matrix3::identity());
    let mut worker = AnchorWorkerState::default();
    let profile = VehicleDynamicsProfile::high_performance();

    // Pseudo-random linear congruential generator for deterministic stress testing
    let mut seed = 0xDEADBEEF_u64;
    let mut lcg = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (seed >> 33) as f64 / (1_u64 << 31) as f64
    };

    for i in 0..200 {
        let dt = 0.01 + lcg() * 0.5;
        let dx = (lcg() - 0.5) * 40.0;
        let dy = (lcg() - 0.5) * 40.0;
        let dz = (lcg() - 0.5) * 10.0;
        let holdout = (lcg() * 25.0) as usize;
        let inliers = (lcg() * 100.0) as usize;

        let allow_v = can_update_vertical_velocity(holdout, inliers);
        assert_eq!(allow_v, holdout >= 12 || inliers >= 48);

        let prev_vz = state.vel_enu_mps[2];
        let new_eye = [
            state.eye_enu[0] + dx,
            state.eye_enu[1] + dy,
            state.eye_enu[2] + dz,
        ];

        state.update_from_pose_gated(new_eye, Matrix3::identity(), dt, &profile, allow_v);
        if !allow_v {
            assert_eq!(
                state.vel_enu_mps[2], prev_vz,
                "Iteration {i}: vertical velocity must remain locked when allow_v is false"
            );
        }

        // Test anchor worker state transitions
        if i % 25 == 0 {
            worker.on_track_loss();
            assert_eq!(worker.vel_enu_mps, [0.0, 0.0, 0.0]);
        } else if i % 10 == 0 {
            worker.on_relocation(new_eye, dt);
            assert!(worker.vel_enu_mps[0].is_finite());
        } else {
            worker.on_anchor_accepted(new_eye, Matrix3::identity(), dt);
            assert!(worker.vel_enu_mps[0].is_finite());
        }
    }

    Ok(())
}
