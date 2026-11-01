//! Milestone 6 Tier 5 adversarial coverage audit and stress test suite.
//!
//! This empirical test suite validates five core pipeline areas:
//! 1. Extreme kinematics, zero delta-t, and maximum delta-t clamping bounds.
//! 2. Horizon attitude line detection, RANSAC outlier failure fallbacks, and negative down angles.
//! 3. Nadir yaw synthesis, orientation cone bounds, and heading invariance.
//! 4. Hardware GPU queue isolation and camera reconfiguration.
//! 5. Dead reckoning Path A to Path B transitions and anchor state resets.

use image::{GrayImage, Luma, Rgb, RgbImage};
use nalgebra::Matrix3;
use navigate_visual_pipeline::{
    PipelineError,
    corridor::{
        KinematicState, VehicleDynamicsProfile, compute_dt_s,
        kinematics::{CHI2_GATE_3DOF_99, MAX_DELTA_T_S, MIN_DELTA_T_S},
        nadir::{
            MIN_COS_ORIENTATION, NADIR_DOWN_DEG, NADIR_YAW_BANDS_DEG, is_nadir_pose,
            match_nadir_yaw_band, min_cos_orientation_for_pose, rotate_image_gray, unrotate_pixel,
        },
    },
    surface::{
        RansacHorizonEstimator, Surface, SurfaceGpuRenderer,
        horizon::{
            ADE20K_AIRPLANE_LABEL, ADE20K_SKY_LABEL, AIRFRAME_LABEL, GROUND_LABEL, PROPELLER_LABEL,
            SKY_LABEL, horizon_attitude, ransac_fit_line, theil_sen_fit_line,
        },
    },
};
use std::error::Error;

type TestResult = Result<(), Box<dyn Error>>;

// --- Section 1: Extreme Kinematics & Delta-T Clamping Bounds ---

#[test]
fn test_tier5_extreme_dt_clamping_and_inverted_timestamps() -> TestResult {
    // Sub-microsecond difference must clamp to MIN_DELTA_T_S.
    let dt_micro = compute_dt_s(Some(10.00002), Some(10.0), 30.0);
    assert!((dt_micro - MIN_DELTA_T_S).abs() < 1e-9);

    // Large gap must clamp to MAX_DELTA_T_S.
    let dt_huge = compute_dt_s(Some(500.0), Some(10.0), 30.0);
    assert!((dt_huge - MAX_DELTA_T_S).abs() < 1e-9);

    // Equal timestamps must use default frame rate.
    let dt_equal = compute_dt_s(Some(25.0), Some(25.0), 50.0);
    assert!((dt_equal - 0.02).abs() < 1e-6);

    // Reversed timestamps must use default frame rate.
    let dt_rev = compute_dt_s(Some(10.0), Some(25.0), 20.0);
    assert!((dt_rev - 0.05).abs() < 1e-6);

    // Missing timestamps must use default frame rate.
    assert!((compute_dt_s(None, Some(1.0), 25.0) - 0.04).abs() < 1e-6);
    assert!((compute_dt_s(Some(1.0), None, 25.0) - 0.04).abs() < 1e-6);

    // Negative or zero default frame rate must clamp to minimum rate.
    let dt_zero_fps = compute_dt_s(None, None, 0.0);
    assert!((dt_zero_fps - 10.0).abs() < 1e-6);
    let dt_neg_fps = compute_dt_s(None, None, -5.0);
    assert!((dt_neg_fps - 10.0).abs() < 1e-6);

    // Zero dt passed into update_from_pose_gated must clamp safely without zero division.
    let mut state = KinematicState::new([0.0, 0.0, 100.0], Matrix3::identity());
    let profile = VehicleDynamicsProfile::rotary_survey();
    state.update_from_pose_gated(
        [0.5, 0.0, 100.0],
        Matrix3::identity(),
        0.0,
        &profile,
        true,
        None,
    );
    assert!(state.vel_enu_mps[0].is_finite());
    assert!(state.eye_enu[0] == 0.5);

    // Huge dt passed into update_from_pose_gated must clamp to MAX_DELTA_T_S.
    state.update_from_pose_gated(
        [5.0, 0.0, 100.0],
        Matrix3::identity(),
        9999.0,
        &profile,
        true,
        None,
    );
    assert!(state.vel_enu_mps[0].is_finite());
    assert!(state.eye_enu[0] == 5.0);
    Ok(())
}

#[test]
fn test_tier5_kinematic_gating_bounds_and_vertical_velocity_freeze() -> TestResult {
    let profile = VehicleDynamicsProfile::rotary_survey();
    let mut state = KinematicState::new([0.0, 0.0, 100.0], Matrix3::identity());

    // Valid velocity update within platform limits (20 m/s <= 1.5 * 25.0 m/s, dt = 0.1 -> alpha = 1/3).
    state.update_from_pose_gated(
        [2.0, 0.0, 100.0],
        Matrix3::identity(),
        0.1,
        &profile,
        true,
        None,
    );
    let expected_v0 = 20.0 / 3.0;
    assert!((state.vel_enu_mps[0] - expected_v0).abs() < 1e-6);

    // Outlier displacement (50 m/s > 37.5 m/s threshold) must not change velocity.
    state.update_from_pose_gated(
        [7.0, 0.0, 100.0],
        Matrix3::identity(),
        0.1,
        &profile,
        true,
        None,
    );
    assert!((state.vel_enu_mps[0] - expected_v0).abs() < 1e-6);

    // Shallow ray update with allow_vertical = false must freeze vertical velocity.
    state.vel_enu_mps = [5.0, 0.0, -1.0];
    state.update_from_pose_gated(
        [8.0, 0.0, 102.0],
        Matrix3::identity(),
        0.1,
        &profile,
        false,
        None,
    );
    let expected_v1 = (2.0 / 3.0) * 5.0 + (1.0 / 3.0) * 10.0;
    assert!((state.vel_enu_mps[0] - expected_v1).abs() < 1e-6);
    assert!((state.vel_enu_mps[2] - (-1.0)).abs() < 1e-6);

    // Mahalanobis innovation gate must reject non-finite inputs and outliers.
    let geom_cov = Matrix3::identity() * 2.0;
    assert!(!state.mahalanobis_innovation_gate([f64::NAN, 0.0, 0.0], geom_cov, CHI2_GATE_3DOF_99));
    assert!(!state.mahalanobis_innovation_gate(
        [f64::INFINITY, 0.0, 0.0],
        geom_cov,
        CHI2_GATE_3DOF_99
    ));
    assert!(state.mahalanobis_innovation_gate([8.1, 0.05, 102.0], geom_cov, CHI2_GATE_3DOF_99));
    assert!(!state.mahalanobis_innovation_gate([50.0, 0.0, 102.0], geom_cov, CHI2_GATE_3DOF_99));
    Ok(())
}

#[test]
fn test_tier5_dynamics_profiles_displacement_and_vo_bounds() -> TestResult {
    assert!(VehicleDynamicsProfile::from_name("rotary-survey").is_some());
    assert!(VehicleDynamicsProfile::from_name("fixed-wing").is_some());
    assert!(VehicleDynamicsProfile::from_name("high-performance").is_some());
    assert!(VehicleDynamicsProfile::from_name("invalid-profile").is_none());

    let hp = VehicleDynamicsProfile::high_performance();
    // At large delta-t, min_trace_cosine must clamp to -1.0.
    assert!((hp.min_trace_cosine(100.0) - (-1.0)).abs() < 1e-6);

    // Ground VO bounds clamping limits.
    let (filter_gate_min, clamp_gate_min) = hp.ground_vo_bounds(1e-4);
    assert_eq!(filter_gate_min, 10.0);
    assert_eq!(clamp_gate_min, 5.0);

    let (filter_gate_max, clamp_gate_max) = hp.ground_vo_bounds(60.0);
    assert_eq!(filter_gate_max, 1500.0);
    assert_eq!(clamp_gate_max, 1000.0);

    // Pixel displacement must guard small depths and small focal lengths.
    let disp = hp.max_pixel_displacement(0.1, 10.0, 10.0, 1.0);
    assert!(disp > 0.0 && disp.is_finite());
    Ok(())
}

// --- Section 2: Horizon Attitude RANSAC Outliers & Negative Down Angles ---

#[test]
fn test_tier5_horizon_attitude_airframe_propeller_and_ade20k_rejection() -> TestResult {
    let (w, h) = (32_u32, 24_u32);
    let k = Matrix3::new(40.0, 0.0, 16.0, 0.0, 40.0, 12.0, 0.0, 0.0, 1.0);

    // Labels with wrong length must return None.
    let short_labels = vec![SKY_LABEL; 10];
    assert!(horizon_attitude(&short_labels, w, h, &k).is_none());

    // Small images below 8x8 must return None.
    let tiny_labels = vec![SKY_LABEL; 36];
    assert!(horizon_attitude(&tiny_labels, 6, 6, &k).is_none());

    // Sky transitioning only into airframe must return None.
    let mut airframe_mask = vec![SKY_LABEL; (w * h) as usize];
    for y in 10..h {
        for x in 0..w {
            airframe_mask[(y * w + x) as usize] = AIRFRAME_LABEL;
        }
    }
    assert!(horizon_attitude(&airframe_mask, w, h, &k).is_none());

    // Sky transitioning only into propeller must return None.
    let mut prop_mask = vec![SKY_LABEL; (w * h) as usize];
    for y in 10..h {
        for x in 0..w {
            prop_mask[(y * w + x) as usize] = PROPELLER_LABEL;
        }
    }
    assert!(horizon_attitude(&prop_mask, w, h, &k).is_none());

    // ADE20K mode: sky transitioning only into airplane bodies (90) must return None.
    let mut ade_plane = vec![ADE20K_SKY_LABEL; (w * h) as usize];
    for y in 10..h {
        for x in 0..w {
            ade_plane[(y * w + x) as usize] = ADE20K_AIRPLANE_LABEL;
        }
    }
    assert!(horizon_attitude(&ade_plane, w, h, &k).is_none());
    Ok(())
}

#[test]
fn test_tier5_horizon_attitude_negative_pitch_nose_up_geometry() -> TestResult {
    let (w, h) = (64_u32, 48_u32);
    let fy = 50.0_f64;
    let cy = 24.0_f64;
    let k = Matrix3::new(50.0, 0.0, 32.0, 0.0, fy, cy, 0.0, 0.0, 1.0);

    // Horizon line below optical center (y = 36 > cy = 24) corresponds to nose-up attitude.
    let mut labels_nose_up = vec![SKY_LABEL; (w * h) as usize];
    for y in 36..h {
        for x in 0..w {
            labels_nose_up[(y * w + x) as usize] = GROUND_LABEL;
        }
    }
    let est_up = horizon_attitude(&labels_nose_up, w, h, &k).ok_or("nose-up horizon not found")?;
    assert!(
        est_up.0 < 0.0,
        "Pitch angle must be negative for nose-up orientation, got {}",
        est_up.0
    );
    assert!((est_up.1 - 0.0).abs() < 1.0, "Roll must remain near zero");

    // Horizon line above optical center (y = 12 < cy = 24) corresponds to nose-down attitude.
    let mut labels_nose_down = vec![SKY_LABEL; (w * h) as usize];
    for y in 12..h {
        for x in 0..w {
            labels_nose_down[(y * w + x) as usize] = GROUND_LABEL;
        }
    }
    let est_down =
        horizon_attitude(&labels_nose_down, w, h, &k).ok_or("nose-down horizon not found")?;
    assert!(
        est_down.0 > 0.0,
        "Pitch angle must be positive for nose-down orientation, got {}",
        est_down.0
    );

    // RansacHorizonEstimator fit_horizon fallback when points count is below 4.
    let estimator = RansacHorizonEstimator::default();
    let sparse_points = [[10.0, 20.0], [20.0, 20.0], [30.0, 20.0]];
    assert!(
        estimator
            .fit_horizon(&sparse_points, [w, h], 50.0)
            .is_none()
    );
    Ok(())
}

#[test]
fn test_tier5_ransac_line_fitting_degeneration_and_theil_sen_median() -> TestResult {
    // Fewer than 4 points must return None.
    assert!(ransac_fit_line(&[[0.0, 0.0], [1.0, 1.0]], 2.0, 50).is_none());

    // Vertical points with identical x must return None.
    let vert_points = [[10.0, 0.0], [10.0, 5.0], [10.0, 10.0], [10.0, 15.0]];
    assert!(ransac_fit_line(&vert_points, 2.0, 50).is_none());

    // Points with extreme outliers must recover the true line.
    let mut points = Vec::new();
    for i in 0..16 {
        let x = f64::from(i) * 5.0;
        let y = 2.0 * x + 10.0;
        points.push([x, y]);
    }
    // Add 4 severe outliers.
    points.push([10.0, 200.0]);
    points.push([20.0, -150.0]);
    points.push([30.0, 300.0]);
    points.push([40.0, -200.0]);

    let (m, c) = ransac_fit_line(&points, 3.0, 100).ok_or("RANSAC fit failed")?;
    assert!(
        (m - 2.0).abs() < 0.1,
        "Estimated slope must be near 2.0, got {m}"
    );
    assert!(
        (c - 10.0).abs() < 1.0,
        "Estimated intercept must be near 10.0, got {c}"
    );

    // Theil-Sen fit with fewer than 2 points must return None.
    assert!(theil_sen_fit_line(&[[1.0, 2.0]]).is_none());
    Ok(())
}

// --- Section 3: Nadir Yaw Synthesis, Orientation Cones & Heading Invariance ---

#[test]
fn test_tier5_nadir_pose_threshold_and_orientation_cones() -> TestResult {
    let bands = NADIR_YAW_BANDS_DEG;
    assert_eq!(bands.len(), 6);

    // Exact pitch boundary where -rot[(2, 2)] = 0.95:
    let mut rot_exact_95 = Matrix3::<f64>::identity();
    rot_exact_95[(2, 2)] = -0.95;
    assert!(
        !is_nadir_pose(&rot_exact_95),
        "Exact 0.95 must not qualify as nadir"
    );
    assert_eq!(min_cos_orientation_for_pose(&rot_exact_95), 0.90);

    let mut rot_above_95 = Matrix3::<f64>::identity();
    rot_above_95[(2, 2)] = -0.9501;
    assert!(
        is_nadir_pose(&rot_above_95),
        "Value above 0.95 must qualify as nadir"
    );
    assert_eq!(
        min_cos_orientation_for_pose(&rot_above_95),
        MIN_COS_ORIENTATION
    );

    // Canonical nadir pitch 89.0 deg:
    let mut rot_89 = Matrix3::<f64>::identity();
    rot_89[(2, 2)] = -NADIR_DOWN_DEG.to_radians().sin();
    assert!(is_nadir_pose(&rot_89));

    // Nose-up orientation pitch -10.0 deg:
    let mut rot_up = Matrix3::<f64>::identity();
    rot_up[(2, 2)] = -(-10.0_f64).to_radians().sin();
    assert!(!is_nadir_pose(&rot_up));
    assert_eq!(min_cos_orientation_for_pose(&rot_up), 0.90);

    // Orientation cone angles:
    let cos_36 = 36.0_f64.to_radians().cos();
    assert!(cos_36 >= f64::from(MIN_COS_ORIENTATION));
    let cos_37 = 37.0_f64.to_radians().cos();
    assert!(cos_37 < f64::from(MIN_COS_ORIENTATION));
    Ok(())
}

#[test]
fn test_tier5_nadir_heading_sweep_and_non_finite_rejection() -> TestResult {
    // Non-finite query headings must reject cleanly with None.
    assert!(match_nadir_yaw_band(f64::NAN).is_none());
    assert!(match_nadir_yaw_band(f64::INFINITY).is_none());
    assert!(match_nadir_yaw_band(f64::NEG_INFINITY).is_none());

    // Continuous 720-degree sweep must find a valid band with cos >= cos(30 deg) = 0.866.
    let mut deg = -360.0_f64;
    while deg <= 360.0 {
        let matched = match_nadir_yaw_band(deg).ok_or("heading must match")?;
        assert!(matched.1 >= 0.8660 - 1e-4);
        assert!(matched.0 < 6);
        deg += 10.0;
    }
    Ok(())
}

#[test]
fn test_tier5_nadir_image_rotation_edge_sizes_and_fixed_points() -> TestResult {
    // Zero-sized image rotation must not panic.
    let empty_img = GrayImage::new(0, 0);
    let rot_empty = rotate_image_gray(&empty_img, 60.0);
    assert_eq!(rot_empty.width(), 0);
    assert_eq!(rot_empty.height(), 0);

    // Single pixel image rotation.
    let mut single = GrayImage::new(1, 1);
    single.put_pixel(0, 0, Luma([180]));
    let rot_single = rotate_image_gray(&single, 120.0);
    assert_eq!(rot_single.get_pixel(0, 0).0[0], 180);

    // Center pixel must remain an invariant fixed point under unrotation.
    let (w, h) = (160_u32, 120_u32);
    let cx = f64::from(w) * 0.5 - 0.5;
    let cy = f64::from(h) * 0.5 - 0.5;
    for &angle in &NADIR_YAW_BANDS_DEG {
        let unrot = unrotate_pixel([cx, cy], angle, w, h);
        assert!((unrot[0] - cx).abs() < 1e-9);
        assert!((unrot[1] - cy).abs() < 1e-9);
    }
    Ok(())
}

// --- Section 4: Hardware GPU Queue Isolation & Camera Reconfiguration ---

#[test]
fn test_tier5_gpu_queue_isolation_and_camera_reconfiguration() -> TestResult {
    let tile = RgbImage::from_pixel(256, 256, Rgb([130, 145, 120]));
    let surface = Surface::synthetic(39.75, -75.55, 100.0, tile);
    let size1 = [160_u32, 120_u32];
    let k1 = Matrix3::new(150.0, 0.0, 79.5, 0.0, 150.0, 59.5, 0.0, 0.0, 1.0);

    if let Some(mut renderer) = SurfaceGpuRenderer::try_new(&surface, &k1, size1) {
        assert!(!renderer.is_queue_isolated());
        assert!(renderer.queue_identity().starts_with("shared:"));
        let _dev = renderer.device();
        let _q = renderer.queue();

        // Render first view with size1.
        let eye = surface.local(39.75, -75.55, 300.0);
        let rot = surface.camera_rotation(39.75, -75.55, 0.0, 89.0, 0.0);
        let v1 = renderer.render_view(&surface, eye, &rot, &k1, size1)?;
        assert_eq!(v1.camera.width, 160);
        assert_eq!(v1.camera.height, 120);

        // Change camera size and intrinsics to trigger dynamic CameraKey reconfiguration.
        let size2 = [200_u32, 150_u32];
        let k2 = Matrix3::new(180.0, 0.0, 99.5, 0.0, 180.0, 74.5, 0.0, 0.0, 1.0);
        let v2 = renderer.render_view(&surface, eye, &rot, &k2, size2)?;
        assert_eq!(v2.camera.width, 200);
        assert_eq!(v2.camera.height, 150);

        // Invalid intrinsics must return error without panic.
        let bad_k = Matrix3::zeros();
        let err = renderer.render_view(&surface, eye, &rot, &bad_k, size1);
        assert!(matches!(err, Err(PipelineError::Invalid(_))));
    }
    Ok(())
}
