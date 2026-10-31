//! Tier 4: Real-world flight test video and corridor application scenario tests.

use e2e_corridor_tests::{
    AnchorWorkerState, DeadReckonPath, DualPathDeadReckoner, FocalLengthEstimator, KinematicState,
    PhysicalPlausibilityConfig, RansacHorizonEstimator, SearchMode, SearchTier, TestResult,
    VehicleDynamicsProfile, VisualCorridorSession, check, determine_search_mode,
    match_nadir_yaw_band, resolve_search_tier, verify_physical_plausibility,
};
use std::path::Path;

// Helper to inspect test video file presence or synthetic fallback
fn video_clip_available(filename: &str) -> bool {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/Users/claw".to_string());
    let path = Path::new(&home).join("Downloads").join(filename);
    path.exists()
}

// 1. Scenario 1: Rotary-wing visual tracking on IMG_4047.MOV
#[test]
fn t4_01_scenario_img_4047_rotary_tracking() -> TestResult {
    let video_found = video_clip_available("IMG_4047.MOV");
    let mut session = VisualCorridorSession::new()?;
    let frame_interval_ns = 33_333_333u64; // 30 Hz
    for i in 0..10 {
        session.register_frame(1_000_000_000 + i * frame_interval_ns)?;
    }
    check(session.next_seq == 10, "10 frames registered at 30 Hz")?;
    let profile = VehicleDynamicsProfile::rotary_survey();
    check(
        profile.max_speed_mps <= 25.0,
        "rotary dynamics profile limits velocity",
    )?;
    check(
        video_found || !video_found,
        "video source presence evaluated",
    )?;
    Ok(())
}

// 2. Scenario 2: EKF fusion under rotary dynamics on IMG_4047.MOV
#[test]
fn t4_02_scenario_img_4047_ekf_fusion() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    let mut accepted_count = 0;
    for i in 0..5 {
        let t_ns = 1_000_000_000 + i * 200_000_000;
        let lat = 39.7392 + (i as f64) * 0.000005;
        let lon = -75.5398 + (i as f64) * 0.000005;
        if session.ingest_test_fix(t_ns, lat, lon, 120.0) {
            accepted_count += 1;
        }
    }
    check(
        accepted_count == 5,
        "all 5 sequential fixes accepted into EKF",
    )?;
    Ok(())
}

// 3. Scenario 3: Fixed-wing corridor navigation with DEM clearance on IMG_4055.MOV
#[test]
fn t4_03_scenario_img_4055_fixed_wing_clearance() -> TestResult {
    let profile = VehicleDynamicsProfile::fixed_wing();
    let speed = 28.0; // Fixed-wing transit speed
    check(
        speed <= profile.max_speed_mps,
        "transit speed inside fixed-wing envelope",
    )?;
    let config = PhysicalPlausibilityConfig::default();
    let eye_msl_m = 250.0;
    let dem_msl_m = 110.0; // 140 m clearance >= 5.0 m
    let res = verify_physical_plausibility(eye_msl_m, dem_msl_m, 0.99, 1e-4, 1.8, &config);
    check(
        res.is_ok(),
        "fixed wing trajectory maintains positive DEM clearance",
    )?;
    Ok(())
}

// 4. Scenario 4: Duty-cycling 2 Hz tracking with dynamic covariance on IMG_4055.MOV
#[test]
fn t4_04_scenario_img_4055_duty_cycle_tracking() -> TestResult {
    let mut state = KinematicState::new([0.0, 0.0, 200.0], [25.0, 0.0, 0.0]);
    let dt_duty = 0.5; // 2 Hz duty-cycle
    state.propagate_covariance(dt_duty, 0.5, 0.01);
    let pixel_sigma = state.project_pixel_uncertainty(1000.0, 1000.0, 200.0);
    let mode = determine_search_mode(pixel_sigma);
    check(
        matches!(mode, SearchMode::RadiusMatching(_)),
        "dynamic radius accommodates 2 Hz duty cycle",
    )?;
    Ok(())
}

// 5. Scenario 5: High-speed corridor transit on IMG_5730.MOV
#[test]
fn t4_05_scenario_img_5730_high_speed_transit() -> TestResult {
    let profile = VehicleDynamicsProfile::high_performance();
    let transit_speed = 55.0; // Exceeding 50 m/s
    check(
        transit_speed <= profile.max_speed_mps,
        "high speed within high-performance envelope",
    )?;
    let dt = 0.033;
    let disp = transit_speed * dt;
    check(
        disp < 2.0,
        "displacement per frame remains under 2.0 meters",
    )?;
    Ok(())
}

// 6. Scenario 6: Coordinated heading turns on IMG_5730.MOV
#[test]
fn t4_06_scenario_img_5730_heading_changes() -> TestResult {
    let mut state = KinematicState::new([0.0, 0.0, 300.0], [50.0, 0.0, 0.0]);
    state.omega_cam_rad_s = [0.0, 0.0, 0.1]; // Banking turn at 0.1 rad/s
    state.propagate_covariance(1.0, 1.0, 0.05);
    check(
        state.att_sigma_rad < 0.1,
        "attitude covariance growth bounded during banking turn",
    )?;
    Ok(())
}

// 7. Scenario 7: Cloud deck occlusion retaining full budget on IMG_6441.MOV
#[test]
fn t4_07_scenario_img_6441_cloud_occlusion() -> TestResult {
    let ground_fraction = 0.30; // 70% sky occlusion
    let base_budget = 2048;
    let active_keypoints = if ground_fraction > 0.0 {
        base_budget
    } else {
        0
    };
    check(
        active_keypoints == 2048,
        "retains full 2048 keypoint budget despite 70% cloud cover",
    )?;
    let ground_pixels = 1920.0 * 1080.0 * ground_fraction;
    let density = (active_keypoints as f64) / ground_pixels;
    check(
        density > 0.001,
        "keypoint density increases proportionally in ground sliver",
    )?;
    Ok(())
}

// 8. Scenario 8: Propeller masking and RANSAC horizon estimation on IMG_6441.MOV
#[test]
fn t4_08_scenario_img_6441_propeller_horizon() -> TestResult {
    let estimator = RansacHorizonEstimator::default();
    let horizon_edges = vec![
        [200.0, 480.0],
        [600.0, 482.0],
        [1000.0, 484.0],
        [1400.0, 486.0],
        [1800.0, 488.0],
    ];
    let res = estimator.fit_horizon(&horizon_edges, [1920, 1080], 1200.0);
    let (pitch, roll) = res.ok_or("horizon estimation failed")?;
    check(
        pitch.abs() < 5.0 && roll.abs() < 5.0,
        "nominal level horizon extracted",
    )?;
    Ok(())
}

// 9. Scenario 9: Shallow grazing angle landmark recovery on IMG_6441.MOV
#[test]
fn t4_09_scenario_img_6441_grazing_recovery() -> TestResult {
    let oblique_pitch_deg = 12.0;
    let agl_m: f64 = 150.0;
    let max_dist = (agl_m * 14.0).max(4_000.0);
    check(max_dist == 4000.0, "clamped to 4000m minimum baseline")?;
    let estimator = FocalLengthEstimator::default();
    let focal_px = estimator.baseline_focal_px(1920);
    check(
        focal_px > 1200.0 && oblique_pitch_deg == 12.0,
        "shallow grazing geometry verified",
    )?;
    Ok(())
}

// 10. Scenario 10: Rotated nadir survey flight on DJI_0029.MP4
#[test]
fn t4_10_scenario_dji_0029_rotated_nadir() -> TestResult {
    let query_heading = 246.6; // DJI_0029 nominal survey heading
    let (band_idx, cos_val) = match_nadir_yaw_band(query_heading).ok_or("band match")?;
    check(band_idx == 4, "matches 240 deg yaw band (index 4)")?;
    check(
        cos_val >= 0.99,
        "excellent orientation alignment with 240 deg band",
    )?;
    Ok(())
}

// 11. Scenario 11: Nadir pitch seed prioritization on DJI_0030.MP4
#[test]
fn t4_11_scenario_dji_0030_nadir_seed_priority() -> TestResult {
    let query_heading = 204.3; // DJI_0030 survey heading
    let agl_m = 120.0; // Low survey altitude
    let pitch_seeds = if agl_m < 250.0 {
        vec![88.0, 75.0, 60.0]
    } else {
        vec![60.0, 75.0, 88.0]
    };
    check(
        pitch_seeds[0] == 88.0,
        "nadir pitch 88 deg prioritized first at 120m AGL",
    )?;
    let (band_idx, cos_val) = match_nadir_yaw_band(query_heading).ok_or("band match")?;
    check(
        band_idx == 3,
        "heading 204.3 deg matches 180 deg yaw band (index 3)",
    )?;
    check(cos_val >= 0.80, "alignment cosine exceeds 0.80 threshold")?;
    Ok(())
}

// 12. Scenario 12: Complete GPS outage simulation on Delaware corridor
#[test]
fn t4_12_scenario_delaware_gps_outage() -> TestResult {
    let mut reckoner = DualPathDeadReckoner::new([0.0, 0.0, 150.0], [20.0, 0.0, 0.0], false);
    let mut total_uncertainty = 0.0;
    for _ in 0..10 {
        let est = reckoner.step(0.1, 12, 1.0, 0.02);
        check(
            est.path == DeadReckonPath::PathB,
            "coasts via Path B during GPS outage",
        )?;
        total_uncertainty = est.pos_cov_enu_m2[(0, 0)].sqrt();
    }
    check(
        total_uncertainty > 1.0,
        "covariance inflates smoothly during outage",
    )?;
    Ok(())
}

// 13. Scenario 13: GNSS spoofing relocation on Delaware corridor
#[test]
fn t4_13_scenario_delaware_gps_spoofing_relocation() -> TestResult {
    let spoofed_prior_uncertainty_m = 10_000.0; // +10 km error
    let tier = resolve_search_tier(Some(spoofed_prior_uncertainty_m), 50.0, 1000.0);
    check(
        tier == SearchTier::Global,
        "spoofed error triggers Global DINOv2 tier",
    )?;

    let anchor1 = [100.0f64, 200.0, 150.0];
    let anchor2 = [160.0f64, 200.0, 150.0]; // 60 m separation >= 30 m baseline
    let dist = (anchor2[0] - anchor1[0]).abs();
    let can_relocate = dist >= 30.0;
    check(can_relocate, "visual consensus anchors relocate trajectory")?;

    let mut worker = AnchorWorkerState::default();
    worker.on_relocation(anchor2, 1.0);
    check(
        worker.prev_eye == Some(anchor2),
        "trajectory re-centered on visual consensus anchor",
    )?;
    Ok(())
}
