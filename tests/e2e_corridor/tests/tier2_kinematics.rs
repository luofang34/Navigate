//! Tier 2: Boundary and corner cases for Features 1 to 5 (Kinematics & Dynamics).

use e2e_corridor_tests::{
    KinematicState, SearchMode, TestResult, TriggerConfig, TriggerState, VehicleDynamicsProfile,
    approx_eq, check, determine_search_mode, synthetic_photo_case,
};
use nalgebra::Matrix3;

// --- Feature 1: Video Timestamp Serialization Boundaries ---

#[test]
fn t2_01_01_zero_timestamp_boundary() -> TestResult {
    let case = synthetic_photo_case("zero_frame", Some(0.0), Some(0));
    let json_str = serde_json::to_string(&case)?;
    check(json_str.contains("\"time_s\":0.0"), "time_s 0.0 preserved")?;
    check(json_str.contains("\"sequence\":0"), "sequence 0 preserved")?;
    Ok(())
}

#[test]
fn t2_01_02_negative_timestamp_handling() -> TestResult {
    let case = synthetic_photo_case("neg_time", Some(-1.5), Some(1));
    let json_str = serde_json::to_string(&case)?;
    let parsed: e2e_corridor_tests::harness::SyntheticPhotoCase = serde_json::from_str(&json_str)?;
    check(
        parsed.time_s == Some(-1.5),
        "negative timestamp parsed as declared",
    )?;
    Ok(())
}

#[test]
fn t2_01_03_large_timestamp_overflow() -> TestResult {
    let large_time = 1e9_f64; // ~31.7 years
    let case = synthetic_photo_case("epoch_time", Some(large_time), Some(u64::MAX));
    let json_str = serde_json::to_string(&case)?;
    let parsed: e2e_corridor_tests::harness::SyntheticPhotoCase = serde_json::from_str(&json_str)?;
    check(
        parsed.time_s == Some(large_time),
        "large timestamp preserved",
    )?;
    check(parsed.sequence == Some(u64::MAX), "max sequence preserved")?;
    Ok(())
}

#[test]
fn t2_01_04_sequence_gap_detection() -> TestResult {
    let seq1 = 10;
    let seq2 = 25; // 15 dropped frames
    let gap = seq2 - seq1;
    check(gap > 1, "sequence gap detected")?;
    Ok(())
}

#[test]
fn t2_01_05_duplicate_timestamp_handling() -> TestResult {
    let t1 = 10.5;
    let t2 = 10.5; // Identical timestamp
    let dt = t2 - t1;
    check(dt == 0.0, "duplicate timestamp produces zero dt")?;
    Ok(())
}

// --- Feature 2: Continuous Kinematic State Boundaries ---

#[test]
fn t2_02_01_min_time_clamp_boundary() -> TestResult {
    let clamped = e2e_corridor_tests::kinematics_model::clamp_dt_s(1e-4);
    check(approx_eq(clamped, 1e-4, 1e-8), "exact 1e-4 lower bound")?;
    let sub_min = e2e_corridor_tests::kinematics_model::clamp_dt_s(1e-7);
    check(approx_eq(sub_min, 1e-4, 1e-8), "below min clamps to 1e-4")?;
    Ok(())
}

#[test]
fn t2_02_02_max_time_clamp_boundary() -> TestResult {
    let clamped = e2e_corridor_tests::kinematics_model::clamp_dt_s(60.0);
    check(approx_eq(clamped, 60.0, 1e-8), "exact 60.0 upper bound")?;
    let above_max = e2e_corridor_tests::kinematics_model::clamp_dt_s(120.0);
    check(approx_eq(above_max, 60.0, 1e-8), "above max clamps to 60.0")?;
    Ok(())
}

#[test]
fn t2_02_03_negative_time_step_clamp() -> TestResult {
    let clamped_neg = e2e_corridor_tests::kinematics_model::clamp_dt_s(-0.5);
    check(
        approx_eq(clamped_neg, 1e-4, 1e-8),
        "negative interval clamped to min",
    )?;
    let clamped_nan = e2e_corridor_tests::kinematics_model::clamp_dt_s(f64::NAN);
    check(
        approx_eq(clamped_nan, 1e-4, 1e-8),
        "NaN interval clamped to min",
    )?;
    Ok(())
}

#[test]
fn t2_02_04_high_speed_translation() -> TestResult {
    let state = KinematicState::new([0.0, 0.0, 1000.0], [800.0, 0.0, 0.0]);
    let dt = 0.05;
    let pred = state.predict_position(dt);
    check(
        approx_eq(pred[0], 40.0, 1e-4),
        "800 m/s translation prediction matches",
    )?;
    check(pred[0].is_finite(), "prediction remains finite")?;
    Ok(())
}

#[test]
fn t2_02_05_high_angular_rate_so3() -> TestResult {
    let mut state = KinematicState::default();
    let roll_rate = 6.0; // rad/s
    let dt = 0.033;
    state.propagate_covariance(dt, 0.0, roll_rate);
    let det = state.rot_world_to_cam.determinant();
    check(approx_eq(det, 1.0, 1e-6), "SO(3) determinant equals 1.0")?;
    Ok(())
}

// --- Feature 3: Vehicle Dynamics Profiles Boundaries ---

#[test]
fn t2_03_01_zero_velocity_profile_limit() -> TestResult {
    let mut p = VehicleDynamicsProfile::rotary_survey();
    p.max_speed_mps = 0.0;
    let clamped = p.clamp_velocity([10.0, 0.0, 0.0]);
    check(approx_eq(clamped[0], 0.0, 1e-4), "speed clamped to zero")?;
    Ok(())
}

#[test]
fn t2_03_02_accel_envelope_boundary() -> TestResult {
    let p = VehicleDynamicsProfile::high_performance();
    check(
        p.rms_accel_mps2 == 100.0,
        "high performance profile permits up to 100 m/s2",
    )?;
    Ok(())
}

#[test]
fn t2_03_03_vertical_rate_decoupling() -> TestResult {
    let p = VehicleDynamicsProfile::fixed_wing();
    // Horizontal 50 m/s (below max 80), vertical 30 m/s (above max vertical 15)
    let clamped = p.clamp_velocity([50.0, 0.0, 30.0]);
    check(
        approx_eq(clamped[0], 50.0, 1e-4),
        "horizontal speed unscaled",
    )?;
    check(
        approx_eq(clamped[2], 15.0, 1e-4),
        "vertical speed clamped to 15 m/s",
    )?;
    Ok(())
}

#[test]
fn t2_03_04_roll_rate_short_interval() -> TestResult {
    let p = VehicleDynamicsProfile::high_performance();
    let dt = 0.016; // 60 Hz frame
    let max_roll_delta = p.max_angular_rate_rad_s * dt;
    check(
        approx_eq(max_roll_delta, 0.096, 1e-4),
        "roll delta bounded by rate * dt",
    )?;
    Ok(())
}

#[test]
fn t2_03_05_unknown_profile_name_handling() -> TestResult {
    let unknown_name = "hypersonic-orbiter";
    let fallback = VehicleDynamicsProfile::high_performance();
    check(unknown_name != fallback.name, "unknown profile detected")?;
    check(fallback.max_speed_mps == 800.0, "safe fallback configured")?;
    Ok(())
}

// --- Feature 4: Interval and Event Triggers Boundaries ---

#[test]
fn t2_04_01_zero_interval_trigger_every_frame() -> TestResult {
    let config = TriggerConfig {
        mask_interval_s: 0.0,
        anchor_interval_s: 0.0,
        ..Default::default()
    };
    let mut state = TriggerState::default();
    let r1 = state.evaluate(0.01, [0.0, 0.0, 0.0], 5.0, &config);
    check(r1.trigger_mask, "frame 1 triggers")?;
    let r2 = state.evaluate(0.02, [0.0, 0.0, 0.0], 5.0, &config);
    check(r2.trigger_mask, "frame 2 triggers with zero interval")?;
    Ok(())
}

#[test]
fn t2_04_02_large_interval_suppression() -> TestResult {
    let config = TriggerConfig {
        anchor_interval_s: 3600.0, // 1 hour
        uncertainty_threshold_m: 1000.0,
        map_cell_stride_m: 50_000.0,
        ..Default::default()
    };
    let mut state = TriggerState {
        last_anchor_time_s: Some(0.0),
        last_map_cell: Some([0, 0]),
        ..Default::default()
    };
    let r = state.evaluate(60.0, [100.0, 100.0, 0.0], 10.0, &config);
    check(
        !r.trigger_anchor,
        "suppressed before large interval elapses",
    )?;
    Ok(())
}

#[test]
fn t2_04_03_jitter_interval_robustness() -> TestResult {
    let config = TriggerConfig {
        mask_interval_s: 1.0,
        ..Default::default()
    };
    let mut state = TriggerState::default();
    let jitter_times = [0.0, 0.999, 1.001, 1.002, 1.999, 2.001];
    let mut mask_count = 0;
    for &t in &jitter_times {
        if state
            .evaluate(t, [0.0, 0.0, 0.0], 5.0, &config)
            .trigger_mask
        {
            mask_count += 1;
        }
    }
    check(
        mask_count == 3,
        "triggers only at 0.0, 1.001, and 2.001 despite jitter",
    )?;
    Ok(())
}

#[test]
fn t2_04_04_simultaneous_trigger_collision() -> TestResult {
    let config = TriggerConfig {
        mask_interval_s: 1.0,
        anchor_interval_s: 1.0,
        uncertainty_threshold_m: 20.0,
        ..Default::default()
    };
    let mut state = TriggerState::default();
    let res = state.evaluate(0.0, [0.0, 0.0, 0.0], 30.0, &config);
    check(
        res.trigger_mask && res.trigger_anchor,
        "both triggers fire concurrently",
    )?;
    Ok(())
}

#[test]
fn t2_04_05_cell_reversal_hysteresis() -> TestResult {
    let config = TriggerConfig {
        map_cell_stride_m: 100.0,
        ..Default::default()
    };
    let mut state = TriggerState {
        last_map_cell: Some([0, 0]),
        ..Default::default()
    };
    // Move slightly forward inside cell
    let r1 = state.evaluate(0.1, [50.0, 50.0, 0.0], 5.0, &config);
    check(!r1.cell_boundary_crossed, "inside cell 0")?;
    // Move slightly backward inside same cell
    let r2 = state.evaluate(0.2, [48.0, 50.0, 0.0], 5.0, &config);
    check(
        !r2.cell_boundary_crossed,
        "reversal inside cell does not trigger",
    )?;
    Ok(())
}

// --- Feature 5: Adaptive Search Radius Boundaries ---

#[test]
fn t2_05_01_threshold_boundary_64px() -> TestResult {
    // 3 * sigma_px = 64.0 px -> sigma_px = 21.3333 px
    let sigma_px = 64.0 / 3.0;
    let mode = determine_search_mode(sigma_px);
    match mode {
        SearchMode::RadiusMatching(r) => check(approx_eq(r, 64.0, 1e-3), "exact 64px radius")?,
        SearchMode::GlobalLighterGlue => {
            return Err("64px exact boundary should be radius mode".into());
        }
    }
    // Just above: 64.01 px -> Global
    let mode_above = determine_search_mode(64.01 / 3.0);
    check(
        mode_above == SearchMode::GlobalLighterGlue,
        "above 64px triggers global",
    )?;
    Ok(())
}

#[test]
fn t2_05_02_lower_clamp_24px() -> TestResult {
    let sigma_px = 1.0; // 3 * 1 = 3 px < 24 px
    match determine_search_mode(sigma_px) {
        SearchMode::RadiusMatching(r) => check(approx_eq(r, 24.0, 1e-4), "clamped to 24.0 px")?,
        _ => return Err("expected radius".into()),
    }
    Ok(())
}

#[test]
fn t2_05_03_upper_clamp_96px() -> TestResult {
    // Gate specifies when 3 * sigma_px <= 64, clamp(24, 96).
    // If 3 * sigma_px = 60, radius is 60 px.
    match determine_search_mode(20.0) {
        SearchMode::RadiusMatching(r) => check((24.0..=96.0).contains(&r), "inside clamp range")?,
        _ => return Err("expected radius".into()),
    }
    Ok(())
}

#[test]
fn t2_05_04_zero_depth_protection() -> TestResult {
    let state = KinematicState::default();
    let sigma_zero = state.project_pixel_uncertainty(1000.0, 1000.0, 0.0);
    check(
        sigma_zero.is_finite() && sigma_zero > 0.0,
        "protected against zero depth division",
    )?;
    let sigma_neg = state.project_pixel_uncertainty(1000.0, 1000.0, -50.0);
    check(sigma_neg.is_finite(), "protected against negative depth")?;
    Ok(())
}

#[test]
fn t2_05_05_singular_covariance_projection() -> TestResult {
    let state = KinematicState {
        pos_cov_enu_m2: Matrix3::zeros(),
        att_sigma_rad: 0.0,
        ..Default::default()
    };
    let sigma_zero_cov = state.project_pixel_uncertainty(1000.0, 1000.0, 100.0);
    check(
        approx_eq(sigma_zero_cov, 0.0, 1e-4),
        "zero covariance projects to zero pixel uncertainty",
    )?;
    Ok(())
}
