//! Tier 1: Feature coverage tests for Features 1 to 5 (Kinematics & Dynamics).

use e2e_corridor_tests::{
    KinematicState, SearchMode, TestResult, TriggerConfig, TriggerState, VehicleDynamicsProfile,
    approx_eq, check, determine_search_mode, synthetic_photo_case,
};
use nalgebra::Matrix3;

// --- Feature 1: Video Timestamp Serialization ---

#[test]
fn t1_01_01_serialize_nominal_fields() -> TestResult {
    let case = synthetic_photo_case("frame_001", Some(12.345678), Some(42));
    let json_str = serde_json::to_string(&case)?;
    check(
        json_str.contains("\"sequence\":42"),
        "must contain sequence",
    )?;
    check(
        json_str.contains("\"time_s\":12.345678"),
        "must contain time_s",
    )?;
    Ok(())
}

#[test]
fn t1_01_02_deserialize_nominal_json() -> TestResult {
    let raw = r#"{"id":"f2","size":[1920,1080],"sequence":105,"time_s":3.5,"valid_fraction":0.85}"#;
    let parsed: e2e_corridor_tests::harness::SyntheticPhotoCase = serde_json::from_str(raw)?;
    check(parsed.sequence == Some(105), "sequence must match 105")?;
    check(parsed.time_s == Some(3.5), "time_s must match 3.5")?;
    check(
        approx_eq(parsed.valid_fraction, 0.85, 1e-4),
        "valid_fraction",
    )?;
    Ok(())
}

#[test]
fn t1_01_03_deserialize_legacy_json_backward_compat() -> TestResult {
    let legacy = r#"{"id":"legacy_01","size":[1280,720]}"#;
    let parsed: e2e_corridor_tests::harness::SyntheticPhotoCase = serde_json::from_str(legacy)?;
    check(parsed.sequence.is_none(), "sequence defaults to None")?;
    check(parsed.time_s.is_none(), "time_s defaults to None")?;
    check(
        approx_eq(parsed.valid_fraction, 1.0, 1e-4),
        "valid_fraction defaults to 1.0",
    )?;
    Ok(())
}

#[test]
fn t1_01_04_timestamp_subsecond_precision() -> TestResult {
    let stamp = 1713369600.1234567;
    let case = synthetic_photo_case("subsecond", Some(stamp), Some(1));
    let serialized = serde_json::to_string(&case)?;
    let parsed: e2e_corridor_tests::harness::SyntheticPhotoCase =
        serde_json::from_str(&serialized)?;
    let actual = parsed.time_s.ok_or("time_s missing")?;
    check(approx_eq(actual, stamp, 1e-6), "precision preserved")?;
    Ok(())
}

#[test]
fn t1_01_05_sequence_monotonicity() -> TestResult {
    let cases = vec![
        synthetic_photo_case("f1", Some(10.0), Some(1)),
        synthetic_photo_case("f2", Some(10.033), Some(2)),
        synthetic_photo_case("f3", Some(10.066), Some(3)),
    ];
    let mut prev_seq = 0;
    for c in &cases {
        let seq = c.sequence.ok_or("seq missing")?;
        check(seq > prev_seq, "sequence must monotonically increase")?;
        prev_seq = seq;
    }
    Ok(())
}

// --- Feature 2: Continuous Kinematic State ---

#[test]
fn t1_02_01_predict_position_at_30hz() -> TestResult {
    let state = KinematicState::new([100.0, 200.0, 50.0], [30.0, -10.0, 2.0]);
    let dt_s = 1.0 / 30.0;
    let pred = state.predict_position(dt_s);
    check(approx_eq(pred[0], 101.0, 1e-3), "x displacement")?;
    check(approx_eq(pred[1], 199.6667, 1e-3), "y displacement")?;
    check(approx_eq(pred[2], 50.0667, 1e-3), "z displacement")?;
    Ok(())
}

#[test]
fn t1_02_02_predict_position_duty_cycled_2hz() -> TestResult {
    let state = KinematicState::new([0.0, 0.0, 100.0], [50.0, 0.0, 0.0]);
    let dt_s = 0.5; // 2 Hz duty cycle
    let pred = state.predict_position(dt_s);
    check(approx_eq(pred[0], 25.0, 1e-4), "x moves 25m in 0.5s")?;
    check(approx_eq(pred[1], 0.0, 1e-4), "y unchanged")?;
    check(approx_eq(pred[2], 100.0, 1e-4), "z unchanged")?;
    Ok(())
}

#[test]
fn t1_02_03_propagate_covariance_translation() -> TestResult {
    let mut state = KinematicState::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    let initial_var = state.pos_cov_enu_m2[(0, 0)];
    let dt = 1.0;
    let a_rms = 4.0;
    state.propagate_covariance(dt, a_rms, 0.0);
    // q_pos = 0.5 * 4.0 * 1.0 = 2.0; pos_noise = 2.0^2 = 4.0
    let final_var = state.pos_cov_enu_m2[(0, 0)];
    check(
        approx_eq(final_var, initial_var + 4.0, 1e-4),
        "covariance grew by 4.0",
    )?;
    Ok(())
}

#[test]
fn t1_02_04_propagate_covariance_attitude() -> TestResult {
    let mut state = KinematicState::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    let initial_sigma = state.att_sigma_rad;
    let dt = 0.5;
    let omega_rms = 0.02; // rad/s
    state.propagate_covariance(dt, 0.0, omega_rms);
    let expected = initial_sigma + omega_rms * dt;
    check(
        approx_eq(state.att_sigma_rad, expected, 1e-4),
        "attitude sigma grew linearly",
    )?;
    Ok(())
}

#[test]
fn t1_02_05_time_interval_clamping() -> TestResult {
    check(
        approx_eq(
            e2e_corridor_tests::kinematics_model::clamp_dt_s(1e-5),
            1e-4,
            1e-6,
        ),
        "clamp min",
    )?;
    check(
        approx_eq(
            e2e_corridor_tests::kinematics_model::clamp_dt_s(100.0),
            60.0,
            1e-6,
        ),
        "clamp max",
    )?;
    check(
        approx_eq(
            e2e_corridor_tests::kinematics_model::clamp_dt_s(0.033),
            0.033,
            1e-6,
        ),
        "nominal",
    )?;
    Ok(())
}

// --- Feature 3: Vehicle Dynamics Profiles ---

#[test]
fn t1_03_01_profile_rotary_survey_limits() -> TestResult {
    let p = VehicleDynamicsProfile::rotary_survey();
    check(p.max_speed_mps == 25.0, "rotary max speed 25 m/s")?;
    check(
        p.max_vertical_speed_mps == 8.0,
        "rotary vertical speed 8 m/s",
    )?;
    check(p.rms_accel_mps2 == 5.0, "rotary accel 5 m/s2")?;
    check(p.max_angular_rate_rad_s == 1.5, "rotary yaw rate 1.5 rad/s")?;
    Ok(())
}

#[test]
fn t1_03_02_profile_fixed_wing_limits() -> TestResult {
    let p = VehicleDynamicsProfile::fixed_wing();
    check(p.max_speed_mps == 80.0, "fixed wing speed 80 m/s")?;
    check(p.rms_accel_mps2 == 12.0, "fixed wing accel 12 m/s2")?;
    Ok(())
}

#[test]
fn t1_03_03_profile_high_performance_limits() -> TestResult {
    let p = VehicleDynamicsProfile::high_performance();
    check(p.max_speed_mps == 800.0, "high performance speed 800 m/s")?;
    check(p.rms_accel_mps2 == 100.0, "high performance accel 100 m/s2")?;
    check(
        p.max_angular_rate_rad_s == 6.0,
        "high performance roll 6.0 rad/s",
    )?;
    Ok(())
}

#[test]
fn t1_03_04_profile_custom_overrides() -> TestResult {
    let mut p = VehicleDynamicsProfile::rotary_survey();
    p.max_speed_mps = 35.0;
    p.rms_accel_mps2 = 8.0;
    check(p.max_speed_mps == 35.0, "custom speed override")?;
    check(p.rms_accel_mps2 == 8.0, "custom accel override")?;
    Ok(())
}

#[test]
fn t1_03_05_expected_displacement_bound() -> TestResult {
    let p = VehicleDynamicsProfile::rotary_survey();
    let dt = 0.1;
    // max_disp = 25.0 * 0.1 + 0.5 * 5.0 * 0.01 = 2.5 + 0.025 = 2.525 m
    let disp = p.max_expected_displacement(dt);
    check(
        approx_eq(disp, 2.525, 1e-4),
        "expected displacement formula",
    )?;
    Ok(())
}

// --- Feature 4: Interval and Event Triggers ---

#[test]
fn t1_04_01_mask_interval_seconds() -> TestResult {
    let config = TriggerConfig {
        mask_interval_s: 2.0,
        anchor_interval_s: 10.0,
        ..Default::default()
    };
    let mut state = TriggerState::default();
    let r1 = state.evaluate(0.0, [0.0, 0.0, 0.0], 5.0, &config);
    check(r1.trigger_mask, "first frame triggers mask")?;

    let r2 = state.evaluate(1.0, [0.0, 0.0, 0.0], 5.0, &config);
    check(!r2.trigger_mask, "1.0s does not trigger 2.0s mask")?;

    let r3 = state.evaluate(2.1, [0.0, 0.0, 0.0], 5.0, &config);
    check(r3.trigger_mask, "2.1s triggers mask")?;
    Ok(())
}

#[test]
fn t1_04_02_anchor_interval_seconds() -> TestResult {
    let config = TriggerConfig {
        mask_interval_s: 10.0,
        anchor_interval_s: 3.0,
        uncertainty_threshold_m: 100.0,
        map_cell_stride_m: 1000.0,
    };
    let mut state = TriggerState::default();
    let r1 = state.evaluate(0.0, [0.0, 0.0, 0.0], 10.0, &config);
    check(r1.trigger_anchor, "first frame triggers anchor")?;

    let r2 = state.evaluate(2.0, [0.0, 0.0, 0.0], 10.0, &config);
    check(!r2.trigger_anchor, "2.0s < 3.0s does not trigger")?;

    let r3 = state.evaluate(3.1, [0.0, 0.0, 0.0], 10.0, &config);
    check(r3.trigger_anchor, "3.1s >= 3.0s triggers anchor")?;
    Ok(())
}

#[test]
fn t1_04_03_horizontal_uncertainty_trigger() -> TestResult {
    let config = TriggerConfig {
        mask_interval_s: 10.0,
        anchor_interval_s: 10.0,
        uncertainty_threshold_m: 30.0,
        map_cell_stride_m: 1000.0,
    };
    let mut state = TriggerState {
        last_mask_time_s: Some(0.0),
        last_anchor_time_s: Some(0.0),
        last_map_cell: Some([0, 0]),
    };
    let r_low = state.evaluate(1.0, [10.0, 10.0, 0.0], 15.0, &config);
    check(!r_low.trigger_anchor, "15m uncertainty does not trigger")?;

    let r_high = state.evaluate(1.5, [10.0, 10.0, 0.0], 35.0, &config);
    check(r_high.trigger_anchor, "35m uncertainty triggers anchor")?;
    Ok(())
}

#[test]
fn t1_04_04_map_cell_boundary_crossing() -> TestResult {
    let config = TriggerConfig {
        map_cell_stride_m: 500.0,
        ..Default::default()
    };
    let mut state = TriggerState {
        last_mask_time_s: Some(0.0),
        last_anchor_time_s: Some(0.0),
        last_map_cell: Some([0, 0]),
    };
    let r1 = state.evaluate(0.5, [250.0, 100.0, 0.0], 5.0, &config);
    check(!r1.cell_boundary_crossed, "inside cell 0")?;

    let r2 = state.evaluate(1.0, [550.0, 100.0, 0.0], 5.0, &config);
    check(r2.cell_boundary_crossed, "crossed into cell 1")?;
    check(r2.trigger_anchor, "cell crossing triggers anchor")?;
    Ok(())
}

#[test]
fn t1_04_05_variable_rate_temporal_stability() -> TestResult {
    let config = TriggerConfig {
        anchor_interval_s: 1.0,
        ..Default::default()
    };
    let mut state = TriggerState::default();
    let timestamps = [0.0, 0.033, 0.066, 0.5, 0.95, 1.02, 1.05];
    let mut anchor_triggers = 0;
    for &t in &timestamps {
        let res = state.evaluate(t, [0.0, 0.0, 0.0], 5.0, &config);
        if res.trigger_anchor {
            anchor_triggers += 1;
        }
    }
    check(
        anchor_triggers == 2,
        "fires at 0.0 and 1.02, not intermediate frames",
    )?;
    Ok(())
}

// --- Feature 5: Adaptive Search Radius ---

#[test]
fn t1_05_01_project_pixel_uncertainty_nominal() -> TestResult {
    let mut state = KinematicState::new([0.0, 0.0, 200.0], [0.0, 0.0, 0.0]);
    state.pos_cov_enu_m2 = Matrix3::from_diagonal_element(4.0); // sigma_pos = sqrt(8) ~ 2.828
    state.att_sigma_rad = 0.005; // ~0.28 deg
    let fx = 1000.0;
    let fy = 1000.0;
    let depth = 200.0;
    let sigma_px = state.project_pixel_uncertainty(fx, fy, depth);
    check(
        sigma_px > 10.0 && sigma_px < 30.0,
        "projected uncertainty in valid range",
    )?;
    Ok(())
}

#[test]
fn t1_05_02_select_radius_mode_low_uncertainty() -> TestResult {
    let mode = determine_search_mode(12.0); // 3 * 12 = 36 px <= 64 px
    match mode {
        SearchMode::RadiusMatching(r) => check(approx_eq(r, 36.0, 1e-4), "matches 36 px radius")?,
        SearchMode::GlobalLighterGlue => return Err("should be radius mode".into()),
    }
    Ok(())
}

#[test]
fn t1_05_03_clamp_search_radius_bounds() -> TestResult {
    // 3 * 5 = 15 px -> clamped to 24.0
    match determine_search_mode(5.0) {
        SearchMode::RadiusMatching(r) => check(approx_eq(r, 24.0, 1e-4), "clamped to min 24px")?,
        _ => return Err("expected radius".into()),
    }
    // 3 * 20 = 60 px -> unclamped 60.0
    match determine_search_mode(20.0) {
        SearchMode::RadiusMatching(r) => check(approx_eq(r, 60.0, 1e-4), "nominal 60px")?,
        _ => return Err("expected radius".into()),
    }
    Ok(())
}

#[test]
fn t1_05_04_select_global_mode_high_uncertainty() -> TestResult {
    // 3 * 25 = 75 px > 64 px
    let mode = determine_search_mode(25.0);
    check(
        mode == SearchMode::GlobalLighterGlue,
        "global lighterglue for >64px",
    )?;
    Ok(())
}

#[test]
fn t1_05_05_depth_scaling_relationship() -> TestResult {
    let state = KinematicState::new([0.0, 0.0, 100.0], [0.0, 0.0, 0.0]);
    let sigma_near = state.project_pixel_uncertainty(1000.0, 1000.0, 100.0);
    let sigma_far = state.project_pixel_uncertainty(1000.0, 1000.0, 500.0);
    check(
        sigma_near > sigma_far,
        "pixel uncertainty decreases with depth",
    )?;
    Ok(())
}
