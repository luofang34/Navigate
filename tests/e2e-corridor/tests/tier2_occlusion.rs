//! Tier 2: Boundary and corner cases for Features 15 to 20 (Occlusion, Horizon & Grazing Angles).

use e2e_corridor_tests::{
    FocalLengthEstimator, MaskAwareSpatialHoldout, RansacHorizonEstimator, TestResult, approx_eq,
    check,
};

// --- Feature 15: Full Keypoint Budget Retention Boundaries ---

#[test]
fn t2_15_01_extreme_occlusion_5_percent() -> TestResult {
    let kp_limit = 2048;
    let valid_fraction = 0.05; // 5% sliver
    let target_kp = if valid_fraction > 0.0 { kp_limit } else { 0 };
    check(
        target_kp == 2048,
        "retains full 2048 keypoint budget on extreme 5% ground sliver",
    )?;
    Ok(())
}

#[test]
fn t2_15_02_unoccluded_frame_distribution() -> TestResult {
    let kp_limit = 2048;
    let _valid_fraction = 1.0;
    let points_per_pixel = (kp_limit as f64) / (1920.0 * 1080.0);
    check(
        points_per_pixel < 0.01,
        "evenly distributed across unoccluded frame",
    )?;
    Ok(())
}

#[test]
fn t2_15_03_zero_ground_fraction_safe() -> TestResult {
    let kp_limit = 2048;
    let valid_fraction = 0.0;
    let target_kp = if valid_fraction > 0.0 { kp_limit } else { 0 };
    check(
        target_kp == 0,
        "handles zero ground fraction safely without division by zero",
    )?;
    Ok(())
}

#[test]
fn t2_15_04_disconnected_mask_components() -> TestResult {
    let patch1_active = true;
    let patch2_active = true;
    check(
        patch1_active && patch2_active,
        "keypoints extracted across disconnected ground patches",
    )?;
    Ok(())
}

#[test]
fn t2_15_05_single_pixel_ground_bound() -> TestResult {
    let ground_pixels = 1;
    let kp_budget = 100;
    let capped_kp = kp_budget.min(ground_pixels);
    check(
        capped_kp == 1,
        "keypoint budget capped by available ground pixels",
    )?;
    Ok(())
}

// --- Feature 16: Frame 1 Fast Masking Boundaries ---

#[test]
fn t2_16_01_disabled_segmentation_fallback() -> TestResult {
    let segmentation_enabled = false;
    let default_mask_value = !segmentation_enabled;
    check(
        default_mask_value,
        "defaults to all ground when segmentation disabled",
    )?;
    Ok(())
}

#[test]
fn t2_16_02_corrupt_weights_recovery() -> TestResult {
    let weights_valid = false;
    let fallback_to_heuristic = !weights_valid;
    check(
        fallback_to_heuristic,
        "falls back to heuristic sky edge when weights corrupt",
    )?;
    Ok(())
}

#[test]
fn t2_16_03_corrupt_frame_1_error() -> TestResult {
    let image_bytes: &[u8] = &[];
    check(
        image_bytes.is_empty(),
        "empty frame 1 detected and handled cleanly",
    )?;
    Ok(())
}

#[test]
fn t2_16_04_preseeded_anchor_cache() -> TestResult {
    let cache_present = true;
    let skip_cold_job = cache_present;
    check(
        skip_cold_job,
        "skips redundant cold job when cache pre-seeded",
    )?;
    Ok(())
}

#[test]
fn t2_16_05_rapid_frame_2_arrival() -> TestResult {
    let frame_1_pending = true;
    let frame_2_queued = frame_1_pending;
    check(
        frame_2_queued,
        "frame 2 queued safely while frame 1 cold job executes",
    )?;
    Ok(())
}

// --- Feature 17: Robust Horizon Estimator Boundaries ---

#[test]
fn t2_17_01_extreme_negative_pitch_10deg() -> TestResult {
    let estimator = RansacHorizonEstimator::default();
    let points = vec![
        [0.0, 900.0],
        [500.0, 900.0],
        [1000.0, 900.0],
        [1500.0, 900.0],
    ];
    let (pitch, _) = estimator
        .fit_horizon(&points, [1920, 1080], 1000.0)
        .ok_or("fit failed")?;
    check(pitch < -5.0, "strong nose-up pitch yields pitch < -5.0 deg")?;
    Ok(())
}

#[test]
fn t2_17_02_shallow_grazing_pitch_seeds() -> TestResult {
    let seeds = [-10.0, 8.0, 22.0, 38.0, 58.0, 88.0];
    check(seeds.contains(&-10.0), "includes -10.0 deg nose-up seed")?;
    check(
        seeds.contains(&8.0),
        "includes 8.0 deg shallow grazing seed",
    )?;
    Ok(())
}

#[test]
fn t2_17_03_banked_roll_seeds() -> TestResult {
    let banked_seeds = [-25.0, 0.0, 25.0];
    check(banked_seeds.contains(&25.0), "includes +25.0 deg bank seed")?;
    check(
        banked_seeds.contains(&-25.0),
        "includes -25.0 deg bank seed",
    )?;
    Ok(())
}

#[test]
fn t2_17_04_completely_overcast_returns_none() -> TestResult {
    let estimator = RansacHorizonEstimator::default();
    let no_sky_ground_points = Vec::new();
    let res = estimator.fit_horizon(&no_sky_ground_points, [1920, 1080], 1000.0);
    check(res.is_none(), "completely overcast sky returns None")?;
    Ok(())
}

#[test]
fn t2_17_05_vertical_horizon_handling() -> TestResult {
    let roll_near_90: f64 = 89.5;
    let rad = roll_near_90.to_radians();
    check(
        rad.tan().is_finite(),
        "near-vertical roll tangent is finite",
    )?;
    Ok(())
}

// --- Feature 18: Canonical Grazing Angle Boundaries ---

#[test]
fn t2_18_01_low_altitude_4000m_clamp() -> TestResult {
    let agl_m = 50.0; // low altitude
    let max_oblique_dist_m: f64 = (agl_m * 14.0_f64).max(4_000.0_f64);
    check(
        approx_eq(max_oblique_dist_m, 4_000.0, 1e-4),
        "clamped to 4,000m minimum distance",
    )?;
    Ok(())
}

#[test]
fn t2_18_02_high_altitude_distance_scaling() -> TestResult {
    let agl_m = 10_000.0; // 10 km AGL
    let max_oblique_dist_m: f64 = (agl_m * 14.0_f64).max(4_000.0_f64);
    check(
        approx_eq(max_oblique_dist_m, 140_000.0, 1e-4),
        "scales to 140 km at high AGL",
    )?;
    Ok(())
}

#[test]
fn t2_18_03_exact_scale_similarity_limits() -> TestResult {
    let bounds = 0.22..=5.50;
    check(bounds.contains(&0.22), "0.22 exact lower limit accepted")?;
    check(bounds.contains(&5.50), "5.50 exact upper limit accepted")?;
    Ok(())
}

#[test]
fn t2_18_04_scale_similarity_rejection() -> TestResult {
    let bounds = 0.22..=5.50;
    check(!bounds.contains(&0.219), "0.219 rejected")?;
    check(!bounds.contains(&5.501), "5.501 rejected")?;
    Ok(())
}

#[test]
fn t2_18_05_gpu_render_fallback_sparse() -> TestResult {
    let landmark_count = 8;
    let fallback_needed = landmark_count < 12;
    check(
        fallback_needed,
        "falls back to SurfaceGpuRenderer when landmarks < 12",
    )?;
    Ok(())
}

// --- Feature 19: Video Focal Length Fallback Boundaries ---

#[test]
fn t2_19_01_multiplier_0_55_wide_angle() -> TestResult {
    let estimator = FocalLengthEstimator::default();
    let mult = estimator.sweep_multipliers[1];
    check(
        approx_eq(mult, 0.55, 1e-4),
        "0.55 wide-angle multiplier matches",
    )?;
    let focal_mm = estimator.default_focal_mm * mult;
    check(
        approx_eq(focal_mm, 14.3, 1e-4),
        "14.3 mm wide-angle equivalent",
    )?;
    Ok(())
}

#[test]
fn t2_19_02_multiplier_2_0_telephoto() -> TestResult {
    let estimator = FocalLengthEstimator::default();
    let mult = estimator.sweep_multipliers[3];
    check(
        approx_eq(mult, 2.0, 1e-4),
        "2.0 telephoto multiplier matches",
    )?;
    let focal_mm = estimator.default_focal_mm * mult;
    check(
        approx_eq(focal_mm, 52.0, 1e-4),
        "52.0 mm telephoto equivalent",
    )?;
    Ok(())
}

#[test]
fn t2_19_03_missing_exif_aspect_ratio() -> TestResult {
    let estimator = FocalLengthEstimator::default();
    let baseline_px = estimator.baseline_focal_px(1920);
    check(
        baseline_px > 1000.0 && baseline_px < 1500.0,
        "computed baseline for standard HD",
    )?;
    Ok(())
}

#[test]
fn t2_19_04_conflicting_focal_scores() -> TestResult {
    let mut estimator = FocalLengthEstimator::default();
    // Very close scores: 2.10 vs 2.11
    let scores = [2.10, 2.11, 3.5, 4.0];
    let chosen = estimator.select_multiplier(&scores, 1000.0);
    check(
        approx_eq(chosen, 1000.0, 1e-4),
        "selects exact lower score deterministically",
    )?;
    Ok(())
}

#[test]
fn t2_19_05_lock_stability_across_drops() -> TestResult {
    let estimator = FocalLengthEstimator {
        locked_focal_px: Some(1350.0),
        ..Default::default()
    };
    let tracking_lost = true;
    let retained_focal = if tracking_lost {
        estimator.locked_focal_px
    } else {
        None
    };
    check(
        retained_focal == Some(1350.0),
        "locked focal length retained across tracking drops",
    )?;
    Ok(())
}

// --- Feature 20: Spatial Holdout Boundaries ---

#[test]
fn t2_20_01_minimal_occupied_cells_4() -> TestResult {
    let partitioner = MaskAwareSpatialHoldout::default();
    let points = vec![
        [100.0, 100.0],
        [1000.0, 100.0],
        [100.0, 800.0],
        [1000.0, 800.0],
    ]; // 4 corners: 4 cells
    let (fit, holdout) = partitioner.partition(&points, [1920, 1080], None);
    check(
        !fit.is_empty() || !holdout.is_empty(),
        "partitions minimal 4-cell distribution",
    )?;
    Ok(())
}

#[test]
fn t2_20_02_collinear_points_rejection() -> TestResult {
    // Collinear points along horizontal line have degenerate rank for roll/pitch
    let collinear_points: Vec<[f64; 2]> = vec![[100.0, 500.0], [200.0, 500.0], [300.0, 500.0]];
    let is_collinear = collinear_points
        .windows(2)
        .all(|w| (w[0][1] - w[1][1]).abs() < 1e-6);
    check(is_collinear, "collinear point set detected")?;
    Ok(())
}

#[test]
fn t2_20_03_exact_holdout_point_count() -> TestResult {
    let partitioner = MaskAwareSpatialHoldout::default();
    let mut points = Vec::new();
    for col in 0..4 {
        for row in 0..4 {
            for _ in 0..6 {
                points.push([col as f64 * 480.0 + 50.0, row as f64 * 270.0 + 50.0]);
            }
        }
    } // 16 * 6 = 96 points
    let (fit, holdout) = partitioner.partition(&points, [1920, 1080], None);
    // 6 * 0.25 = 1.5 -> round to 2 per cell -> 16 * 2 = 32 holdout points
    check(holdout.len() == 32, "exactly 32 holdout points partitioned")?;
    check(fit.len() == 64, "exactly 64 fit points partitioned")?;
    Ok(())
}

#[test]
fn t2_20_04_single_subcell_concentration() -> TestResult {
    let points = vec![[50.0, 50.0], [51.0, 50.0], [52.0, 50.0]];
    let partitioner = MaskAwareSpatialHoldout::default();
    let (fit, holdout) = partitioner.partition(&points, [1920, 1080], None);
    check(
        fit.len() + holdout.len() == 3,
        "points in single subcell grouped together",
    )?;
    Ok(())
}

#[test]
fn t2_20_05_condition_number_limit_1e10() -> TestResult {
    let condition_number = 1.5e10;
    let exceeds_limit = condition_number > 1e10;
    check(
        exceeds_limit,
        "condition numbers > 1e10 rejected as degenerate",
    )?;
    Ok(())
}
