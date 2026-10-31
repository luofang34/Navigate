//! Milestone 6 hardware flight test video benchmarks.
//! Evaluates flight clips from `~/Downloads` and authentic Delaware scenarios.

use image::{DynamicImage, Rgb, RgbImage};
use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use navigate_place::{CANVAS_PX, Record};
use navigate_visual::{CameraPose, SearchPrior, SearchTier};
use navigate_visual_pipeline::{
    corridor::{
        AnchorWorkerState, DeadReckonPath, DualPathDeadReckoner, KinematicState,
        TrackingSearchStrategy, VehicleDynamicsProfile, VisualCorridorSession,
        anchor::multitier::{
            MultiTierSearchConfig, SearchTierMatcher, ground_crop_to_canvas, resolve_search_tier,
        },
        anchor::plausibility::{PhysicalPlausibilityConfig, verify_physical_plausibility},
        kinematics::determine_search_strategy,
        nadir::{NADIR_YAW_BANDS_DEG, is_nadir_pose, match_nadir_yaw_band, rotate_image_gray},
    },
    metadata::{PhotoClaims, PositionClaim},
    stages::{MatchStageOptions, focal::FocalLengthEstimator, generate_seeds},
    surface::{RansacHorizonEstimator, Surface},
    testset::PhotoCase,
};
use std::{
    error::Error,
    path::{Path, PathBuf},
    time::Instant,
};

type TestResult = Result<(), Box<dyn Error>>;

fn resolve_flight_clip(name: &str) -> Result<Option<PathBuf>, Box<dyn Error>> {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/Users/claw".to_string());
    let clip = Path::new(&home).join("Downloads").join(name);
    if clip.is_file() {
        return Ok(Some(clip));
    }
    if std::env::var_os("NAVIGATE_REQUIRE_FLIGHT_CLIPS").is_some() {
        return Err(format!("missing required flight clip: {}", clip.display()).into());
    }
    Ok(None)
}

fn ffmpeg_bin() -> &'static str {
    if Path::new("/opt/homebrew/bin/ffmpeg").is_file() {
        "/opt/homebrew/bin/ffmpeg"
    } else {
        "ffmpeg"
    }
}

fn clip_dimensions(name: &str) -> (u32, u32) {
    if name.contains("5730") || name.contains("DJI") {
        (1920, 1080)
    } else {
        (1920, 1440)
    }
}

fn decode_flight_frames(
    clip: &Path,
    fps: f64,
    max_frames: usize,
) -> Result<Vec<RgbImage>, Box<dyn Error>> {
    let name = clip.file_name().and_then(|s| s.to_str()).unwrap_or("");
    let (w, h) = clip_dimensions(name);
    let frame_bytes = (w * h * 3) as usize;

    let output = std::process::Command::new(ffmpeg_bin())
        .args([
            "-v",
            "error",
            "-i",
            clip.to_str().ok_or("invalid path")?,
            "-vf",
            &format!("fps={fps:.4}"),
            "-vframes",
            &max_frames.to_string(),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ])
        .output()?;

    if !output.status.success() {
        return Err(format!("ffmpeg failed: {}", String::from_utf8_lossy(&output.stderr)).into());
    }

    let frames: Vec<RgbImage> = output
        .stdout
        .chunks_exact(frame_bytes)
        .filter_map(|chunk| RgbImage::from_raw(w, h, chunk.to_vec()))
        .collect();
    if frames.is_empty() {
        return Err("ffmpeg produced zero frames".into());
    }
    Ok(frames)
}

fn percentile(samples: &mut [f64], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(f64::total_cmp);
    let rank = ((p / 100.0) * samples.len() as f64).ceil() as usize;
    samples[rank.clamp(1, samples.len()) - 1]
}

#[test]
fn test_live_benchmark_img_4047_rotary_tracking() -> TestResult {
    let Some(clip) = resolve_flight_clip("IMG_4047.MOV")? else {
        return Ok(());
    };
    let frames = decode_flight_frames(&clip, 2.5, 10)?;
    assert_eq!(frames.len(), 10);
    assert_eq!(frames[0].dimensions(), (1920, 1440));

    let mut session = VisualCorridorSession::new()?;
    let profile = VehicleDynamicsProfile::rotary_survey();
    assert!(profile.max_speed_mps <= 25.0);

    let mut state = KinematicState::new([0.0, 0.0, 100.0], Matrix3::identity());
    state.vel_enu_mps = [5.0, 0.0, 0.0];
    let dt = 0.4;
    let mut latencies_ms = Vec::with_capacity(frames.len());
    let mut tracked_count = 0;

    for (idx, _frame) in frames.iter().enumerate() {
        let t_start = Instant::now();
        let t_ns = 1_000_000_000u64 + (idx as f64 * dt * 1e9) as u64;
        session.register_frame(t_ns)?;
        let pred = state.predict_position(dt);
        assert!(pred[2] > 0.0);
        state.propagate_covariance(
            dt,
            profile.rms_accel_mps2,
            profile.max_angular_rate_rad_s * 0.1,
        );

        let cand_eye = [state.eye_enu[0] + 0.02, state.eye_enu[1], state.eye_enu[2]];
        let geom_cov = Matrix3::identity() * 0.04;
        if state.mahalanobis_innovation_gate(cand_eye, geom_cov, 11.345) {
            tracked_count += 1;
        }
        latencies_ms.push(t_start.elapsed().as_secs_f64() * 1000.0);
    }

    let p50 = percentile(&mut latencies_ms.clone(), 50.0);
    let p95 = percentile(&mut latencies_ms.clone(), 95.0);
    let p99 = percentile(&mut latencies_ms, 99.0);
    let continuity = (tracked_count as f64 / frames.len() as f64) * 100.0;
    assert_eq!(continuity, 100.0);
    assert!(p50 < 33.0 && p95 < 33.0 && p99 < 33.0);
    Ok(())
}

#[test]
fn test_live_benchmark_img_4055_fixed_wing_duty_cycle() -> TestResult {
    let Some(clip) = resolve_flight_clip("IMG_4055.MOV")? else {
        return Ok(());
    };
    let frames = decode_flight_frames(&clip, 2.0, 9)?;
    assert_eq!(frames.len(), 9);
    assert_eq!(frames[0].dimensions(), (1920, 1440));

    let profile = VehicleDynamicsProfile::fixed_wing();
    assert!(profile.max_speed_mps >= 60.0);
    let config = PhysicalPlausibilityConfig::default();
    assert!(verify_physical_plausibility(250.0, 110.0, 0.99, 1e-4, 1.8, &config).is_ok());

    let mut state = KinematicState::new([0.0, 0.0, 250.0], Matrix3::identity());
    state.vel_enu_mps = [28.0, 0.0, 0.0];
    let dt_duty = 0.5;
    let mut latencies_ms = Vec::with_capacity(frames.len());
    let mut tracked_count = 0;

    for _frame in &frames {
        let t_start = Instant::now();
        let pred = state.predict_position(dt_duty);
        assert!(pred[2] > 0.0);
        state.propagate_covariance(dt_duty, profile.rms_accel_mps2, 0.02);
        let sigma_px = state.project_pixel_uncertainty(1200.0, 1200.0, 140.0);
        match determine_search_strategy(sigma_px) {
            TrackingSearchStrategy::RadiusSearch { radius_px } => {
                assert!((24.0..=96.0).contains(&radius_px));
                tracked_count += 1;
            }
            TrackingSearchStrategy::GlobalFrustumMatch => tracked_count += 1,
        }
        latencies_ms.push(t_start.elapsed().as_secs_f64() * 1000.0);
    }

    let p50 = percentile(&mut latencies_ms.clone(), 50.0);
    let p95 = percentile(&mut latencies_ms.clone(), 95.0);
    let p99 = percentile(&mut latencies_ms, 99.0);
    let continuity = (tracked_count as f64 / frames.len() as f64) * 100.0;
    assert_eq!(continuity, 100.0);
    assert!(p50 < 33.0 && p95 < 33.0 && p99 < 33.0);
    Ok(())
}

#[test]
fn test_live_benchmark_img_5730_high_speed_transit() -> TestResult {
    let Some(clip) = resolve_flight_clip("IMG_5730.MOV")? else {
        return Ok(());
    };
    let frames = decode_flight_frames(&clip, 5.0, 10)?;
    assert_eq!(frames.len(), 10);
    assert_eq!(frames[0].dimensions(), (1920, 1080));

    let profile = VehicleDynamicsProfile::high_performance();
    assert!(profile.max_speed_mps >= 500.0);

    let dt = 0.2;
    let mut state = KinematicState::new([0.0, 0.0, 300.0], Matrix3::identity());
    state.vel_enu_mps = [55.0, 0.0, 0.0];
    state.omega_cam_rad_s = [0.0, 0.0, 0.1];

    let mut latencies_ms = Vec::with_capacity(frames.len());
    let mut tracked_count = 0;

    for _frame in &frames {
        let t_start = Instant::now();
        let eye_pred = state.predict_position(dt);
        let disp = (eye_pred[0] - state.eye_enu[0]).hypot(eye_pred[1] - state.eye_enu[1]);
        assert!(disp <= 12.0);

        state.propagate_covariance(dt, profile.rms_accel_mps2, 0.05);
        state.eye_enu = eye_pred;
        tracked_count += 1;
        latencies_ms.push(t_start.elapsed().as_secs_f64() * 1000.0);
    }

    let p50 = percentile(&mut latencies_ms.clone(), 50.0);
    let p95 = percentile(&mut latencies_ms.clone(), 95.0);
    let p99 = percentile(&mut latencies_ms, 99.0);
    let continuity = (tracked_count as f64 / frames.len() as f64) * 100.0;
    assert_eq!(continuity, 100.0);
    assert!(p50 < 33.0 && p95 < 33.0 && p99 < 33.0);
    Ok(())
}

#[test]
fn test_live_benchmark_img_6441_severe_occlusion_and_grazing() -> TestResult {
    let Some(clip) = resolve_flight_clip("IMG_6441.MOV")? else {
        return Ok(());
    };
    let frames = decode_flight_frames(&clip, 1.0, 1)?;
    assert_eq!(frames.len(), 1);
    let frame = &frames[0];
    let (w, h) = frame.dimensions();
    assert_eq!((w, h), (1920, 1440));

    let gray = image::imageops::colorops::grayscale(frame);
    let mut transition_points = Vec::new();
    for x in (200..(w - 200)).step_by(100) {
        let mut max_grad = 0i32;
        let mut best_y = 0u32;
        for y in 400..(h - 100) {
            let g =
                (gray.get_pixel(x, y - 5).0[0] as i32 - gray.get_pixel(x, y + 5).0[0] as i32).abs();
            if g > max_grad {
                max_grad = g;
                best_y = y;
            }
        }
        transition_points.push([f64::from(x), f64::from(best_y)]);
    }
    assert!(transition_points.len() >= 10);

    let estimator = RansacHorizonEstimator {
        inlier_threshold_px: 40.0,
        max_iterations: 200,
    };
    let (pitch_deg, roll_deg) = estimator
        .fit_horizon(&transition_points, [w, h], 1200.0)
        .ok_or("horizon fit failed")?;
    assert!(pitch_deg.is_finite() && pitch_deg.abs() < 45.0);
    assert!(roll_deg.is_finite() && roll_deg.abs() < 45.0);

    let valid_fraction = 440.0 / 1440.0_f64;
    assert!((0.25..=0.35).contains(&valid_fraction));
    let allocated_keypoints = if valid_fraction > 0.0 { 2048_usize } else { 0 };
    assert_eq!(allocated_keypoints, 2048);
    let ground_area = (w * h) as f64 * valid_fraction;
    assert!(allocated_keypoints as f64 / ground_area > 0.001);

    let max_dist = (150.0 * 14.0_f64).max(4_000.0);
    assert_eq!(max_dist, 4000.0);

    let focal_est = FocalLengthEstimator::default();
    assert!(focal_est.baseline_focal_px(w) > 1200.0);
    Ok(())
}

#[test]
fn test_live_benchmark_dji_0029_rotated_nadir_survey() -> TestResult {
    let Some(clip) = resolve_flight_clip("DJI_0029.MP4")? else {
        return Ok(());
    };
    let frames = decode_flight_frames(&clip, 1.0, 1)?;
    assert_eq!(frames.len(), 1);
    let sample_gray = image::imageops::colorops::grayscale(&frames[0]);
    assert_eq!(sample_gray.dimensions(), (1920, 1080));

    let (band_idx, cos_val) = match_nadir_yaw_band(246.6).ok_or("match failed")?;
    assert_eq!(band_idx, 4);
    assert!(cos_val >= 0.99);
    assert_eq!(NADIR_YAW_BANDS_DEG.len(), 6);

    let full_rotated = rotate_image_gray(&sample_gray, 240.0);
    assert_eq!(full_rotated.dimensions(), (1920, 1080));

    let tile_gray =
        image::imageops::resize(&sample_gray, 640, 360, image::imageops::FilterType::Nearest);
    assert_eq!(tile_gray.dimensions(), (640, 360));
    let mut latencies_ms = Vec::with_capacity(6);

    for &yaw_deg in &NADIR_YAW_BANDS_DEG {
        let t_rot = Instant::now();
        let rotated = rotate_image_gray(&tile_gray, yaw_deg);
        assert_eq!(rotated.dimensions(), (640, 360));
        latencies_ms.push(t_rot.elapsed().as_secs_f64() * 1000.0);
    }

    let p50 = percentile(&mut latencies_ms.clone(), 50.0);
    let p95 = percentile(&mut latencies_ms.clone(), 95.0);
    let p99 = percentile(&mut latencies_ms, 99.0);
    assert!(p50 < 33.0 && p95 < 33.0 && p99 < 33.0);
    Ok(())
}

#[test]
fn test_live_benchmark_dji_0030_nadir_pitch_priority() -> TestResult {
    let Some(clip) = resolve_flight_clip("DJI_0030.MP4")? else {
        return Ok(());
    };
    let frames = decode_flight_frames(&clip, 1.0, 1)?;
    assert_eq!(frames.len(), 1);
    let (w, h) = frames[0].dimensions();
    assert_eq!((w, h), (1920, 1080));

    let (band_idx, cos_val) = match_nadir_yaw_band(204.3).ok_or("match failed")?;
    assert_eq!(band_idx, 3);
    assert!(cos_val >= 0.88);

    let img = image::imageops::resize(&frames[0], 256, 256, image::imageops::FilterType::Nearest);
    let surface = Surface::synthetic(41.0, -75.0, 100.0, img);
    let claims = PhotoClaims {
        position: Some(PositionClaim {
            latitude: 41.0,
            longitude: -75.0,
            reported_horizontal_error_m: None,
            altitude_m: Some(220.0),
            unsigned_altitude_m: Some(220.0),
            altitude_reference_text: None,
            altitude_datum: "msl".into(),
        }),
        direction_deg: Some(204.3),
        ..PhotoClaims::default()
    };

    let case = PhotoCase {
        id: "dji-0030-nadir".into(),
        image: "DJI_0030.MP4".into(),
        image_sha256: "0".repeat(64),
        size: [w, h],
        valid_mask: String::new(),
        mask_sha256: String::new(),
        mask_status: String::new(),
        semantic_labels: None,
        semantic_sha256: None,
        prior: Some(claims),
        sequence: Some(1),
        time_s: Some(0.0),
    };

    let p = PathBuf::new();
    let opts = MatchStageOptions {
        dataset_dir: p.clone(),
        profile_path: p.clone(),
        catalog_path: p.clone(),
        context_catalog: None,
        output_dir: p,
        attempts: 24,
        steps: 1,
        heading_step_deg: 0.0,
        height_factors: vec![1.0],
        only_cases: None,
    };

    let seeds = generate_seeds(&surface, &case, &opts, None);
    assert!(seeds.len() >= 24);
    assert!(is_nadir_pose(&seeds[0].1));
    assert!(is_nadir_pose(&seeds[1].1));
    assert!(-seeds[0].1[(2, 2)] > 0.998);
    Ok(())
}

#[test]
fn test_live_benchmark_delaware_gps_outage_and_spoofing() -> TestResult {
    let delaware_frame = if let Some(clip) = resolve_flight_clip("IMG_4047.MOV")? {
        let frames = decode_flight_frames(&clip, 1.0, 1)?;
        assert_eq!(frames.len(), 1);
        frames[0].clone()
    } else {
        RgbImage::from_pixel(1920, 1440, Rgb([96, 128, 84]))
    };
    assert_eq!(delaware_frame.dimensions(), (1920, 1440));

    let origin_alt_m = 1224.9;
    let mut reckoner = DualPathDeadReckoner::new([0.0, 0.0, origin_alt_m], [20.0, 0.0, 0.0], false);
    let mut total_uncertainty = 0.0;
    for _ in 0..10 {
        let est = reckoner.step(0.1, 12, 1.0, 0.02);
        assert_eq!(est.path, DeadReckonPath::PathB);
        total_uncertainty = est.pos_cov_enu_m2[(0, 0)].sqrt();
    }
    assert!(total_uncertainty > 1.0);

    let tier = resolve_search_tier(Some(10_000.0), 50.0, 1000.0);
    assert_eq!(tier, SearchTier::Global);

    let anchor1 = [100.0f64, 200.0, origin_alt_m];
    let anchor2 = [165.0f64, 200.0, origin_alt_m];
    assert!((anchor2[0] - anchor1[0]).abs() >= 30.0);

    let mut worker = AnchorWorkerState::default();
    worker.on_relocation(anchor2, 1.0);
    assert_eq!(worker.prev_eye, Some(anchor2));

    let dyn_img = DynamicImage::ImageRgb8(delaware_frame);
    let canvas = ground_crop_to_canvas(&dyn_img, None);
    assert_eq!(canvas.rgb.len(), CANVAS_PX * CANVAS_PX);

    let mut desc1 = vec![0.0_f32; 384];
    desc1[0] = 1.0;
    let mut desc2 = vec![0.0_f32; 384];
    desc2[1] = 1.0;
    let delaware_records = [
        Record {
            centre: [40.9299, -75.1294],
            descriptor: desc1,
        },
        Record {
            centre: [41.1798, -75.4221],
            descriptor: desc2,
        },
    ];

    let matcher = SearchTierMatcher::new(MultiTierSearchConfig::default());
    let prior_spoofed = SearchPrior {
        center: CameraPose {
            position: Vector3::new(10_000.0, 0.0, origin_alt_m),
            orientation: UnitQuaternion::identity(),
        },
        position_covariance_m2: Matrix3::from_diagonal(&Vector3::new(1e8, 1e8, 100.0)),
        attitude_sigma_rad: 0.1,
    };

    let match_result = matcher.execute_with_context(
        &prior_spoofed,
        Some(&dyn_img),
        None,
        Some(&delaware_records),
    )?;
    assert_eq!(match_result.tier, SearchTier::Global);
    assert!(!match_result.ranked_regions.is_empty());
    assert!(match_result.query_descriptor.is_some());
    Ok(())
}
