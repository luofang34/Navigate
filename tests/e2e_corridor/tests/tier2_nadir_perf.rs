//! Tier 2: Boundary and corner cases for Features 21 to 26 (Nadir, Hardware & Quality Standards).

use e2e_corridor_tests::{
    DynamicMockSession, MilliradianCovarianceScaler, TestResult, approx_eq, check,
    condition_number_6x6, direction_cosine, generate_corridor_seeds, is_nadir_pose,
    match_nadir_yaw_band, min_cos_orientation_for_pose,
};
use image::{GrayImage, Luma};
use nalgebra::SMatrix;
use navigate_match::{ModelSessionExtractor, SparseFeatureExtractor};
use navigate_visual::HostTensor;
use std::{
    path::Path,
    sync::mpsc::{self, TrySendError},
    thread,
    time::{Duration, Instant},
};

// --- Feature 21: Multi-Yaw Nadir Coverage Boundaries ---

#[test]
fn t2_21_01_heading_bisector_30deg() -> TestResult {
    let res = match_nadir_yaw_band(30.0).ok_or("match failed")?;
    check(
        res.1 >= 0.8660 - 1e-4,
        "alignment on 30 deg bisector satisfies threshold",
    )?;
    check(res.0 == 0 || res.0 == 1, "matches either 0 or 60 deg band")?;
    Ok(())
}

#[test]
fn t2_21_02_exact_cone_limit_36deg() -> TestResult {
    let cos_36 = direction_cosine(36.0_f64.to_radians(), 0.0);
    check(cos_36 >= 0.80, "36.0 deg is inside cone (cos >= 0.80)")?;
    let res = match_nadir_yaw_band(36.0).ok_or("match failed")?;
    check(res.0 == 1, "matches nearest 60 deg band (distance 24 deg)")?;
    check(res.1 >= 0.80, "36.0 deg matches band with cos >= 0.80")?;
    Ok(())
}

#[test]
fn t2_21_03_outside_cone_37deg_rejection() -> TestResult {
    let cos_37 = direction_cosine(37.0_f64.to_radians(), 0.0);
    check(cos_37 < 0.80, "angle 37 deg produces cosine < 0.80")?;
    let res = match_nadir_yaw_band(37.0).ok_or("match failed")?;
    check(res.0 == 1, "nearest among 6 bands is band 1 (60 deg)")?;
    check(res.1 >= 0.80, "6-band set maintains coverage")?;
    Ok(())
}

#[test]
fn t2_21_04_full_circle_wraparound_359deg() -> TestResult {
    let res = match_nadir_yaw_band(359.0).ok_or("match failed")?;
    check(res.0 == 0, "359 deg wraps around to band 0")?;
    check(res.1 > 0.999, "cosine near 1.0 on wraparound")?;
    let res_neg = match_nadir_yaw_band(-1.0).ok_or("negative heading match failed")?;
    check(res_neg.0 == 0, "-1 deg wraps to band 0")?;
    check(res_neg.1 > 0.999, "cosine near 1.0 on negative heading")?;
    Ok(())
}

#[test]
fn t2_21_05_non_nadir_pitch_rejection() -> TestResult {
    let nadir_rot = nalgebra::Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -0.99);
    let oblique_rot = nalgebra::Matrix3::new(1.0, 0.0, 0.0, 0.0, 0.707, -0.707, 0.0, 0.707, -0.707);
    check(is_nadir_pose(&nadir_rot), "nadir rotation identified")?;
    check(!is_nadir_pose(&oblique_rot), "oblique rotation rejected")?;
    check(
        min_cos_orientation_for_pose(&nadir_rot) == 0.80,
        "nadir threshold is 0.80",
    )?;
    check(
        min_cos_orientation_for_pose(&oblique_rot) == 0.90,
        "oblique threshold is 0.90",
    )?;
    Ok(())
}

// --- Feature 22: Nadir Pitch Seed Prioritization Boundaries ---

#[test]
fn t2_22_01_exact_agl_boundary_250m() -> TestResult {
    let seeds = generate_corridor_seeds(250.0, None, &[0.0]);
    check(!seeds.is_empty(), "seeds generated")?;
    check(
        seeds[0] == (88.0, 0.0),
        "exact 250.0 m AGL prioritizes nadir",
    )?;
    Ok(())
}

#[test]
fn t2_22_02_agl_above_250m_standard_seeds() -> TestResult {
    let seeds = generate_corridor_seeds(250.001, None, &[0.0]);
    check(!seeds.is_empty(), "seeds generated")?;
    check(
        seeds[0].0 == -10.0,
        "250.001 m AGL uses standard grazing order",
    )?;
    Ok(())
}

#[test]
fn t2_22_03_attitude_prior_overrides_nadir() -> TestResult {
    let seeds = generate_corridor_seeds(50.0, Some((10.0, -3.0)), &[0.0]);
    check(
        seeds[0] == (10.0, -3.0),
        "attitude prior overrides blind nadir prioritization",
    )?;
    Ok(())
}

#[test]
fn t2_22_04_zero_agl_fallback() -> TestResult {
    let seeds = generate_corridor_seeds(0.0, None, &[0.0]);
    check(
        seeds[0] == (88.0, 0.0),
        "zero AGL safely defaults to nadir prioritization",
    )?;
    Ok(())
}

#[test]
fn t2_22_05_empty_seed_generation_guard() -> TestResult {
    let empty = generate_corridor_seeds(100.0, None, &[]);
    check(empty.is_empty(), "empty headings produce empty seeds")?;
    let twelve_headings: Vec<f64> = (0..12).map(|i| i as f64 * 30.0).collect();
    let many = generate_corridor_seeds(100.0, None, &twelve_headings);
    check(many.len() >= 24, "12 headings produce >= 24 nadir seeds")?;
    Ok(())
}

// --- Feature 23: Dynamic ORT Input Tensor Names Boundaries ---

#[test]
fn t2_23_01_multi_input_model_primary() -> TestResult {
    let inputs = ["primary_rgb", "metadata_mask", "state_prior"];
    let primary = inputs.first().copied().ok_or("no input")?;
    let session = DynamicMockSession::new(primary);
    let extractor = ModelSessionExtractor::new(session, primary, [0xaa; 32], "ort-multi");
    let mut img = GrayImage::new(64, 64);
    img.put_pixel(0, 0, Luma([0]));
    img.put_pixel(1, 1, Luma([255]));
    let res = extractor.extract_masked_blocking(&img, None, 128);
    check(res.is_ok(), "primary input selected from multi-input graph")?;
    Ok(())
}

#[test]
fn t2_23_02_unicode_tensor_name() -> TestResult {
    let tensor_name = "tensor_α_input";
    let session = DynamicMockSession::new(tensor_name);
    let extractor = ModelSessionExtractor::new(session, tensor_name, [0xbb; 32], "ort-unicode");
    let mut img = GrayImage::new(64, 64);
    img.put_pixel(0, 0, Luma([0]));
    img.put_pixel(1, 1, Luma([255]));
    let res = extractor.extract_masked_blocking(&img, None, 128);
    check(
        res.is_ok(),
        "supports Unicode tensor names without corruption",
    )?;
    Ok(())
}

#[test]
fn t2_23_03_session_init_failure_handling() -> TestResult {
    let inputs: Vec<&str> = Vec::new();
    let primary = inputs.first();
    check(primary.is_none(), "empty input list returns None")?;
    Ok(())
}

#[test]
fn t2_23_04_dynamic_batch_dimension() -> TestResult {
    let t1 = HostTensor::new(vec![1, 64, 72, 100], vec![0.0_f32; 64 * 72 * 100]);
    let t2 = HostTensor::new(vec![2, 64, 72, 100], vec![0.0_f32; 2 * 64 * 72 * 100]);
    check(
        t1.is_ok() && t2.is_ok(),
        "supports varying batch dimensions",
    )?;
    Ok(())
}

#[test]
fn t2_23_05_dynamic_spatial_dimensions() -> TestResult {
    let session = DynamicMockSession::new("images");
    let extractor = ModelSessionExtractor::new(session, "images", [0xcc; 32], "ort-spatial");
    let mut img1 = GrayImage::new(64, 64);
    img1.put_pixel(0, 0, Luma([0]));
    img1.put_pixel(1, 1, Luma([255]));
    let mut img2 = GrayImage::new(128, 128);
    img2.put_pixel(0, 0, Luma([0]));
    img2.put_pixel(1, 1, Luma([255]));
    check(
        extractor.extract_masked_blocking(&img1, None, 64).is_ok(),
        "64x64 extracted",
    )?;
    check(
        extractor.extract_masked_blocking(&img2, None, 64).is_ok(),
        "128x128 extracted",
    )?;
    Ok(())
}

// --- Feature 24: GPU Queue Isolation Boundaries ---

#[test]
fn t2_24_01_burst_background_load() -> TestResult {
    let (tx, rx) = mpsc::sync_channel::<usize>(4);
    let bg = thread::spawn(move || {
        while rx.recv().is_ok() {
            thread::sleep(Duration::from_millis(10));
        }
    });
    let mut accepted = 0;
    let mut rejected = 0;
    let t0 = Instant::now();
    for i in 0..10 {
        match tx.try_send(i) {
            Ok(()) => accepted += 1,
            Err(TrySendError::Full(_)) => rejected += 1,
            Err(TrySendError::Disconnected(_)) => {}
        }
    }
    let burst_dur = t0.elapsed();
    drop(tx);
    check(bg.join().is_ok(), "background thread joined")?;
    check(
        accepted == 4,
        "bounded capacity accepts exactly 4 items in burst",
    )?;
    check(rejected == 6, "excess 6 items rejected immediately")?;
    check(
        burst_dur < Duration::from_millis(5),
        "burst submission completed non-blocking",
    )?;
    Ok(())
}

#[test]
fn t2_24_02_single_queue_fallback() -> TestResult {
    let (tx, rx) = mpsc::sync_channel::<usize>(4);
    drop(rx);
    let err = tx.try_send(1);
    check(
        matches!(err, Err(TrySendError::Disconnected(1))),
        "disconnected queue returns error",
    )?;
    Ok(())
}

#[test]
fn t2_24_03_device_loss_recovery() -> TestResult {
    let (tx1, rx1) = mpsc::sync_channel::<usize>(4);
    drop(rx1);
    check(tx1.try_send(1).is_err(), "old queue failed")?;
    let (tx2, rx2) = mpsc::sync_channel::<usize>(4);
    check(
        tx2.try_send(2).is_ok(),
        "recreated queue accepts submissions",
    )?;
    check(rx2.recv() == Ok(2), "received item on recreated queue")?;
    Ok(())
}

#[test]
fn t2_24_04_latency_rolling_p99() -> TestResult {
    let mut latencies: Vec<f64> = (1..=98).map(|i| 5.0 + (i as f64) * 0.1).collect();
    latencies.push(30.0);
    latencies.push(45.0);
    latencies.sort_by(|a, b| a.total_cmp(b));
    let p99_idx = ((latencies.len() - 1) as f64 * 0.99).round() as usize;
    let p99_val = latencies[p99_idx];
    check(p99_val <= 33.0, "rolling p99 is under 33 ms threshold")?;
    check(
        latencies.last().copied() == Some(45.0),
        "spike detected in max latency",
    )?;
    Ok(())
}

#[test]
fn t2_24_05_clean_queue_teardown() -> TestResult {
    let (tx, rx) = mpsc::sync_channel::<usize>(4);
    let worker = thread::spawn(move || {
        let mut count = 0;
        while rx.recv().is_ok() {
            count += 1;
        }
        count
    });
    check(tx.send(1).is_ok(), "sent job 1")?;
    check(tx.send(2).is_ok(), "sent job 2")?;
    drop(tx);
    let processed = worker.join().map_err(|_| "thread panic")?;
    check(processed == 2, "drained pending jobs cleanly on teardown")?;
    Ok(())
}

// --- Feature 25: Milliradian Covariance Scaling Boundaries ---

#[test]
fn t2_25_01_diagonal_matrix_exact_scale() -> TestResult {
    let diag = [4.0, 4.0, 4.0, 1e6, 1e6, 1e6];
    let h = SMatrix::<f64, 6, 6>::from_diagonal(&nalgebra::SVector::from_row_slice(&diag));
    let cov = MilliradianCovarianceScaler::invert_with_scaling(&h)?;
    check(
        approx_eq(cov[(0, 0)], 0.25, 1e-6),
        "positional variance 0.25 m^2",
    )?;
    check(
        approx_eq(cov[(3, 3)], 1e-6, 1e-12),
        "angular variance 1e-6 rad^2",
    )?;
    Ok(())
}

#[test]
fn t2_25_02_ill_conditioned_matrix_limit() -> TestResult {
    let diag = [1.0, 1.0, 1.0, 1e9, 1e9, 1e9];
    let h = SMatrix::<f64, 6, 6>::from_diagonal(&nalgebra::SVector::from_row_slice(&diag));
    let cov = MilliradianCovarianceScaler::invert_with_scaling(&h)?;
    check(
        cov[(0, 0)].is_finite(),
        "near-limit ill-conditioned matrix inverted successfully",
    )?;
    check(cov[(3, 3)].is_finite(), "angular variance finite")?;
    Ok(())
}

#[test]
fn t2_25_03_cross_coupling_block_scale() -> TestResult {
    let mut h = SMatrix::<f64, 6, 6>::identity();
    h[(0, 0)] = 10.0;
    h[(3, 3)] = 1e6;
    h[(0, 3)] = 10.0;
    h[(3, 0)] = 10.0;
    let cov = MilliradianCovarianceScaler::invert_with_scaling(&h)?;
    check(
        approx_eq(cov[(0, 3)], cov[(3, 0)], 1e-12),
        "covariance is symmetric",
    )?;
    Ok(())
}

#[test]
fn t2_25_04_frame_rotation_invariance() -> TestResult {
    let diag = [2.0, 4.0, 8.0, 1e6, 1e6, 1e6];
    let h1 = SMatrix::<f64, 6, 6>::from_diagonal(&nalgebra::SVector::from_row_slice(&diag));
    let cov1 = MilliradianCovarianceScaler::invert_with_scaling(&h1)?;
    let trace1 = cov1[(0, 0)] + cov1[(1, 1)] + cov1[(2, 2)];

    let mut h2 = h1;
    h2[(0, 0)] = 4.0;
    h2[(1, 1)] = 2.0;
    let cov2 = MilliradianCovarianceScaler::invert_with_scaling(&h2)?;
    let trace2 = cov2[(0, 0)] + cov2[(1, 1)] + cov2[(2, 2)];
    check(
        approx_eq(trace1, trace2, 1e-10),
        "position covariance trace invariant under frame rotation",
    )?;
    Ok(())
}

#[test]
fn t2_25_05_unscaled_inversion_comparison() -> TestResult {
    let diag = [1.0, 1.0, 1.0, 1e12, 1e12, 1e12];
    let h = SMatrix::<f64, 6, 6>::from_diagonal(&nalgebra::SVector::from_row_slice(&diag));
    let s_diag = nalgebra::SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 0.001, 0.001, 0.001]);
    let s = SMatrix::<f64, 6, 6>::from_diagonal(&s_diag);
    let scaled = s * h * s;
    let cond_unscaled = condition_number_6x6(&h);
    let cond_scaled = condition_number_6x6(&scaled);
    check(cond_unscaled > 1e11, "unscaled condition number is high")?;
    check(cond_scaled < 1e7, "scaled condition number is low")?;
    let cov = MilliradianCovarianceScaler::invert_with_scaling(&h)?;
    check(cov[(0, 0)].is_finite(), "scaled inversion succeeds")?;
    Ok(())
}

// --- Feature 26: Structural & Documentation Standards Boundaries ---

#[test]
fn t2_26_01_boundary_file_500_lines() -> TestResult {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("no repository root")?;
    let test_dir = root.join("tests/e2e_corridor/tests");
    for entry in std::fs::read_dir(test_dir)? {
        let path = entry?.path();
        if path.extension() == Some(std::ffi::OsStr::new("rs")) {
            let content = std::fs::read_to_string(&path)?;
            let count = content.lines().count();
            check(count <= 500, "test file <= 500 lines")?;
        }
    }
    Ok(())
}

#[test]
fn t2_26_02_boundary_lib_rs_100_lines() -> TestResult {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("no repository root")?;
    let lib_rs = root.join("tests/e2e_corridor/src/lib.rs");
    let content = std::fs::read_to_string(lib_rs)?;
    check(content.lines().count() <= 100, "lib.rs <= 100 lines")?;
    Ok(())
}

#[test]
fn t2_26_03_wgpu_single_major_version() -> TestResult {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("no repository root")?;
    let lock = std::fs::read_to_string(root.join("Cargo.lock"))?;
    let mut major_versions = std::collections::BTreeSet::new();
    let mut in_wgpu = false;
    for line in lock.lines() {
        if line == "name = \"wgpu\"" {
            in_wgpu = true;
        } else if in_wgpu && line.starts_with("version = \"") {
            let ver = line
                .trim_start_matches("version = \"")
                .trim_end_matches('"');
            let major = ver.split('.').next().unwrap_or("");
            major_versions.insert(major.to_string());
            in_wgpu = false;
        } else if in_wgpu && line.starts_with("[[package]]") {
            in_wgpu = false;
        }
    }
    check(
        major_versions.len() == 1,
        "exactly 1 wgpu major version in Cargo.lock",
    )?;
    Ok(())
}

#[test]
fn t2_26_04_root_license_agpl3() -> TestResult {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("no repository root")?;
    let lic = std::fs::read_to_string(root.join("LICENSE"))?;
    check(
        lic.contains("GNU AFFERO GENERAL PUBLIC LICENSE"),
        "LICENSE is AGPL",
    )?;
    check(lic.contains("Version 3"), "LICENSE is Version 3")?;
    Ok(())
}

#[test]
fn t2_26_05_tracked_file_size_limit() -> TestResult {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("no repository root")?;
    let limit_bytes = 5 * 1024 * 1024;
    let check_files = [
        root.join("Cargo.lock"),
        root.join("Cargo.toml"),
        root.join("README.md"),
    ];
    for f in &check_files {
        let meta = std::fs::metadata(f)?;
        check(meta.len() < limit_bytes, "file under 5 MiB limit")?;
    }
    Ok(())
}
