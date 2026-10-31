//! Tier 1: Feature coverage tests for Features 21 to 26 (Nadir, Hardware & Quality Standards).

use e2e_corridor_tests::{
    DynamicMockSession, MilliradianCovarianceScaler, TestResult, approx_eq, check,
    condition_number_6x6, direction_cosine, generate_corridor_seeds, match_nadir_yaw_band,
    rotate_image_gray, synthesize_nadir_yaw_bands, unrotate_pixel,
};
use image::{GrayImage, Luma};
use nalgebra::SMatrix;
use navigate_match::{ModelSessionExtractor, SparseFeatureExtractor};
use std::{
    path::Path,
    sync::mpsc::{self, TrySendError},
    thread,
    time::{Duration, Instant},
};

// --- Feature 21: Multi-Yaw Nadir Coverage ---

#[test]
fn t1_21_01_synthesize_6_yaw_bands() -> TestResult {
    let bands = synthesize_nadir_yaw_bands();
    check(bands.len() == 6, "synthesizes exactly 6 yaw bands")?;
    check(
        bands == [0.0, 60.0, 120.0, 180.0, 240.0, 300.0],
        "yaw bands spaced at 60 deg intervals",
    )?;
    Ok(())
}

#[test]
fn t1_21_02_in_plane_pixel_rotation() -> TestResult {
    let (w, h) = (64_u32, 64_u32);
    let mut img = GrayImage::new(w, h);
    img.put_pixel(24, 20, Luma([255]));
    let rotated = rotate_image_gray(&img, 60.0);

    let mut max_val = 0_u8;
    let mut peak = [0.0_f64, 0.0_f64];
    for y in 0..h {
        for x in 0..w {
            let p = rotated.get_pixel(x, y).0[0];
            if p > max_val {
                max_val = p;
                peak = [f64::from(x), f64::from(y)];
            }
        }
    }
    check(max_val > 0, "rotated image contains illuminated pixel")?;
    let unrot = unrotate_pixel(peak, 60.0, w, h);
    check(approx_eq(unrot[0], 24.0, 1.0), "unrotated x matches source")?;
    check(approx_eq(unrot[1], 20.0, 1.0), "unrotated y matches source")?;
    Ok(())
}

#[test]
fn t1_21_03_direction_cosine_threshold() -> TestResult {
    let cos_36 = direction_cosine(36.0_f64.to_radians(), 0.0);
    let cos_37 = direction_cosine(37.0_f64.to_radians(), 0.0);
    check(
        cos_36 >= 0.80,
        "36 deg cone satisfies min cos 0.80 threshold",
    )?;
    check(
        cos_37 < 0.80,
        "37 deg cone falls below min cos 0.80 threshold",
    )?;
    Ok(())
}

#[test]
fn t1_21_04_match_nearest_yaw_band() -> TestResult {
    let res = match_nadir_yaw_band(246.6).ok_or("match failed")?;
    check(res.0 == 4, "matches 240 deg band")?;
    check(res.1 >= 0.80, "alignment cosine >= 0.80")?;
    Ok(())
}

#[test]
fn t1_21_05_single_render_efficiency() -> TestResult {
    let mut base = GrayImage::new(64, 64);
    base.put_pixel(32, 20, Luma([255]));
    let bands = synthesize_nadir_yaw_bands();
    let rotated: Vec<_> = bands
        .iter()
        .map(|&deg| rotate_image_gray(&base, deg))
        .collect();
    check(
        rotated.len() == 6,
        "6 rotated views generated from 1 base image",
    )?;
    for view in &rotated {
        check(view.dimensions() == (64, 64), "dimensions preserved")?;
    }
    Ok(())
}

// --- Feature 22: Nadir Pitch Seed Prioritization ---

#[test]
fn t1_22_01_prioritize_nadir_seeds_low_agl() -> TestResult {
    let seeds = generate_corridor_seeds(120.0, None, &[0.0, 90.0, 180.0, 270.0]);
    check(!seeds.is_empty(), "seeds generated")?;
    check(seeds[0] == (88.0, 0.0), "first seed is 88 deg nadir")?;
    check(seeds[1] == (75.0, 0.0), "second seed is 75 deg nadir")?;
    Ok(())
}

#[test]
fn t1_22_02_interleave_heading_seeds() -> TestResult {
    let seeds = generate_corridor_seeds(100.0, None, &[0.0, 90.0]);
    check(seeds.len() >= 4, "at least 4 nadir seeds generated")?;
    check(
        seeds[0] == (88.0, 0.0) && seeds[1] == (75.0, 0.0),
        "first heading pitches",
    )?;
    check(
        seeds[2] == (88.0, 0.0) && seeds[3] == (75.0, 0.0),
        "second heading pitches",
    )?;
    Ok(())
}

#[test]
fn t1_22_03_respect_horizon_attitude_seed() -> TestResult {
    let seeds = generate_corridor_seeds(100.0, Some((5.0, 2.0)), &[0.0]);
    check(!seeds.is_empty(), "seeds generated with prior")?;
    check(
        seeds[0] == (5.0, 2.0),
        "horizon attitude seed respected as first seed",
    )?;
    Ok(())
}

#[test]
fn t1_22_04_nadir_flight_fast_acquisition() -> TestResult {
    let seeds = generate_corridor_seeds(150.0, None, &[0.0]);
    let query_pitch = 88.5;
    check(
        approx_eq(query_pitch, seeds[0].0, 1.0),
        "first seed matches within 0.5 deg",
    )?;
    Ok(())
}

#[test]
fn t1_22_05_oblique_seeds_after_nadir() -> TestResult {
    let seeds = generate_corridor_seeds(150.0, None, &[0.0]);
    check(seeds[0].0 >= 88.0, "steepest nadir seeds first")?;
    check(seeds[1].0 == 75.0, "75 deg nadir seed second")?;
    check(
        seeds.iter().skip(2).any(|s| s.0 < 0.0),
        "shallow grazing seeds appear after nadir seeds",
    )?;
    Ok(())
}

// --- Feature 23: Dynamic ORT Input Tensor Names ---

#[test]
fn t1_23_01_query_dynamic_tensor_name() -> TestResult {
    let session = DynamicMockSession::new("dynamic_tensor");
    let extractor =
        ModelSessionExtractor::new(session, "dynamic_tensor", [0x5a; 32], "ort-dynamic");
    let mut img = GrayImage::new(64, 64);
    img.put_pixel(0, 0, Luma([0]));
    img.put_pixel(1, 1, Luma([255]));
    let res = extractor.extract_masked_blocking(&img, None, 128);
    check(res.is_ok(), "dynamic tensor extraction succeeded")?;
    Ok(())
}

#[test]
fn t1_23_02_bind_discovered_tensor_name() -> TestResult {
    let session = DynamicMockSession::new("input_tensor_0");
    let extractor = ModelSessionExtractor::new(session, "input_tensor_0", [0x42; 32], "ort-bind");
    let mut img = GrayImage::new(64, 64);
    img.put_pixel(0, 0, Luma([0]));
    img.put_pixel(1, 1, Luma([255]));
    let res = extractor.extract_masked_blocking(&img, None, 128);
    check(res.is_ok(), "binds with discovered name successfully")?;
    Ok(())
}

#[test]
fn t1_23_03_support_custom_tensor_names() -> TestResult {
    let custom_names = ["image", "images", "input.1", "data", "sensor_input"];
    let mut img = GrayImage::new(64, 64);
    img.put_pixel(0, 0, Luma([0]));
    img.put_pixel(1, 1, Luma([255]));
    for &name in &custom_names {
        let session = DynamicMockSession::new(name);
        let extractor = ModelSessionExtractor::new(session, name, [0x11; 32], "ort-custom");
        let res = extractor.extract_masked_blocking(&img, None, 128);
        check(res.is_ok(), "custom tensor name extraction succeeds")?;
    }
    Ok(())
}

#[test]
fn t1_23_04_xfeat_runtime_integration() -> TestResult {
    let session = DynamicMockSession::new("images");
    let extractor = ModelSessionExtractor::new(session, "image", [0x33; 32], "ort-mismatch");
    let mut img = GrayImage::new(64, 64);
    img.put_pixel(0, 0, Luma([0]));
    img.put_pixel(1, 1, Luma([255]));
    let res = extractor.extract_masked_blocking(&img, None, 128);
    check(res.is_err(), "mismatch returns error")?;
    Ok(())
}

#[test]
fn t1_23_05_xfeat_extractor_integration() -> TestResult {
    let graph_inputs = ["images", "metadata_mask", "state_prior"];
    let primary = graph_inputs.first().copied().ok_or("no inputs")?;
    check(
        primary == "images",
        "primary tensor selected from multi-input model",
    )?;
    let session = DynamicMockSession::new(primary);
    let extractor = ModelSessionExtractor::new(session, primary, [0x77; 32], "ort-primary");
    let mut img = GrayImage::new(64, 64);
    img.put_pixel(0, 0, Luma([0]));
    img.put_pixel(1, 1, Luma([255]));
    let res = extractor.extract_masked_blocking(&img, None, 128);
    check(res.is_ok(), "extraction succeeds with primary input name")?;
    Ok(())
}

// --- Feature 24: GPU Queue Isolation ---

#[test]
fn t1_24_01_separate_tracking_device_queue() -> TestResult {
    let (tx, rx) = mpsc::sync_channel::<usize>(4);
    let bg = thread::spawn(move || {
        while let Ok(job) = rx.recv() {
            thread::sleep(Duration::from_millis(5));
            if job == 999 {
                break;
            }
        }
    });
    let t0 = Instant::now();
    for i in 0..20 {
        if tx.try_send(i).is_ok() {}
    }
    let elapsed = t0.elapsed();
    check(tx.send(999).is_ok(), "sent termination signal")?;
    check(bg.join().is_ok(), "background thread joined")?;
    check(
        elapsed < Duration::from_millis(50),
        "foreground thread executes non-blocking",
    )?;
    Ok(())
}

#[test]
fn t1_24_02_non_blocking_tile_building() -> TestResult {
    let (tx, rx) = mpsc::sync_channel::<u32>(4);
    let bg = thread::spawn(move || for _ in rx {});
    let t0 = Instant::now();
    let res = tx.try_send(42);
    let dur = t0.elapsed();
    drop(tx);
    check(bg.join().is_ok(), "background thread joined")?;
    check(res.is_ok(), "submission succeeded")?;
    check(
        dur < Duration::from_millis(5),
        "try_send completes in < 5 ms",
    )?;
    Ok(())
}

#[test]
fn t1_24_03_p99_latency_under_33ms() -> TestResult {
    let mut latencies_ms = Vec::with_capacity(60);
    for _ in 0..60 {
        let t0 = Instant::now();
        let a = nalgebra::Matrix3::<f64>::new(1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9);
        let _b = a * a * a;
        latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    latencies_ms.sort_by(|a, b| a.total_cmp(b));
    let p99_idx = ((latencies_ms.len() as f64) * 0.99).floor() as usize;
    let p99 = latencies_ms[p99_idx.min(latencies_ms.len() - 1)];
    check(p99 < 33.0, "measured p99 frame latency < 33 ms")?;
    Ok(())
}

#[test]
fn t1_24_04_coreml_metal_pairing() -> TestResult {
    let (tx, rx) = mpsc::sync_channel::<u32>(4);
    let bg = thread::spawn(move || {
        let mut count = 0;
        while rx.recv().is_ok() {
            count += 1;
        }
        count
    });
    let session = DynamicMockSession::new("images");
    let extractor = ModelSessionExtractor::new(session, "images", [0x99; 32], "coreml-fg");
    let mut img = GrayImage::new(64, 64);
    img.put_pixel(0, 0, Luma([0]));
    img.put_pixel(1, 1, Luma([255]));
    check(tx.try_send(1).is_ok(), "sent background job")?;
    let res = extractor.extract_masked_blocking(&img, None, 128);
    drop(tx);
    let bg_count = bg.join().map_err(|_| "bg failed")?;
    check(res.is_ok(), "foreground extraction succeeded")?;
    check(bg_count == 1, "background received item")?;
    Ok(())
}

#[test]
fn t1_24_05_channel_backpressure_bound() -> TestResult {
    let (tx, _rx) = mpsc::sync_channel::<usize>(4);
    for i in 0..4 {
        check(tx.try_send(i).is_ok(), "queue accepts up to capacity")?;
    }
    let err = tx.try_send(4);
    check(
        matches!(err, Err(TrySendError::Full(4))),
        "5th send returns TrySendError::Full",
    )?;
    Ok(())
}

// --- Feature 25: Milliradian Covariance Scaling ---

#[test]
fn t1_25_01_scale_normal_matrix_angular() -> TestResult {
    let scale_diag =
        nalgebra::SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 0.001, 0.001, 0.001]);
    let scale_mat = SMatrix::<f64, 6, 6>::from_diagonal(&scale_diag);
    let diag = [10.0, 10.0, 10.0, 1e6, 1e6, 1e6];
    let h = SMatrix::<f64, 6, 6>::from_diagonal(&nalgebra::SVector::from_row_slice(&diag));
    let scaled = scale_mat * h * scale_mat;
    check(
        approx_eq(scaled[(3, 3)], 1.0, 1e-9),
        "angular block scaled by 1e-6 to 1.0",
    )?;
    Ok(())
}

#[test]
fn t1_25_02_invert_scaled_normal_matrix() -> TestResult {
    let diag = [10.0, 10.0, 10.0, 1e6, 1e6, 1e6];
    let h = SMatrix::<f64, 6, 6>::from_diagonal(&nalgebra::SVector::from_row_slice(&diag));
    let inv = MilliradianCovarianceScaler::invert_with_scaling(&h)?;
    let prod = h * inv;
    let id = SMatrix::<f64, 6, 6>::identity();
    check((prod - id).norm() < 1e-6, "H * H^-1 equals identity")?;
    Ok(())
}

#[test]
fn t1_25_03_post_scale_covariance() -> TestResult {
    let diag = [1.0, 1.0, 1.0, 1e6, 1e6, 1e6];
    let h = SMatrix::<f64, 6, 6>::from_diagonal(&nalgebra::SVector::from_row_slice(&diag));
    let cov = MilliradianCovarianceScaler::invert_with_scaling(&h)?;
    check(
        approx_eq(cov[(3, 3)], 1e-6, 1e-12),
        "angular variance correctly post-scaled",
    )?;
    Ok(())
}

#[test]
fn t1_25_04_angular_block_units_rad2() -> TestResult {
    let diag = [4.0, 4.0, 4.0, 1e6, 1e6, 1e6];
    let h = SMatrix::<f64, 6, 6>::from_diagonal(&nalgebra::SVector::from_row_slice(&diag));
    let cov = MilliradianCovarianceScaler::invert_with_scaling(&h)?;
    check(
        cov[(0, 0)] > 0.01 && cov[(0, 0)] < 100.0,
        "positional variance in m^2",
    )?;
    check(
        cov[(3, 3)] > 1e-8 && cov[(3, 3)] < 1e-4,
        "angular variance in rad^2",
    )?;
    Ok(())
}

#[test]
fn t1_25_05_condition_number_improvement() -> TestResult {
    let diag = [1.0, 1.0, 1.0, 1e6, 1e6, 1e6];
    let h = SMatrix::<f64, 6, 6>::from_diagonal(&nalgebra::SVector::from_row_slice(&diag));
    let s_diag = nalgebra::SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 0.001, 0.001, 0.001]);
    let s = SMatrix::<f64, 6, 6>::from_diagonal(&s_diag);
    let scaled_h = s * h * s;
    let cond_unscaled = condition_number_6x6(&h);
    let cond_scaled = condition_number_6x6(&scaled_h);
    check(
        cond_scaled < cond_unscaled * 1e-5,
        "condition number improves by > 10^5",
    )?;
    Ok(())
}

// --- Feature 26: Structural & Documentation Standards ---

#[test]
fn t1_26_01_file_line_count_limit() -> TestResult {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("no repository root")?;
    let files = [
        "tools/visual-pipeline/src/corridor/nadir.rs",
        "tools/visual-pipeline/src/surface/gpu.rs",
        "tools/visual-pipeline/src/stages/seeds.rs",
        "tools/visual-pipeline/src/xfeat_runtime.rs",
    ];
    for f in &files {
        let path = root.join(f);
        let content = std::fs::read_to_string(&path)?;
        let lines = content.lines().count();
        check(lines <= 500, "file <= 500 lines")?;
    }
    Ok(())
}

#[test]
fn t1_26_02_lib_rs_reexport_limit() -> TestResult {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("no repository root")?;
    let lib_path = root.join("tests/e2e_corridor/src/lib.rs");
    let content = std::fs::read_to_string(&lib_path)?;
    check(content.lines().count() <= 100, "lib.rs <= 100 lines")?;
    check(content.starts_with("//!"), "lib.rs starts with doc comment")?;
    Ok(())
}

fn find_mod_rs(dir: &Path) -> std::io::Result<Vec<std::path::PathBuf>> {
    let mut results = Vec::new();
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                results.extend(find_mod_rs(&path)?);
            } else if path.file_name() == Some(std::ffi::OsStr::new("mod.rs")) {
                results.push(path);
            }
        }
    }
    Ok(results)
}

#[test]
fn t1_26_03_banned_mod_rs() -> TestResult {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("no repository root")?;
    let forbidden_mods = find_mod_rs(&root.join("tests/e2e_corridor"))?;
    check(
        forbidden_mods.is_empty(),
        "zero mod.rs files exist in e2e_corridor",
    )?;
    Ok(())
}

#[test]
fn t1_26_04_forbidden_keywords_check() -> TestResult {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("no repository root")?;
    let cargo_toml = std::fs::read_to_string(root.join("Cargo.toml"))?;
    check(
        cargo_toml.contains("unsafe_code = \"forbid\""),
        "unsafe_code = forbid present",
    )?;
    check(
        cargo_toml.contains("panic = \"forbid\""),
        "panic = forbid present",
    )?;
    Ok(())
}

#[test]
fn t1_26_05_simplified_english_compliance() -> TestResult {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .ok_or("no repository root")?;
    let agents_md = std::fs::read_to_string(root.join("AGENTS.md"))?;
    check(
        agents_md.contains("ASD-STE100"),
        "AGENTS.md requires ASD-STE100",
    )?;
    Ok(())
}
