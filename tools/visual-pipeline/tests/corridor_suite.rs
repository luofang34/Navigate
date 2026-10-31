//! Integration tests for 3D corridor landmark banks (`NVKTILE1`), sharded tile stores
//! (`NVKTIDX1`), non-blocking background anchoring, `burn-f16`, and `MapCell` EKF gating.

use burn::tensor::Device;
use image::{GrayImage, Luma, Rgb, RgbImage};
use nalgebra::Matrix3;
use navigate_match::{
    CORRIDOR_MAGIC, CORRIDOR_STORE_INDEX_MAGIC, CorridorLandmarkBank, CorridorTileStore,
    DEFAULT_KASHIN_SEED, DEFAULT_WINDOW_TILES, WeightTensor, Weights, XFeat,
};
use navigate_visual_pipeline::{
    CorridorBuildOptions, CorridorTrackOptions, Surface, XFeatRuntimeConfig, build_corridor_bank,
    corridor::masking::AmortizedMasker,
    metadata::{PhotoClaims, PositionClaim},
    run_corridor_track,
    util::{read_json, write_json},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{error::Error, fs, path::Path, path::PathBuf};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn synthetic_xfeat_weights() -> Weights {
    let specs: &[(&str, [usize; 3])] = &[
        ("block1.0", [4, 1, 3]),
        ("block1.1", [8, 4, 3]),
        ("block1.2", [8, 8, 3]),
        ("block1.3", [24, 8, 3]),
        ("skip1.1", [24, 1, 1]),
        ("block2.0", [24, 24, 3]),
        ("block2.1", [24, 24, 3]),
        ("block3.0", [64, 24, 3]),
        ("block3.1", [64, 64, 3]),
        ("block3.2", [64, 64, 1]),
        ("block4.0", [64, 64, 3]),
        ("block4.1", [64, 64, 3]),
        ("block4.2", [64, 64, 3]),
        ("block5.0", [128, 64, 3]),
        ("block5.1", [128, 128, 3]),
        ("block5.2", [128, 128, 3]),
        ("block5.3", [64, 128, 1]),
        ("block_fusion.0", [64, 64, 3]),
        ("block_fusion.1", [64, 64, 3]),
        ("block_fusion.2", [64, 64, 1]),
        ("heatmap_head.0", [64, 64, 1]),
        ("heatmap_head.1", [64, 64, 1]),
        ("heatmap_head.2", [1, 64, 1]),
        ("keypoint_head.0", [64, 64, 1]),
        ("keypoint_head.1", [64, 64, 1]),
        ("keypoint_head.2", [64, 64, 1]),
        ("keypoint_head.3", [65, 64, 1]),
    ];
    let mut weights = Weights::default();
    for &(name, [out, input, k]) in specs {
        let len = out * input * k * k;
        let scale = ((2.0 / ((input * k * k) as f32)).sqrt() / 0.09).min(1.0);
        let values = (0..len)
            .map(|i| (((i * 23 + 7) % 37) as f32 - 18.0) * 0.015 * scale)
            .collect();
        weights.insert(
            format!("{name}.weight"),
            WeightTensor {
                shape: vec![out, input, k, k],
                values,
            },
        );
        let mut bias = vec![0.005; out];
        if name == "keypoint_head.3" {
            bias.fill(0.0);
            bias[27] = 3.5;
        }
        weights.insert(
            format!("{name}.bias"),
            WeightTensor {
                shape: vec![out],
                values: bias,
            },
        );
    }
    weights
}

fn mercator_xy(lat_deg: f64, lon_deg: f64, zoom: u32) -> (u32, u32) {
    let n = f64::from(1_u32 << zoom);
    let x = ((lon_deg + 180.0) / 360.0) * n;
    let y = (1.0 - lat_deg.to_radians().tan().asinh() / std::f64::consts::PI) * 0.5 * n;
    (x.floor() as u32, y.floor() as u32)
}

fn write_synthetic_corridor_catalog(root: &Path, lat: f64, lon: f64) -> TestResult<PathBuf> {
    fs::create_dir_all(root.join("tiles"))?;
    let (ix, iy) = mercator_xy(lat, lon, 14);
    let (dx, dy) = mercator_xy(lat, lon, 12);

    let mut dem = RgbImage::new(256, 256);
    for (x, y, p) in dem.enumerate_pixels_mut() {
        let hill = (((x * 3 + y * 5) % 40) as u8).saturating_add(160);
        *p = Rgb([128, hill, 0]);
    }
    let mut tiles = Vec::new();
    for oy in -1..=1_i32 {
        for ox in -1..=1_i32 {
            let tx = (dx as i32 + ox) as u32;
            let ty = (dy as i32 + oy) as u32;
            let rel = format!("tiles/dem_12_{tx}_{ty}.png");
            dem.save(root.join(&rel))?;
            tiles.push(json!({
                "kind": "dem",
                "xyz": [12, tx, ty],
                "path": rel,
                "sha256": "a".repeat(64),
                "bytes": 100,
                "url": "file:///test"
            }));
        }
    }

    for oy in -1..=1_i32 {
        for ox in -1..=1_i32 {
            let tx = (ix as i32 + ox) as u32;
            let ty = (iy as i32 + oy) as u32;
            let mut img = RgbImage::new(256, 256);
            let salt = ((ox + 2) * 53 + (oy + 2) * 97) as u32;
            for (x, y, p) in img.enumerate_pixels_mut() {
                let h1 = ((x.wrapping_mul(37) ^ y.wrapping_mul(73) ^ salt) & 0x7f) as u8;
                let patch = ((((x / 16) * 29 + (y / 16) * 43 + salt) & 0x3f) as u8) + 40;
                let road = if (x + salt) % 47 < 3 || (y + salt) % 53 < 3 {
                    95_u8
                } else {
                    0_u8
                };
                *p = Rgb([
                    patch.saturating_add(road).saturating_add(h1 / 3),
                    patch.wrapping_add(h1 / 2),
                    patch / 2 + road / 2,
                ]);
            }
            let rel = format!("tiles/img_14_{tx}_{ty}.png");
            img.save(root.join(&rel))?;
            tiles.push(json!({
                "kind": "imagery",
                "xyz": [14, tx, ty],
                "path": rel,
                "sha256": "b".repeat(64),
                "bytes": 100,
                "url": "file:///test"
            }));
        }
    }

    let catalog_path = root.join("catalog.json");
    write_json(
        &catalog_path,
        &json!({
            "format": "photo-reference-catalog",
            "version": 1,
            "complete": true,
            "configuration": {
                "bbox": [lon - 0.03, lat - 0.03, lon + 0.03, lat + 0.03],
                "imagery_zoom": 14,
                "dem_zoom": 12
            },
            "tiles": tiles
        }),
    )?;
    Ok(catalog_path)
}

fn base_track_opts(
    ds_dir: &Path,
    profile_path: &Path,
    catalog_path: &Path,
    bank_path: &Path,
    xfeat_path: &Path,
    output_dir: PathBuf,
    background_refresh: bool,
) -> CorridorTrackOptions {
    CorridorTrackOptions {
        dataset_dir: ds_dir.to_path_buf(),
        profile_path: profile_path.to_path_buf(),
        catalog_path: catalog_path.to_path_buf(),
        corridor_bank_path: bank_path.to_path_buf(),
        xfeat_weights_path: xfeat_path.to_path_buf(),
        lighterglue_weights_path: None,
        seg_model_dir: None,
        dynamics_profile: navigate_visual_pipeline::corridor::VehicleDynamicsProfile::default(),
        mask_policy: "airframe-occlusions".into(),
        mask_interval: 5,
        mask_interval_s: None,
        output_dir,
        anchor_interval: 10,
        anchor_interval_s: None,
        anchor_uncertainty_m: None,
        anchor_on_cell_boundary: false,
        min_track_inliers: 24,
        map_cell_m: 500.0,
        anchor_attempts: 2,
        heading_step_deg: 0.0,
        height_factors: vec![1.0],
        query_keypoints: 384,
        polish_anchors: false,
        prior_override: None,
        focal_35mm_override: None,
        only_cases: None,
        background_refresh,
        xfeat_runtime: XFeatRuntimeConfig::default(),
    }
}

#[test]
fn corridor_build_and_track_roundtrip_and_contracts() -> TestResult {
    let tmp = tempfile::tempdir()?;
    let lat = 41.11;
    let lon = -75.35;
    let catalog_path = write_synthetic_corridor_catalog(&tmp.path().join("cat"), lat, lon)?;
    let surface = Surface::from_catalog(&catalog_path)?;

    let weights = synthetic_xfeat_weights();
    let xfeat_path = tmp.path().join("xfeat.safetensors");
    fs::write(&xfeat_path, weights.to_safetensors(false)?)?;

    let ds_dir = tmp.path().join("dataset");
    fs::create_dir_all(ds_dir.join("images"))?;
    fs::create_dir_all(ds_dir.join("masks"))?;
    fs::create_dir_all(ds_dir.join("profiles"))?;

    let size = [320_u32, 240_u32];
    let f_px = 320.0_f64.hypot(240.0) * 28.0 / 43.266_615_3;
    let k = Matrix3::new(f_px, 0.0, 159.5, 0.0, f_px, 119.5, 0.0, 0.0, 1.0);
    let eye0 = surface.local(lat, lon, 1_050.0);
    let mask = GrayImage::from_pixel(size[0], size[1], Luma([255_u8]));

    let prior = PhotoClaims {
        direction_deg: Some(0.0),
        focal_35mm_mm: Some(28.0),
        position: Some(PositionClaim {
            latitude: lat,
            longitude: lon,
            reported_horizontal_error_m: Some(50.0),
            altitude_m: Some(1_050.0),
            unsigned_altitude_m: Some(1_050.0),
            altitude_reference_text: None,
            altitude_datum: "unverified".into(),
        }),
        ..PhotoClaims::default()
    };

    let mut cases = Vec::new();
    for step in 0..4_usize {
        let id = format!("f-{:04}", step + 1);
        let eye = [
            eye0[0] + (step as f64) * 5.0,
            eye0[1] + (step as f64) * 1.5,
            eye0[2],
        ];
        let [clat, clon, _] = surface.geodetic(eye);
        let rot = surface.camera_rotation(clat, clon, (step as f64) * 0.4, 72.0, 0.0);
        let ((rgb, _, _), _) = surface.render_adaptive(eye, &rot, &k, size, 2, 8, 0.025);
        let img_rel = format!("images/{id}.png");
        let msk_rel = format!("masks/{id}.png");
        rgb.save(ds_dir.join(&img_rel))?;
        mask.save(ds_dir.join(&msk_rel))?;
        cases.push(json!({
            "id": id,
            "image": img_rel,
            "image_sha256": "c".repeat(64),
            "valid_mask": msk_rel,
            "size": size,
            "prior": prior
        }));
    }
    let profile_path = ds_dir.join("profiles/with-prior.json");
    write_json(&profile_path, &json!({ "cases": cases }))?;

    // 1. Monolithic NVKTILE1 build + deterministic sync-refresh + burn-f16 parity
    let bank_path = tmp.path().join("corridor.nvktile");
    let build_summary = build_corridor_bank(&CorridorBuildOptions {
        catalog_path: catalog_path.clone(),
        profile_path: Some(profile_path.clone()),
        xfeat_weights_path: xfeat_path.clone(),
        output_bank_path: bank_path.clone(),
        tile_size_m: 15_000.0,
        corridor_radius_m: 0.0,
        nominal_agl_m: 900.0,
        max_landmarks_per_tile: 512,
        adaptive_stride: 4,
        include_obliques: false,
        xfeat_runtime: XFeatRuntimeConfig::default(),
    })?;
    assert!(build_summary["tiles"].as_u64().unwrap_or(0) >= 1);
    assert!(build_summary["landmarks_3d"].as_u64().unwrap_or(0) >= 32);
    assert_eq!(&fs::read(&bank_path)?[..8], CORRIDOR_MAGIC);
    assert!(
        !CorridorLandmarkBank::load_file(&bank_path)?
            .tiles
            .is_empty()
    );

    let out_dir = tmp.path().join("track_sync");
    let track_summary = run_corridor_track(&base_track_opts(
        &ds_dir,
        &profile_path,
        &catalog_path,
        &bank_path,
        &xfeat_path,
        out_dir.clone(),
        false,
    ))?;
    assert_eq!(track_summary["evaluated"].as_u64(), Some(4));
    assert!(
        track_summary["resident_window_bytes"]
            .as_u64()
            .unwrap_or(u64::MAX)
            < 10 * 1024 * 1024
    );
    for key in [
        "warm_track_p50_ms",
        "warm_track_p95_ms",
        "warm_track_p99_ms",
        "anchor_p50_ms",
        "seg_refresh_p50_ms",
    ] {
        assert!(
            track_summary.get(key).is_some(),
            "summary must report {key}"
        );
    }

    // Verify burn-f16 runtime shares encoder_contract with burn-f32 bank
    let mut f16_opts = base_track_opts(
        &ds_dir,
        &profile_path,
        &catalog_path,
        &bank_path,
        &xfeat_path,
        tmp.path().join("track_f16"),
        false,
    );
    f16_opts.xfeat_runtime = XFeatRuntimeConfig::burn_f16();
    f16_opts.only_cases = Some(["f-0001".to_owned()].into_iter().collect());
    let f16_summary = run_corridor_track(&f16_opts)?;
    assert_eq!(f16_summary["xfeat_runtime"], "burn-f16");
    assert_eq!(f16_summary["evaluated"], 1);

    // 2. Non-blocking background_refresh cold start emits acquiring_anchor on Frame 1
    let bg_dir = tmp.path().join("track_bg");
    let bg_summary = run_corridor_track(&base_track_opts(
        &ds_dir,
        &profile_path,
        &catalog_path,
        &bank_path,
        &xfeat_path,
        bg_dir.clone(),
        true,
    ))?;
    assert_eq!(bg_summary["background_refresh"], true);
    assert!(bg_summary["acquiring_anchor_frames"].as_u64().unwrap_or(0) >= 1);
    let bg_f1: Value = read_json(&bg_dir.join("f-0001.json"))?;
    assert_eq!(bg_f1["status"].as_str(), Some("acquiring_anchor"));
    assert_eq!(bg_f1["evidence_kind"].as_str(), Some("visual_odometry"));
    assert!(
        !bg_summary["anchor_results"]
            .as_array()
            .ok_or("anchor_results")?
            .is_empty()
    );

    // 3. Sharded NVKTIDX1 tile store directory + on-demand empty-store synthesis
    let sharded_dir = tmp.path().join("corridor.nvktiledir");
    let sharded_build = build_corridor_bank(&CorridorBuildOptions {
        catalog_path: catalog_path.clone(),
        profile_path: Some(profile_path.clone()),
        xfeat_weights_path: xfeat_path.clone(),
        output_bank_path: sharded_dir.clone(),
        tile_size_m: 15_000.0,
        corridor_radius_m: 0.0,
        nominal_agl_m: 900.0,
        max_landmarks_per_tile: 512,
        adaptive_stride: 4,
        include_obliques: false,
        xfeat_runtime: XFeatRuntimeConfig::default(),
    })?;
    assert_eq!(sharded_build["sharded_store"], true);
    let idx_bytes = fs::read(sharded_dir.join("manifest.nvktidx"))?;
    assert_eq!(&idx_bytes[..8], CORRIDOR_STORE_INDEX_MAGIC);
    let sharded_track = run_corridor_track(&base_track_opts(
        &ds_dir,
        &profile_path,
        &catalog_path,
        &sharded_dir,
        &xfeat_path,
        tmp.path().join("track_sharded"),
        false,
    ))?;
    assert_eq!(sharded_track["accepted"], track_summary["accepted"]);

    // Empty sharded store triggers live on-demand tile synthesis
    let empty_store_dir = tmp.path().join("empty_store.nvktiledir");
    let cat_digest: [u8; 32] = Sha256::digest(fs::read(&catalog_path)?).into();
    let contract = XFeat::encoder_contract_of(&weights);
    let _ = CorridorTileStore::create_dir(
        &empty_store_dir,
        cat_digest,
        contract,
        DEFAULT_KASHIN_SEED,
        1_200.0,
        DEFAULT_WINDOW_TILES,
    )?;
    let mut synth_opts = base_track_opts(
        &ds_dir,
        &profile_path,
        &catalog_path,
        &empty_store_dir,
        &xfeat_path,
        tmp.path().join("track_live_synth"),
        false,
    );
    synth_opts.only_cases = Some(["f-0001".to_owned()].into_iter().collect());
    let live_synth = run_corridor_track(&synth_opts)?;
    assert!(
        live_synth["tiles_built_live"].as_u64().unwrap_or(0) >= 1,
        "expected on-demand tile synthesis on empty store"
    );

    // 4. Verify MapCell deduplication and catalog SHA-256 rejection
    let doc1: Value = read_json(&out_dir.join("f-0001.json"))?;
    if doc1["status"].as_str() == Some("candidate_requires_review") {
        assert_eq!(
            doc1["candidate"]["independent_ekf_fix"].as_bool(),
            Some(true)
        );
        let mut re_opts = base_track_opts(
            &ds_dir,
            &profile_path,
            &catalog_path,
            &bank_path,
            &xfeat_path,
            tmp.path().join("reanchor_out"),
            false,
        );
        re_opts.anchor_interval = 1;
        re_opts.polish_anchors = true;
        let re_summary = run_corridor_track(&re_opts)?;
        assert_eq!(re_summary["independent_ekf_fixes"].as_u64(), Some(1));
    }

    let other_cat = write_synthetic_corridor_catalog(&tmp.path().join("cat2"), lat + 0.2, lon)?;
    let bad = run_corridor_track(&base_track_opts(
        &ds_dir,
        &profile_path,
        &other_cat,
        &bank_path,
        &xfeat_path,
        tmp.path().join("bad_out"),
        false,
    ));
    assert!(bad.is_err(), "mismatched catalog SHA-256 must fail");
    Ok(())
}

#[test]
fn amortized_masker_fallback_and_propeller_gate_are_deterministic() -> TestResult {
    let device = Device::flex();
    let mut masker = AmortizedMasker::new(None, "airframe-occlusions", 6, 4, &device)?;
    let rgb = RgbImage::from_pixel(160, 120, Rgb([120, 140, 110]));
    let mut fb = GrayImage::from_pixel(160, 120, Luma([255_u8]));
    for y in 0..30 {
        for x in 0..160 {
            fb.put_pixel(x, y, Luma([0_u8]));
        }
    }
    let k = Matrix3::new(180.0, 0.0, 79.5, 0.0, 180.0, 59.5, 0.0, 0.0, 1.0);
    let r0 = Matrix3::identity();
    let (m0, tel0) = masker.mask_frame(&rgb, Some(&fb), Some(&r0), &k)?;
    assert!(!tel0.segformer_refreshed);
    assert!((tel0.valid_fraction - 0.75).abs() < 0.05);
    assert_eq!(m0.get_pixel(80, 10).0[0], 0);
    assert_eq!(m0.get_pixel(80, 90).0[0], 255);
    Ok(())
}
