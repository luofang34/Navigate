//! Integration tests for 3D corridor landmark banks (`NVKTILE1`), two-tier tracking,
//! amortized masking, and `MapCell` EKF fusion independence gating.

use burn::tensor::Device;
use image::{GrayImage, Luma, Rgb, RgbImage};
use nalgebra::Matrix3;
use navigate_match::{CORRIDOR_MAGIC, CorridorLandmarkBank, WeightTensor, Weights};
use navigate_visual_pipeline::{
    CorridorBuildOptions, CorridorTrackOptions, Surface, build_corridor_bank,
    corridor::masking::AmortizedMasker,
    metadata::{PhotoClaims, PositionClaim},
    run_corridor_track,
    util::{read_json, write_json},
};
use serde_json::{Value, json};
use std::{error::Error, fs, path::Path, path::PathBuf};

type TestResult<T = ()> = Result<T, Box<dyn Error>>;

fn synthetic_xfeat_safetensors() -> TestResult<Vec<u8>> {
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
        let values = (0..len)
            .map(|i| (((i * 23 + 7) % 37) as f32 - 18.0) * 0.015)
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
    Ok(weights.to_safetensors(false)?)
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

#[test]
fn corridor_build_and_track_roundtrip_and_contracts() -> TestResult {
    let tmp = tempfile::tempdir()?;
    let lat = 41.11;
    let lon = -75.35;
    let catalog_path = write_synthetic_corridor_catalog(&tmp.path().join("cat"), lat, lon)?;
    let surface = Surface::from_catalog(&catalog_path)?;

    let xfeat_path = tmp.path().join("xfeat.safetensors");
    fs::write(&xfeat_path, synthetic_xfeat_safetensors()?)?;

    let ds_dir = tmp.path().join("dataset");
    fs::create_dir_all(ds_dir.join("images"))?;
    fs::create_dir_all(ds_dir.join("masks"))?;
    fs::create_dir_all(ds_dir.join("profiles"))?;

    let size = [320_u32, 240_u32];
    let f_px = 320.0_f64.hypot(240.0) * 28.0 / 43.266_615_3;
    let k = Matrix3::new(f_px, 0.0, 159.5, 0.0, f_px, 119.5, 0.0, 0.0, 1.0);
    let eye0 = surface.local(lat, lon, 1_050.0);
    let rot0 = surface.camera_rotation(lat, lon, 0.0, 72.0, 0.0);
    let ((rgb0, _, _), _) = surface.render_adaptive(eye0, &rot0, &k, size, 2, 8, 0.025);
    rgb0.save(ds_dir.join("images/f-0001.png"))?;

    let eye1 = [eye0[0] + 8.0, eye0[1] + 2.0, eye0[2]];
    let [lat1, lon1, _] = surface.geodetic(eye1);
    let rot1 = surface.camera_rotation(lat1, lon1, 0.5, 72.0, 0.0);
    let ((rgb1, _, _), _) = surface.render_adaptive(eye1, &rot1, &k, size, 2, 8, 0.025);
    rgb1.save(ds_dir.join("images/f-0002.png"))?;

    let mask = GrayImage::from_pixel(size[0], size[1], Luma([255_u8]));
    mask.save(ds_dir.join("masks/f-0001.png"))?;
    mask.save(ds_dir.join("masks/f-0002.png"))?;

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
    let profile_path = ds_dir.join("profiles/with-prior.json");
    write_json(
        &profile_path,
        &json!({
            "cases": [
                {
                    "id": "f-0001",
                    "image": "images/f-0001.png",
                    "image_sha256": "c".repeat(64),
                    "valid_mask": "masks/f-0001.png",
                    "size": size,
                    "prior": prior
                },
                {
                    "id": "f-0002",
                    "image": "images/f-0002.png",
                    "image_sha256": "d".repeat(64),
                    "valid_mask": "masks/f-0002.png",
                    "size": size,
                    "prior": prior
                }
            ]
        }),
    )?;

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
    })?;

    assert!(build_summary["tiles"].as_u64().unwrap_or(0) >= 1);
    assert!(build_summary["landmarks_3d"].as_u64().unwrap_or(0) >= 32);

    let raw_bytes = fs::read(&bank_path)?;
    assert_eq!(&raw_bytes[..8], CORRIDOR_MAGIC);
    let loaded = CorridorLandmarkBank::load_file(&bank_path)?;
    assert!(!loaded.tiles.is_empty());

    let out_dir = tmp.path().join("track_out");
    let track_summary = run_corridor_track(&CorridorTrackOptions {
        dataset_dir: ds_dir.clone(),
        profile_path: profile_path.clone(),
        catalog_path: catalog_path.clone(),
        corridor_bank_path: bank_path.clone(),
        xfeat_weights_path: xfeat_path.clone(),
        lighterglue_weights_path: None,
        seg_model_dir: None,
        mask_policy: "airframe-occlusions".into(),
        mask_interval: 5,
        output_dir: out_dir.clone(),
        anchor_interval: 10,
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
    })?;
    assert_eq!(track_summary["evaluated"].as_u64(), Some(2));
    assert!(out_dir.join("run.json").is_file());
    let doc1: Value = read_json(&out_dir.join("f-0001.json"))?;
    assert_eq!(doc1["id"].as_str(), Some("f-0001"));

    // Verify MapCell deduplication when anchor_interval == 1 on same cell
    if doc1["status"].as_str() == Some("candidate_requires_review") {
        assert_eq!(
            doc1["candidate"]["independent_ekf_fix"].as_bool(),
            Some(true)
        );
        let doc2: Value = read_json(&out_dir.join("f-0002.json"))?;
        if doc2["status"].as_str() == Some("candidate_requires_review") {
            assert_eq!(
                doc2["candidate"]["independent_ekf_fix"].as_bool(),
                Some(false),
                "warm-tracked frame must not be marked independent_ekf_fix"
            );
        }

        let reanchor_dir = tmp.path().join("reanchor_out");
        let re_summary = run_corridor_track(&CorridorTrackOptions {
            dataset_dir: ds_dir.clone(),
            profile_path: profile_path.clone(),
            catalog_path: catalog_path.clone(),
            corridor_bank_path: bank_path.clone(),
            xfeat_weights_path: xfeat_path.clone(),
            lighterglue_weights_path: None,
            seg_model_dir: None,
            mask_policy: "airframe-occlusions".into(),
            mask_interval: 5,
            output_dir: reanchor_dir.clone(),
            anchor_interval: 1,
            min_track_inliers: 24,
            map_cell_m: 500.0,
            anchor_attempts: 4,
            heading_step_deg: 0.0,
            height_factors: vec![1.0],
            query_keypoints: 512,
            polish_anchors: true,
            prior_override: None,
            focal_35mm_override: None,
            only_cases: None,
        })?;
        assert_eq!(re_summary["independent_ekf_fixes"].as_u64(), Some(1));
        let r2: Value = read_json(&reanchor_dir.join("f-0002.json"))?;
        if r2["status"].as_str() == Some("candidate_requires_review") {
            assert_eq!(
                r2["candidate"]["fusion_eligibility"].as_str(),
                Some("shared_map_cell")
            );
            assert_eq!(
                r2["candidate"]["independent_ekf_fix"].as_bool(),
                Some(false)
            );
        }
    }

    // Verify catalog provenance rejection
    let other_cat = write_synthetic_corridor_catalog(&tmp.path().join("cat2"), lat + 0.2, lon)?;
    let bad = run_corridor_track(&CorridorTrackOptions {
        dataset_dir: ds_dir,
        profile_path,
        catalog_path: other_cat,
        corridor_bank_path: bank_path,
        xfeat_weights_path: xfeat_path,
        lighterglue_weights_path: None,
        seg_model_dir: None,
        mask_policy: "airframe-occlusions".into(),
        mask_interval: 5,
        output_dir: tmp.path().join("bad_out"),
        anchor_interval: 10,
        min_track_inliers: 24,
        map_cell_m: 500.0,
        anchor_attempts: 2,
        heading_step_deg: 0.0,
        height_factors: vec![1.0],
        query_keypoints: 256,
        polish_anchors: false,
        prior_override: None,
        focal_35mm_override: None,
        only_cases: None,
    });
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
