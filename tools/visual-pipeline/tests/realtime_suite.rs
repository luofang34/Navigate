//! Integration tests for offline Kashin bank construction and real-time bank matching.

use image::{GrayImage, Luma, Rgb, RgbImage};
use navigate_match::{PrecomputedReferenceBank, WeightTensor, Weights, kashin::BANK_MAGIC};
use navigate_visual_pipeline::{
    BankBuildOptions, BankMatchOptions, build_reference_bank,
    metadata::{PhotoClaims, PositionClaim},
    run_realtime_bank_match,
    util::write_json,
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
            .map(|i| (((i * 17 + 3) % 29) as f32 - 14.0) * 0.01)
            .collect();
        weights.insert(
            format!("{name}.weight"),
            WeightTensor {
                shape: vec![out, input, k, k],
                values,
            },
        );
        let mut bias = vec![0.01; out];
        if name == "keypoint_head.3" {
            bias.fill(0.0);
            bias[27] = 4.0;
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
    let n = f64::from(1u32 << zoom);
    let x = ((lon_deg + 180.0) / 360.0) * n;
    let y = (1.0 - lat_deg.to_radians().tan().asinh() / std::f64::consts::PI) * 0.5 * n;
    (x.floor() as u32, y.floor() as u32)
}

fn write_synthetic_catalog(root: &Path, lat: f64, lon: f64) -> TestResult<PathBuf> {
    fs::create_dir_all(root.join("tiles"))?;
    let (ix, iy) = mercator_xy(lat, lon, 14);
    let (dx, dy) = mercator_xy(lat, lon, 12);

    let mut dem = RgbImage::new(256, 256);
    for p in dem.pixels_mut() {
        *p = Rgb([128, 180, 0]);
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

    let mut img = RgbImage::new(256, 256);
    for (x, y, p) in img.enumerate_pixels_mut() {
        let field = (((x / 32) * 37 + (y / 32) * 53) & 0x7f) as u8 + 60;
        let road = if x % 64 < 2 || y % 64 < 2 { 90u8 } else { 0u8 };
        *p = Rgb([
            field.saturating_add(road),
            field.wrapping_add(20),
            field / 2,
        ]);
    }
    for oy in -1..=1_i32 {
        for ox in -1..=1_i32 {
            let tx = (ix as i32 + ox) as u32;
            let ty = (iy as i32 + oy) as u32;
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
                "bbox": [lon - 0.05, lat - 0.05, lon + 0.05, lat + 0.05],
                "imagery_zoom": 14,
                "dem_zoom": 12
            },
            "tiles": tiles
        }),
    )?;
    Ok(catalog_path)
}

#[test]
fn bank_build_and_realtime_bank_match_roundtrip_and_contracts() -> TestResult {
    let tmp = tempfile::tempdir()?;
    let lat = 41.11;
    let lon = -75.35;
    let catalog_path = write_synthetic_catalog(&tmp.path().join("cat"), lat, lon)?;

    let xfeat_path = tmp.path().join("xfeat.safetensors");
    fs::write(&xfeat_path, synthetic_xfeat_safetensors()?)?;

    let ds_dir = tmp.path().join("dataset");
    fs::create_dir_all(ds_dir.join("images"))?;
    fs::create_dir_all(ds_dir.join("masks"))?;
    fs::create_dir_all(ds_dir.join("profiles"))?;

    let mut q_img = RgbImage::new(320, 240);
    for (x, y, p) in q_img.enumerate_pixels_mut() {
        let v = (((x * 11 + y * 17) ^ (x * y)) & 0xff) as u8;
        *p = Rgb([v, v.wrapping_add(37), v.wrapping_mul(3)]);
    }
    q_img.save(ds_dir.join("images/q-test.png"))?;

    let mut mask = GrayImage::from_pixel(320, 240, Luma([255u8]));
    for y in 0..48 {
        for x in 0..320 {
            mask.put_pixel(x, y, Luma([0u8]));
        }
    }
    mask.save(ds_dir.join("masks/q-test.png"))?;

    let prior = PhotoClaims {
        direction_deg: Some(90.0),
        focal_35mm_mm: Some(28.0),
        position: Some(PositionClaim {
            latitude: lat,
            longitude: lon,
            reported_horizontal_error_m: Some(25.0),
            altitude_m: Some(950.0),
            unsigned_altitude_m: Some(950.0),
            altitude_reference_text: None,
            altitude_datum: "unverified".into(),
        }),
        ..PhotoClaims::default()
    };
    let profile_path = ds_dir.join("profiles/with-prior.json");
    write_json(
        &profile_path,
        &json!({
            "cases": [{
                "id": "q-test",
                "image": "images/q-test.png",
                "image_sha256": "c".repeat(64),
                "valid_mask": "masks/q-test.png",
                "semantic_labels": null,
                "size": [320, 240],
                "prior": prior
            }]
        }),
    )?;

    let bank_path = tmp.path().join("ref.kashin");
    let build_summary = build_reference_bank(&BankBuildOptions {
        dataset_dir: ds_dir.clone(),
        profile_path: profile_path.clone(),
        catalog_path: catalog_path.clone(),
        xfeat_weights_path: xfeat_path.clone(),
        output_bank_path: bank_path.clone(),
        attempts: 1,
        heading_step_deg: 0.0,
        height_factors: vec![1.0],
        max_keypoints: 128,
        adaptive_stride: 4,
        only_cases: None,
    })?;

    assert_eq!(build_summary["views"].as_u64(), Some(1));
    let reduction = build_summary["adaptive_ray_reduction"]
        .as_f64()
        .ok_or("missing adaptive_ray_reduction")?;
    assert!(
        reduction > 1.5,
        "expected adaptive ray reduction > 1.5x, got {reduction}"
    );

    let bank_bytes = fs::read(&bank_path)?;
    assert_eq!(&bank_bytes[..8], BANK_MAGIC);
    let loaded = PrecomputedReferenceBank::load_file(&bank_path)?;
    assert_eq!(loaded.entries.len(), 1);
    assert!(
        loaded
            .entries
            .iter()
            .all(|e| e.world_points_enu.len() == e.pixels.len())
    );

    let out_dir = tmp.path().join("run_out");
    let match_summary = run_realtime_bank_match(&BankMatchOptions {
        dataset_dir: ds_dir.clone(),
        profile_path: profile_path.clone(),
        catalog_path: catalog_path.clone(),
        bank_path: bank_path.clone(),
        xfeat_weights_path: xfeat_path.clone(),
        lighterglue_weights_path: None,
        output_dir: out_dir.clone(),
        shortlist_k: 2,
        query_keypoints: 128,
        polish: false,
        only_cases: None,
    })?;
    assert_eq!(match_summary["evaluated"].as_u64(), Some(1));
    assert_eq!(match_summary["bank_views"].as_u64(), Some(1));
    assert!(out_dir.join("run.json").is_file());
    let case_doc: Value = navigate_visual_pipeline::util::read_json(&out_dir.join("q-test.json"))?;
    assert_eq!(case_doc["id"].as_str(), Some("q-test"));

    let other_cat = write_synthetic_catalog(&tmp.path().join("cat2"), lat + 0.1, lon)?;
    let err = run_realtime_bank_match(&BankMatchOptions {
        dataset_dir: ds_dir,
        profile_path,
        catalog_path: other_cat,
        bank_path,
        xfeat_weights_path: xfeat_path,
        lighterglue_weights_path: None,
        output_dir: tmp.path().join("run_bad"),
        shortlist_k: 2,
        query_keypoints: 128,
        polish: false,
        only_cases: None,
    });
    assert!(err.is_err(), "mismatched catalog SHA-256 must fail");
    Ok(())
}
