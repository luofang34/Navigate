//! Integration test for the persistent `JSONL` corridor worker (`corridor-serve`).

use image::{Rgb, RgbImage};
use nalgebra::Matrix3;
use navigate_match::{WeightTensor, Weights};
use navigate_visual_pipeline::{
    CorridorBuildOptions, CorridorServeOptions, Surface, XFeatRuntimeConfig, build_corridor_bank,
    corridor::VehicleDynamicsProfile, run_corridor_serve, util::write_json,
};
use serde_json::{Value, json};
use std::{error::Error, fs, io::Cursor, path::Path, path::PathBuf};

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

fn write_synthetic_catalog(root: &Path, lat: f64, lon: f64) -> TestResult<PathBuf> {
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
                *p = Rgb([
                    patch.saturating_add(h1 / 3),
                    patch.wrapping_add(h1 / 2),
                    patch / 2,
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
fn corridor_serve_reuses_resident_session_across_single_image_requests() -> TestResult {
    let tmp = tempfile::tempdir()?;
    let (lat, lon) = (41.11, -75.35);
    let catalog_path = write_synthetic_catalog(&tmp.path().join("cat"), lat, lon)?;
    let surface = Surface::from_catalog(&catalog_path)?;
    let xfeat_path = tmp.path().join("xfeat.safetensors");
    fs::write(
        &xfeat_path,
        synthetic_xfeat_weights().to_safetensors(false)?,
    )?;

    let size = [320_u32, 240_u32];
    let f_px = 320.0_f64.hypot(240.0) * 28.0 / 43.266_615_3;
    let k = Matrix3::new(f_px, 0.0, 159.5, 0.0, f_px, 119.5, 0.0, 0.0, 1.0);
    let eye = surface.local(lat, lon, 1_050.0);
    let rot = surface.camera_rotation(lat, lon, 0.0, 88.0, 0.0);
    let ((rgb, _, _), _) = surface.render_adaptive(eye, &rot, &k, size, 2, 8, 0.025);
    let query_img = tmp.path().join("query_frame.png");
    rgb.save(&query_img)?;
    let profile_path = tmp.path().join("profile.json");
    write_json(
        &profile_path,
        &json!({
            "cases": [{
                "id": "query_frame",
                "image": "query_frame.png",
                "image_sha256": "c".repeat(64),
                "size": size,
                "prior": {
                    "direction_deg": 0.0,
                    "focal_35mm_mm": 28.0,
                    "position": {
                        "latitude": lat,
                        "longitude": lon,
                        "altitude_m": 1050.0,
                        "unsigned_altitude_m": 1050.0,
                        "altitude_datum": "unverified"
                    }
                }
            }]
        }),
    )?;

    let bank_path = tmp.path().join("corridor.nvktile");
    build_corridor_bank(&CorridorBuildOptions {
        catalog_path: catalog_path.clone(),
        profile_path: Some(profile_path),
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

    let serve_opts = CorridorServeOptions {
        catalog_path,
        corridor_bank_path: bank_path,
        xfeat_weights_path: xfeat_path,
        lighterglue_weights_path: None,
        seg_model_dir: None,
        dinov2_weights_path: None,
        place_index_path: None,
        mask_policy: "airframe-occlusions".into(),
        dynamics_profile: VehicleDynamicsProfile::default(),
        query_keypoints: 384,
        polish_anchors: false,
        xfeat_runtime: XFeatRuntimeConfig::default(),
    };

    let req1 = json!({
        "op": "ping",
        "request_id": "ping-1"
    });
    let req2 = json!({
        "request_id": "loc-1",
        "image": query_img,
        "prior_lat": lat,
        "prior_lon": lon,
        "prior_alt": 1050.0,
        "prior_heading": 0.0,
        "focal_35mm": 28.0,
        "attempts": 2
    });
    let req3 = json!({
        "op": "shutdown",
        "request_id": "bye-1"
    });
    let input = format!("{req1}\n{req2}\n{req3}\n");
    let mut output = Vec::new();
    let summary = run_corridor_serve(&serve_opts, Cursor::new(input.into_bytes()), &mut output)?;
    assert_eq!(summary["requests_served"].as_u64(), Some(1));

    let lines: Vec<Value> = String::from_utf8(output)?
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    assert_eq!(lines.len(), 4);
    assert_eq!(lines[0]["status"].as_str(), Some("ready"));
    assert_eq!(lines[1]["status"].as_str(), Some("pong"));
    assert_eq!(lines[1]["request_id"].as_str(), Some("ping-1"));
    assert_eq!(lines[2]["ok"].as_bool(), Some(true));
    assert_eq!(lines[2]["request_id"].as_str(), Some("loc-1"));
    assert_eq!(lines[2]["summary"]["evaluated"].as_u64(), Some(1));
    assert_eq!(
        lines[2]["summary"]["shader_warmup_ms"].as_f64(),
        Some(0.0),
        "resident worker request must skip cold shader compilation"
    );
    assert_eq!(lines[2]["case_report"]["id"].as_str(), Some("query_frame"));
    assert_eq!(lines[3]["status"].as_str(), Some("shutdown"));
    Ok(())
}
