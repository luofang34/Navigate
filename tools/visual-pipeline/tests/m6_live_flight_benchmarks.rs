//! End-to-end live flight video benchmarks for `build_corridor_bank` and `run_corridor_track`.
//!
//! Each live flight test is marked `#[ignore]` because it requires local flight video clips,
//! `.safetensors` neural weights (`XFeat`, `LighterGlue`, `Segformer`), and terrain catalogs.

#[path = "m6_live_flight_benchmarks/harness.rs"]
mod harness;

use harness::{
    ClipSpec, TestResult, execute_real_corridor_clip, prior_diag_cov, resolve_clip_from,
};
use image::{DynamicImage, Rgb, RgbImage};
use nalgebra::{UnitQuaternion, Vector3};
use navigate_place::Record;
use navigate_visual::{CameraPose, SearchPrior, SearchTier};
use navigate_visual_pipeline::{
    XFeatRuntime, XFeatRuntimeConfig,
    corridor::{
        VehicleDynamicsProfile,
        anchor::multitier::{MultiTierSearchConfig, SearchTierMatcher},
    },
};
use std::{fs, path::PathBuf};

#[test]
fn flight_clip_resolution_honors_explicit_dji_0030_and_clip_dir_overrides() -> TestResult {
    let tmp = tempfile::tempdir()?;
    let explicit_dji = tmp.path().join("custom_dji_0030.mp4");
    fs::write(&explicit_dji, b"placeholder")?;
    assert_eq!(
        resolve_clip_from("DJI_0030.MP4", Some(&explicit_dji), None, None)?,
        explicit_dji
    );

    let clip_dir = tmp.path().join("clips");
    fs::create_dir_all(&clip_dir)?;
    let dir_dji = clip_dir.join("DJI_0030.MP4");
    fs::write(&dir_dji, b"placeholder")?;
    assert_eq!(
        resolve_clip_from("DJI_0030.MP4", None, Some(&clip_dir), None)?,
        dir_dji
    );
    assert!(resolve_clip_from("DJI_0030.MP4", None, Some(tmp.path()), None).is_err());
    Ok(())
}

#[test]
fn unconfigured_search_tier_matcher_fails_closed_without_dummy_descriptors() -> TestResult {
    let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(640, 480, Rgb([96, 128, 84])));
    let records = vec![Record {
        centre: [40.9299, -75.1294],
        descriptor: vec![1.0_f32; 384],
    }];
    let matcher = SearchTierMatcher::new(MultiTierSearchConfig::default()).with_records(records);
    let prior = SearchPrior {
        center: CameraPose {
            position: Vector3::new(0.0, 0.0, 1000.0),
            orientation: UnitQuaternion::identity(),
        },
        position_covariance_m2: prior_diag_cov(),
        attitude_sigma_rad: 0.2,
    };
    let res = matcher.execute_with_context(&prior, Some(&img), None, None)?;
    assert_eq!(res.tier, SearchTier::Global);
    assert!(res.query_descriptor.is_none());
    assert!(res.ranked_regions.is_empty());
    Ok(())
}

#[test]
#[ignore = "requires local IMG_4047.MOV, terrain catalog, and .safetensors weights"]
fn live_flight_img_4047_corridor_track() -> TestResult {
    let spec = ClipSpec::aircraft(
        "IMG_4047.MOV",
        "NAVIGATE_IMG_4047_PATH",
        "NAVIGATE_IMG_4047_CATALOG",
        &[
            "reference-video/v4047/catalog.json",
            "reference-region/catalog.json",
        ],
        [40.3273, -75.4756, 1120.859],
    );
    let r = execute_real_corridor_clip(&spec)?;
    assert!(
        r.build_elapsed_ms > 0.0
            && r.warm_tracked_frames >= 1
            && r.p50_warm_ms > 0.0
            && r.p50_warm_ms < 33.3
            && r.max_inliers >= 24
    );
    Ok(())
}

#[test]
#[ignore = "requires local IMG_4055.MOV, terrain catalog, and .safetensors weights"]
fn live_flight_img_4055_corridor_track() -> TestResult {
    let spec = ClipSpec::aircraft(
        "IMG_4055.MOV",
        "NAVIGATE_IMG_4055_PATH",
        "NAVIGATE_IMG_4055_CATALOG",
        &[
            "reference-video/v4055/catalog.json",
            "reference-region/catalog.json",
        ],
        [40.4201, -75.1011, 1105.271],
    );
    let r = execute_real_corridor_clip(&spec)?;
    assert!(
        r.build_elapsed_ms > 0.0
            && r.warm_tracked_frames >= 1
            && r.p50_warm_ms > 0.0
            && r.p50_warm_ms < 33.3
            && r.max_inliers >= 24
    );
    Ok(())
}

#[test]
#[ignore = "requires local IMG_5730.MOV, terrain catalog, and .safetensors weights"]
fn live_flight_img_5730_corridor_track() -> TestResult {
    let mut spec = ClipSpec::aircraft(
        "IMG_5730.MOV",
        "NAVIGATE_IMG_5730_PATH",
        "NAVIGATE_IMG_5730_CATALOG",
        &[
            "reference-video/v5730/catalog.json",
            "reference-video/v5730-gap/catalog.json",
        ],
        [40.8646, -75.0246, 1359.802],
    );
    spec.nominal_agl_m = 950.0;
    spec.dynamics = VehicleDynamicsProfile::high_performance();
    spec.max_pos_offset_m = 5000.0;
    let r = execute_real_corridor_clip(&spec)?;
    assert!(
        r.build_elapsed_ms > 0.0
            && r.warm_tracked_frames >= 1
            && r.p50_warm_ms > 0.0
            && r.p50_warm_ms < 33.3
            && r.max_inliers >= 24
    );
    Ok(())
}

#[test]
#[ignore = "requires local IMG_6441.MOV, Water Gap catalog, and .safetensors weights"]
fn live_flight_img_6441_high_agl_occlusion_corridor_track() -> TestResult {
    let mut spec = ClipSpec::aircraft(
        "IMG_6441.MOV",
        "NAVIGATE_IMG_6441_PATH",
        "NAVIGATE_IMG_6441_CATALOG",
        &[
            "reference-watergap/catalog.json",
            "reference-details/watergap/catalog.json",
        ],
        [40.9299, -75.1294, 1224.892],
    );
    spec.duration_s = 1.75;
    spec.fps = 10.0;
    spec.tile_size_m = 2000.0;
    spec.corridor_radius_m = 9500.0;
    spec.nominal_agl_m = 950.0;
    spec.min_accepted = 2;
    spec.max_pos_offset_m = 8000.0;
    let r = execute_real_corridor_clip(&spec)?;
    assert!((0.05..=0.75).contains(&r.first_valid_mask_fraction));
    assert!(r.locked_focal_px.is_some());
    assert!(r.warm_tracked_frames >= 1 && r.p50_warm_ms > 0.0 && r.p50_warm_ms < 33.3);
    Ok(())
}

#[test]
#[ignore = "requires local DJI_0029.MP4, local high-zoom catalog, and .safetensors weights"]
fn live_flight_dji_0029_nadir_survey_corridor_track() -> TestResult {
    let spec = ClipSpec::nadir_drone(
        "DJI_0029.MP4",
        "NAVIGATE_DJI_0029_PATH",
        [40.5442, -74.4564],
    );
    let r = execute_real_corridor_clip(&spec)?;
    assert!(
        r.frames >= 4
            && r.warm_tracked_frames >= 1
            && r.p50_warm_ms > 0.0
            && r.p50_warm_ms < 33.3
            && r.max_inliers >= 24
            && r.best_filter_quality != "none"
    );
    Ok(())
}

#[test]
#[ignore = "requires local DJI_0030.MP4, local high-zoom catalog, and .safetensors weights"]
fn live_flight_dji_0030_nadir_survey_corridor_track() -> TestResult {
    let spec = ClipSpec::nadir_drone(
        "DJI_0030.MP4",
        "NAVIGATE_DJI_0030_PATH",
        [40.5439, -74.4556],
    );
    let r = execute_real_corridor_clip(&spec)?;
    assert!(
        r.frames >= 4
            && r.warm_tracked_frames >= 1
            && r.p50_warm_ms > 0.0
            && r.p50_warm_ms < 33.3
            && r.max_inliers >= 24
            && r.best_filter_quality != "none"
    );
    Ok(())
}

#[test]
#[ignore = "requires local IMG_4047.MOV, xfeat.onnx, and ONNX Runtime with CoreML"]
fn live_flight_coreml_xfeat_runtime_smoke() -> TestResult {
    let onnx_model_path = std::env::var_os("NAVIGATE_XFEAT_ONNX")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join("delaware-video/assets/xfeat.onnx"))
                .filter(|p| p.is_file())
        })
        .ok_or("missing NAVIGATE_XFEAT_ONNX")?;
    let onnx_runtime_path = std::env::var_os("ORT_DYLIB_PATH")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(|h| PathBuf::from(h).join("delaware-video/assets/libonnxruntime.1.22.0.dylib"))
                .filter(|p| p.is_file())
        });
    let mut spec = ClipSpec::aircraft(
        "IMG_4047.MOV",
        "NAVIGATE_IMG_4047_PATH",
        "NAVIGATE_IMG_4047_CATALOG",
        &[
            "reference-video/v4047/catalog.json",
            "reference-region/catalog.json",
        ],
        [40.3273, -75.4756, 1120.859],
    );
    spec.duration_s = 2.0;
    spec.min_accepted = 2;
    spec.runtime_override = Some(XFeatRuntimeConfig {
        runtime: XFeatRuntime::CoreMl,
        onnx_model_path: Some(onnx_model_path),
        onnx_runtime_path,
    });
    let r = execute_real_corridor_clip(&spec)?;
    assert!(r.accepted >= 2 && r.max_inliers >= 24 && r.p50_total_ms > 0.0);
    Ok(())
}
