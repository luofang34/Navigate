//! Shared live flight video extraction, bank-build, and `run_corridor_track` harness.

use nalgebra::{Matrix3, Vector3};
use navigate_visual_pipeline::{
    CorridorBuildOptions, CorridorTrackOptions, XFeatRuntime, XFeatRuntimeConfig,
    build_corridor_bank, build_reference_catalog_zoom, corridor::VehicleDynamicsProfile,
    run_corridor_track, surface::Surface, video_ops::prepare_video_testset,
};
use serde_json::{Value, json};
use std::{
    error::Error,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

pub(super) type TestResult = Result<(), Box<dyn Error>>;

const ESRI_URL: &str = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const MAPZEN_URL: &str = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";

pub(super) struct ClipSpec<'a> {
    pub file_name: &'a str,
    pub explicit_env: &'a str,
    pub catalog_env: &'a str,
    pub rel_catalogs: &'a [&'a str],
    pub prior_lat_lon_alt: [f64; 3],
    pub prior_agl_default_m: f64,
    pub start_s: f64,
    pub duration_s: f64,
    pub fps: f64,
    pub tile_size_m: f64,
    pub corridor_radius_m: f64,
    pub nominal_agl_m: f64,
    pub include_obliques: bool,
    pub dynamics: VehicleDynamicsProfile,
    pub min_accepted: usize,
    pub max_pos_offset_m: f64,
    pub auto_catalog_zoom: Option<(u32, u32, f64)>,
    pub runtime_override: Option<XFeatRuntimeConfig>,
}

impl<'a> ClipSpec<'a> {
    pub fn aircraft(
        file_name: &'a str,
        explicit_env: &'a str,
        catalog_env: &'a str,
        rel_catalogs: &'a [&'a str],
        prior_lat_lon_alt: [f64; 3],
    ) -> Self {
        Self {
            file_name,
            explicit_env,
            catalog_env,
            rel_catalogs,
            prior_lat_lon_alt,
            prior_agl_default_m: 900.0,
            start_s: 0.0,
            duration_s: 3.0,
            fps: 5.0,
            tile_size_m: 1200.0,
            corridor_radius_m: 3500.0,
            nominal_agl_m: 900.0,
            include_obliques: true,
            dynamics: VehicleDynamicsProfile::fixed_wing(),
            min_accepted: 3,
            max_pos_offset_m: 4500.0,
            auto_catalog_zoom: None,
            runtime_override: None,
        }
    }

    pub fn nadir_drone(file_name: &'a str, explicit_env: &'a str, prior_lat_lon: [f64; 2]) -> Self {
        Self {
            file_name,
            explicit_env,
            catalog_env: "NAVIGATE_DJI_CATALOG",
            rel_catalogs: &["reference-dji/catalog.json"],
            prior_lat_lon_alt: [prior_lat_lon[0], prior_lat_lon[1], 0.0],
            prior_agl_default_m: 98.0,
            start_s: 1.0,
            duration_s: 3.0,
            fps: 5.0,
            tile_size_m: 150.0,
            corridor_radius_m: 300.0,
            nominal_agl_m: 98.0,
            include_obliques: false,
            dynamics: VehicleDynamicsProfile::rotary_survey(),
            min_accepted: 3,
            max_pos_offset_m: 600.0,
            auto_catalog_zoom: Some((19, 14, 360.0)),
            runtime_override: None,
        }
    }
}

pub(super) struct ClipRunReport {
    pub frames: usize,
    pub accepted: usize,
    pub warm_tracked_frames: usize,
    pub max_inliers: usize,
    pub first_valid_mask_fraction: f64,
    pub locked_focal_px: Option<f64>,
    pub best_filter_quality: String,
    pub build_elapsed_ms: f64,
    pub p50_total_ms: f64,
    pub p50_warm_ms: f64,
}

pub(super) fn resolve_clip_from(
    file_name: &str,
    explicit_path: Option<&Path>,
    clip_dir: Option<&Path>,
    home_dir: Option<&Path>,
) -> Result<PathBuf, Box<dyn Error>> {
    if let Some(p) = explicit_path {
        return p
            .is_file()
            .then(|| p.to_path_buf())
            .ok_or_else(|| format!("missing explicit clip path: {}", p.display()).into());
    }
    if let Some(dir) = clip_dir {
        let cand = dir.join(file_name);
        return cand
            .is_file()
            .then_some(cand)
            .ok_or_else(|| format!("missing {file_name} in {}", dir.display()).into());
    }
    if let Some(cand) = home_dir
        .map(|h| h.join("Downloads").join(file_name))
        .filter(|p| p.is_file())
    {
        return Ok(cand);
    }
    Err(format!("missing {file_name}; set NAVIGATE_FLIGHT_CLIP_DIR or explicit path").into())
}

fn require_env_path(key: &str) -> Result<PathBuf, Box<dyn Error>> {
    let p = PathBuf::from(std::env::var_os(key).ok_or_else(|| format!("missing {key}"))?);
    p.exists()
        .then_some(p.clone())
        .ok_or_else(|| format!("{key} does not exist: {}", p.display()).into())
}

fn resolve_catalog(
    spec: &ClipSpec<'_>,
    profile_path: &Path,
    work_dir: &Path,
) -> Result<PathBuf, Box<dyn Error>> {
    if let Some(explicit) = std::env::var_os(spec.catalog_env).map(PathBuf::from) {
        return explicit
            .is_file()
            .then_some(explicit.clone())
            .ok_or_else(|| format!("missing catalog: {}", explicit.display()).into());
    }
    let roots = std::env::var_os("NAVIGATE_CATALOG_ROOT")
        .map(PathBuf::from)
        .into_iter()
        .chain(std::env::var_os("HOME").map(|h| PathBuf::from(h).join("delaware-video")));
    for root in roots {
        for rel in spec.rel_catalogs {
            let cand = root.join(rel);
            if cand.is_file() {
                return Ok(cand);
            }
        }
    }
    if let Some((img_z, dem_z, radius_m)) = spec.auto_catalog_zoom {
        let cache = std::env::var_os("NAVIGATE_TILE_CACHE_DIR").map(PathBuf::from);
        let cat_dir = work_dir.join("auto_catalog");
        build_reference_catalog_zoom(
            profile_path,
            &cat_dir,
            ESRI_URL,
            MAPZEN_URL,
            cache.as_deref(),
            radius_m,
            img_z,
            dem_z,
        )?;
        return Ok(cat_dir.join("catalog.json"));
    }
    Err(format!("missing terrain catalog for {}", spec.file_name).into())
}

fn trim_video_window(src: &Path, dst: &Path, start_s: f64, duration_s: f64) -> TestResult {
    let bin = if Path::new("/opt/homebrew/bin/ffmpeg").is_file() {
        "/opt/homebrew/bin/ffmpeg"
    } else {
        "ffmpeg"
    };
    let st = Command::new(bin)
        .args([
            "-y",
            "-v",
            "error",
            "-ss",
            &format!("{start_s:.3}"),
            "-t",
            &format!("{duration_s:.3}"),
            "-i",
            src.to_str().ok_or("non-utf8 clip")?,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "18",
            dst.to_str().ok_or("non-utf8 out")?,
        ])
        .status()?;
    st.success()
        .then_some(())
        .ok_or_else(|| format!("ffmpeg failed on {}", src.display()).into())
}

fn seed_profile_position_prior(
    profile_path: &Path,
    lat_lon_alt: [f64; 3],
    catalog_for_dem: Option<&Path>,
    agl_default_m: f64,
) -> Result<[f64; 3], Box<dyn Error>> {
    let mut alt_m = lat_lon_alt[2];
    if alt_m <= 0.0
        && let Some(cat) = catalog_for_dem
        && let Ok(surf) = Surface::from_catalog(cat)
        && let Some(dem_m) = surf.sample_dem(lat_lon_alt[0], lat_lon_alt[1])
    {
        alt_m = dem_m + agl_default_m;
    }
    if alt_m <= 0.0 {
        alt_m = agl_default_m;
    }
    let resolved = [lat_lon_alt[0], lat_lon_alt[1], alt_m];
    let mut doc: Value = serde_json::from_str(&fs::read_to_string(profile_path)?)?;
    for c in doc["cases"].as_array_mut().ok_or("missing cases")? {
        c["prior"] = json!({
            "position": {
                "latitude": resolved[0],
                "longitude": resolved[1],
                "altitude_m": resolved[2],
                "unsigned_altitude_m": resolved[2].abs(),
                "altitude_datum": "msl"
            }
        });
    }
    fs::write(profile_path, serde_json::to_string_pretty(&doc)?)?;
    Ok(resolved)
}

fn percentile_p50(samples: &mut [f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(f64::total_cmp);
    samples[((0.5 * samples.len() as f64).ceil() as usize).clamp(1, samples.len()) - 1]
}

pub(super) fn prior_diag_cov() -> Matrix3<f64> {
    Matrix3::from_diagonal(&Vector3::new(1e8, 1e8, 100.0))
}

pub(super) fn execute_real_corridor_clip(
    spec: &ClipSpec<'_>,
) -> Result<ClipRunReport, Box<dyn Error>> {
    let explicit = std::env::var_os(spec.explicit_env).map(PathBuf::from);
    let clip_dir = std::env::var_os("NAVIGATE_FLIGHT_CLIP_DIR").map(PathBuf::from);
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let clip_path = resolve_clip_from(
        spec.file_name,
        explicit.as_deref(),
        clip_dir.as_deref(),
        home.as_deref(),
    )?;
    let xfeat = require_env_path("NAVIGATE_XFEAT_WEIGHTS")?;
    let lighterglue = require_env_path("NAVIGATE_LIGHTERGLUE_WEIGHTS")?;
    let seg_dir = require_env_path("NAVIGATE_SEGFORMER_DIR")?;

    let tmp = tempfile::tempdir()?;
    let trimmed = tmp.path().join("window.mp4");
    trim_video_window(&clip_path, &trimmed, spec.start_s, spec.duration_s)?;

    let dataset_dir = tmp.path().join("dataset");
    prepare_video_testset(&trimmed, &dataset_dir, spec.fps, 1280)?;
    let profile_path = dataset_dir.join("profiles").join("with-prior.json");
    seed_profile_position_prior(
        &profile_path,
        spec.prior_lat_lon_alt,
        None,
        spec.prior_agl_default_m,
    )?;

    let catalog_path = resolve_catalog(spec, &profile_path, tmp.path())?;
    let resolved_prior = seed_profile_position_prior(
        &profile_path,
        spec.prior_lat_lon_alt,
        Some(&catalog_path),
        spec.prior_agl_default_m,
    )?;

    let bank_dir = tmp.path().join("bank.nvktiledir");
    let build_runtime = XFeatRuntimeConfig {
        runtime: XFeatRuntime::BurnF16,
        ..XFeatRuntimeConfig::default()
    };
    let track_runtime = spec
        .runtime_override
        .clone()
        .unwrap_or_else(|| build_runtime.clone());
    let build_doc = build_corridor_bank(&CorridorBuildOptions {
        catalog_path: catalog_path.clone(),
        profile_path: Some(profile_path.clone()),
        xfeat_weights_path: xfeat.clone(),
        output_bank_path: bank_dir.clone(),
        tile_size_m: spec.tile_size_m,
        corridor_radius_m: spec.corridor_radius_m,
        nominal_agl_m: spec.nominal_agl_m,
        max_landmarks_per_tile: if spec.include_obliques { 4096 } else { 8192 },
        adaptive_stride: 2,
        include_obliques: spec.include_obliques,
        xfeat_runtime: build_runtime,
    })?;
    let build_elapsed_ms = build_doc["elapsed_ms"].as_f64().unwrap_or(0.0);
    assert!(build_doc["landmarks_3d"].as_u64().unwrap_or(0) >= 64);

    let out_dir = tmp.path().join("track_out");
    let surf = Surface::from_catalog(&catalog_path)?;
    let summary = run_corridor_track(&CorridorTrackOptions {
        dataset_dir,
        profile_path,
        catalog_path,
        corridor_bank_path: bank_dir,
        xfeat_weights_path: xfeat,
        lighterglue_weights_path: Some(lighterglue),
        seg_model_dir: Some(seg_dir),
        dinov2_weights_path: None,
        place_index_path: None,
        dynamics_profile: spec.dynamics,
        mask_policy: "airframe-occlusions".into(),
        mask_interval: 5,
        mask_interval_s: None,
        output_dir: out_dir.clone(),
        anchor_interval: 0,
        anchor_interval_s: None,
        anchor_uncertainty_m: None,
        anchor_on_cell_boundary: true,
        min_track_inliers: 24,
        map_cell_m: (spec.tile_size_m * 0.5).max(80.0),
        anchor_attempts: 16,
        heading_step_deg: 0.0,
        height_factors: vec![1.0, 0.75, 1.3],
        query_keypoints: 2048,
        polish_anchors: true,
        prior_override: Some((resolved_prior, None)),
        focal_35mm_override: None,
        only_cases: None,
        background_refresh: true,
        pace_realtime: false,
        xfeat_runtime: track_runtime,
    })?;

    let frames = summary["evaluated"].as_u64().unwrap_or(0) as usize;
    let accepted = summary["accepted"].as_u64().unwrap_or(0) as usize;
    let warm_tracked_frames = summary["warm_tracked_frames"].as_u64().unwrap_or(0) as usize;
    let seg_refreshes = summary["segformer_refreshes"].as_u64().unwrap_or(0) as usize;

    let telemetry = summary["telemetry"].as_array().ok_or("missing telemetry")?;
    let mut total_ms = Vec::with_capacity(frames);
    let mut warm_ms = Vec::new();
    let mut max_inliers = 0_usize;
    let mut first_valid_mask_fraction = 1.0_f64;
    let locked_focal_px = summary["locked_focal_px"].as_f64();
    let mut best_filter_quality = summary["filter_integrity"]["quality"]
        .as_str()
        .unwrap_or("none")
        .to_owned();
    let mut first_pose = [0.0_f64; 3];
    let prior_enu =
        Vector3::from(surf.local(resolved_prior[0], resolved_prior[1], resolved_prior[2]));

    for (idx, row) in telemetry.iter().enumerate() {
        let case_id = row["id"].as_str().ok_or("missing row id")?;
        let t_ms = row["total_case_ms"]
            .as_f64()
            .ok_or("missing total_case_ms")?;
        let mask_ms = row["mask_ms"].as_f64().unwrap_or(0.0);
        let xf_ms = row["query_xfeat_ms"]
            .as_f64()
            .ok_or("missing query_xfeat_ms")?;
        let solve_ms = row["solve_ms"].as_f64().unwrap_or(0.0);
        assert!(t_ms.is_finite() && t_ms > 0.0 && xf_ms.is_finite());
        total_ms.push(t_ms);
        if row["tracking_mode"].as_str() == Some("warm_track") {
            warm_ms.push(mask_ms + xf_ms + solve_ms);
        }
        if idx == 0
            && let Some(vf) = row["valid_mask_fraction"].as_f64()
        {
            first_valid_mask_fraction = vf;
        }
        if let Some(q) = row["filter_integrity"]["quality"].as_str()
            && q != "none"
        {
            best_filter_quality = q.to_owned();
        }
        let doc: Value = serde_json::from_str(&fs::read_to_string(
            out_dir.join(format!("{case_id}.json")),
        )?)?;
        if row["accepted"].as_bool() == Some(true)
            && let Some(cand) = doc["candidate"].as_object()
        {
            let inl = cand["score"]["inliers"].as_u64().unwrap_or(0) as usize;
            assert!(inl >= 24);
            let pose_deg = [
                cand["latitude"].as_f64().ok_or("lat")?,
                cand["longitude"].as_f64().ok_or("lon")?,
                cand["altitude_m"].as_f64().ok_or("alt")?,
            ];
            if max_inliers == 0 {
                first_pose = pose_deg;
            }
            max_inliers = max_inliers.max(inl);
            let p_enu = Vector3::from(surf.local(pose_deg[0], pose_deg[1], pose_deg[2]));
            let horiz_m =
                ((p_enu.x - prior_enu.x).powi(2) + (p_enu.y - prior_enu.y).powi(2)).sqrt();
            assert!(
                horiz_m <= spec.max_pos_offset_m,
                "{}: horiz_m={horiz_m:.1} > {}",
                spec.file_name,
                spec.max_pos_offset_m
            );
        }
    }

    let p50_total_ms = percentile_p50(&mut total_ms);
    let p50_warm_ms = percentile_p50(&mut warm_ms);
    writeln!(
        std::io::stdout(),
        "[live-benchmark] {} | build_ms={build_elapsed_ms:.0} frames={frames} accepted={accepted} warm={warm_tracked_frames} max_inliers={max_inliers} mask_frac={first_valid_mask_fraction:.3} locked_focal_px={locked_focal_px:?} filter={best_filter_quality} p50_total_ms={p50_total_ms:.2} p50_warm_ms={p50_warm_ms:.2} first_pose={first_pose:?}",
        spec.file_name
    )?;
    assert!(
        seg_refreshes >= 1,
        "{}: Segformer never executed",
        spec.file_name
    );
    assert!(
        accepted >= spec.min_accepted,
        "{}: accepted={accepted} < {} ({summary})",
        spec.file_name,
        spec.min_accepted
    );
    assert!(best_filter_quality != "none");

    Ok(ClipRunReport {
        frames,
        accepted,
        warm_tracked_frames,
        max_inliers,
        first_valid_mask_fraction,
        locked_focal_px,
        best_filter_quality,
        build_elapsed_ms,
        p50_total_ms,
        p50_warm_ms,
    })
}
