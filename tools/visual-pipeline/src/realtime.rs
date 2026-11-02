//! Precomputed Kashin reference banks and real-time pose matching (`bank-build`, `bank-match`).

pub(crate) mod helpers;

use crate::{
    PipelineError,
    pose::candidate_priority,
    stages::{
        MatchStageOptions, estimate_intrinsics, generate_seeds, load_case_mask, load_case_rgb,
        load_horizon_attitude, load_profile_cases, scale_k, scaled_size, write_case_report,
    },
    surface::{Surface, SurfaceGpuRenderer},
    util::write_json,
    xfeat_runtime::{LoadedXFeat, XFeatRuntimeConfig},
};
use helpers::{
    check_encoder_contract, filter_terrain_features, hex, load_lighterglue_model, log_attempt,
    match_shortlisted_views, polish_pose_sparse, raw_file_sha256, rot_to_row_major,
};
use image::{DynamicImage, imageops::FilterType};
use navigate_match::{
    DEFAULT_KASHIN_SEED, HadamardProjector, PrecomputedReferenceBank, PrecomputedViewEntry,
    PreparedKashinQuery, SparseFeatures,
};
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, path::PathBuf, sync::Mutex, time::Instant};

/// Options that control offline [`PrecomputedReferenceBank`] construction (`bank-build`).
#[derive(Clone, Debug)]
pub struct BankBuildOptions {
    /// Dataset root directory (`manifest.json`, `images/`, `masks/`).
    pub dataset_dir: PathBuf,
    /// Case profile JSON (`profiles/with-prior.json` or `no-prior.json`).
    pub profile_path: PathBuf,
    /// Primary terrain and orthoimagery `catalog.json` path.
    pub catalog_path: PathBuf,
    /// Path to `XFeat` `.safetensors` weights.
    pub xfeat_weights_path: PathBuf,
    /// Output `NVKBANK1` file path.
    pub output_bank_path: PathBuf,
    /// Maximum seed poses precomputed per case.
    pub attempts: usize,
    /// Heading sweep step in degrees (`0.0` uses prior heading offsets).
    pub heading_step_deg: f64,
    /// Multipliers applied to nominal height above ground.
    pub height_factors: Vec<f64>,
    /// Maximum `XFeat` keypoints stored per reference view (`1..=4096`).
    pub max_keypoints: usize,
    /// Coarse pixel stride for [`Surface::render_adaptive`] (`1` disables subsampling).
    pub adaptive_stride: u32,
    /// Optional case ID allow-list.
    pub only_cases: Option<BTreeSet<String>>,
    /// Sparse `XFeat` runtime configuration (`burn-f32`, `burn-f16`, `coreml`, `tensorrt`).
    pub xfeat_runtime: XFeatRuntimeConfig,
}

/// Options that control real-time pose matching against a [`PrecomputedReferenceBank`] (`bank-match`).
#[derive(Clone, Debug)]
pub struct BankMatchOptions {
    /// Dataset root directory (`manifest.json`, `images/`, `masks/`).
    pub dataset_dir: PathBuf,
    /// Case profile JSON (`profiles/with-prior.json` or `no-prior.json`).
    pub profile_path: PathBuf,
    /// Primary terrain and orthoimagery `catalog.json` path.
    pub catalog_path: PathBuf,
    /// Input `NVKBANK1` file path.
    pub bank_path: PathBuf,
    /// Path to `XFeat` `.safetensors` weights.
    pub xfeat_weights_path: PathBuf,
    /// Optional path to `LighterGlue` `.safetensors` weights for adaptive cross-attention.
    pub lighterglue_weights_path: Option<PathBuf>,
    /// Stage output directory (`run.json`, `{id}.json`, `{id}/candidate.npz`).
    pub output_dir: PathBuf,
    /// Maximum reference views shortlisted by `ViewSketch` similarity per query.
    pub shortlist_k: usize,
    /// Maximum `XFeat` keypoints extracted from each masked query image (`1..=4096`).
    pub query_keypoints: usize,
    /// Run one GPU (`wgpu-terrain-v1`) or adaptive CPU render and depth-unproject pose polish pass.
    pub polish: bool,
    /// Optional case ID allow-list.
    pub only_cases: Option<BTreeSet<String>>,
    /// Sparse `XFeat` runtime configuration (`burn-f32`, `burn-f16`, `coreml`, `tensorrt`).
    pub xfeat_runtime: XFeatRuntimeConfig,
}

/// Build and save a [`PrecomputedReferenceBank`] (`NVKBANK1`) from terrain seed views.
///
/// # Errors
///
/// Returns [`PipelineError`] when catalog loading, weight loading, feature extraction, or bank writing fails.
pub fn build_reference_bank(opts: &BankBuildOptions) -> Result<Value, PipelineError> {
    let t0 = Instant::now();
    let surface = Surface::from_catalog(&opts.catalog_path)?;
    let catalog_digest = raw_file_sha256(&opts.catalog_path)?;
    let xfeat = LoadedXFeat::load(&opts.xfeat_weights_path, &opts.xfeat_runtime)?;
    let cases = load_profile_cases(&opts.profile_path)?;
    let projector = HadamardProjector::new(DEFAULT_KASHIN_SEED);

    let stage_opts = MatchStageOptions {
        dataset_dir: opts.dataset_dir.clone(),
        profile_path: opts.profile_path.clone(),
        catalog_path: opts.catalog_path.clone(),
        context_catalog: None,
        output_dir: PathBuf::new(),
        attempts: opts.attempts.max(1),
        steps: 1,
        heading_step_deg: opts.heading_step_deg,
        height_factors: if opts.height_factors.is_empty() {
            vec![1.0]
        } else {
            opts.height_factors.clone()
        },
        only_cases: opts.only_cases.clone(),
    };

    let kp_limit = opts.max_keypoints.clamp(16, 4096);
    let stride = opts.adaptive_stride.max(1);
    let mut entries = Vec::new();
    let mut total_pixels = 0usize;
    let mut total_rays = 0usize;
    let mut total_keypoints_3d = 0usize;

    for case in &cases {
        if let Some(allow) = &opts.only_cases
            && !allow.contains(&case.id)
        {
            continue;
        }
        let k = estimate_intrinsics(case);
        let attitude = load_horizon_attitude(&opts.dataset_dir, case, &k);
        let render_size = scaled_size(case.size, 800);
        let sx = f64::from(render_size[0]) / f64::from(case.size[0].max(1));
        let sy = f64::from(render_size[1]) / f64::from(case.size[1].max(1));
        let k_render = scale_k(&k, sx, sy);
        let seeds = generate_seeds(&surface, case, &stage_opts, attitude);

        for (eye, rot) in seeds {
            let ((ref_rgb, _, _), stats) =
                surface.render_adaptive(eye, &rot, &k_render, render_size, stride, 8, 0.025);
            total_pixels += stats.total_pixels;
            total_rays += stats.rays_cast;
            let ref_gray = DynamicImage::ImageRgb8(ref_rgb).into_luma8();
            let raw_feat = xfeat.extract(&ref_gray, kp_limit)?;
            let hits = surface.raycast_pixels(eye, &rot, &k_render, &raw_feat.pixels);
            let (feat_3d, world_pts) = filter_terrain_features(&raw_feat, &hits);
            total_keypoints_3d += world_pts.len();
            let view_id = entries.len() as u32;
            let entry = PrecomputedViewEntry::from_features(
                view_id,
                eye,
                rot_to_row_major(&rot),
                render_size,
                &projector,
                &feat_3d,
                world_pts,
            )?;
            entries.push(entry);
        }
    }

    let bank = PrecomputedReferenceBank::new(
        catalog_digest,
        xfeat.encoder_contract(),
        DEFAULT_KASHIN_SEED,
        entries,
    )?;
    if let Some(parent) = opts.output_bank_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|source| PipelineError::Io {
            path: parent.to_path_buf(),
            source,
        })?;
    }
    let encoded = bank.to_bytes();
    fs::write(&opts.output_bank_path, &encoded).map_err(|source| PipelineError::Io {
        path: opts.output_bank_path.clone(),
        source,
    })?;
    let ray_reduction = if total_rays > 0 {
        (total_pixels as f64) / (total_rays as f64)
    } else {
        1.0
    };
    Ok(json!({
        "bank_path": opts.output_bank_path.display().to_string(),
        "bank_bytes": encoded.len(),
        "xfeat_runtime": xfeat.runtime_name(),
        "xfeat_weights_sha256": hex(xfeat.weights_sha256()),
        "xfeat_encoder_contract": hex(&xfeat.encoder_contract()),
        "views": bank.entries.len(),
        "keypoints_3d": total_keypoints_3d,
        "adaptive_total_pixels": total_pixels,
        "adaptive_rays_cast": total_rays,
        "adaptive_ray_reduction": ray_reduction,
        "elapsed_ms": t0.elapsed().as_secs_f64() * 1000.0,
    }))
}

/// Match query photos in real time against a saved [`PrecomputedReferenceBank`] (`bank-match`).
///
/// # Errors
///
/// Returns [`PipelineError`] when bank verification, feature extraction, matching, or report writing fails.
pub fn run_realtime_bank_match(opts: &BankMatchOptions) -> Result<Value, PipelineError> {
    let surface = Surface::from_catalog(&opts.catalog_path)?;
    let catalog_digest = raw_file_sha256(&opts.catalog_path)?;
    let xfeat = LoadedXFeat::load(&opts.xfeat_weights_path, &opts.xfeat_runtime)?;
    let lighterglue = opts
        .lighterglue_weights_path
        .as_deref()
        .map(|p| load_lighterglue_model(p, xfeat.device()))
        .transpose()?;
    let bank = PrecomputedReferenceBank::load_file(&opts.bank_path)?;
    if bank.catalog_sha256 != catalog_digest {
        return Err(PipelineError::Invalid(
            "precomputed bank catalog_sha256 does not match --catalog".into(),
        ));
    }
    check_encoder_contract("precomputed bank", &bank.encoder_contract, &xfeat)?;

    fs::create_dir_all(&opts.output_dir).map_err(|source| PipelineError::Io {
        path: opts.output_dir.clone(),
        source,
    })?;
    write_json(
        &opts.output_dir.join("run.json"),
        &json!({ "catalog_sha256": surface.catalog_sha256 }),
    )?;

    let projector = HadamardProjector::new(bank.seed);
    let cases = load_profile_cases(&opts.profile_path)?;
    let effective_cases = cases
        .iter()
        .filter(|c| {
            opts.only_cases
                .as_ref()
                .is_none_or(|allow| allow.contains(&c.id))
        })
        .count();
    let t_warmup = Instant::now();
    if effective_cases > 8 {
        let _ = xfeat.warmup(lighterglue.as_ref())?;
    }
    let init_gpu = opts
        .polish
        .then(|| {
            let k_init =
                nalgebra::Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
            SurfaceGpuRenderer::try_new(&surface, &k_init, [800, 600])
        })
        .flatten();
    let gpu_renderer: Mutex<Option<SurfaceGpuRenderer>> = Mutex::new(init_gpu);
    let shader_warmup_ms = t_warmup.elapsed().as_secs_f64() * 1000.0;
    let kp_limit = opts.query_keypoints.clamp(16, 4096);
    let top_k = opts.shortlist_k.max(1);
    let mut evaluated = 0usize;
    let mut accepted = 0usize;
    let mut case_telemetry = Vec::new();

    for case in &cases {
        if let Some(allow) = &opts.only_cases
            && !allow.contains(&case.id)
        {
            continue;
        }
        evaluated += 1;
        let case_t0 = Instant::now();
        let query_rgb = load_case_rgb(&opts.dataset_dir, &case.image)?;
        let valid_mask = load_case_mask(&opts.dataset_dir, &case.valid_mask, case.size)?;
        let k = estimate_intrinsics(case);
        let render_size = scaled_size(case.size, 800);
        let sx = f64::from(render_size[0]) / f64::from(case.size[0].max(1));
        let sy = f64::from(render_size[1]) / f64::from(case.size[1].max(1));
        let k_render = scale_k(&k, sx, sy);

        let qry_small = image::imageops::resize(
            &query_rgb,
            render_size[0],
            render_size[1],
            FilterType::Triangle,
        );
        let qry_gray = DynamicImage::ImageRgb8(qry_small).into_luma8();
        let mask_small = image::imageops::resize(
            &valid_mask,
            render_size[0],
            render_size[1],
            FilterType::Nearest,
        );

        let valid_px = mask_small.as_raw().iter().filter(|&&v| v >= 128).count();
        let total_px = (render_size[0] as usize)
            .saturating_mul(render_size[1] as usize)
            .max(1);
        let valid_frac = (valid_px as f64) / (total_px as f64);
        let qry_kp_cap = if valid_frac > 0.0 { kp_limit } else { 0 };
        let (qry_feat, query_xfeat_ms) = if qry_kp_cap > 0 {
            let t_xfeat = Instant::now();
            let feat = xfeat.extract_masked(&qry_gray, Some(&mask_small), qry_kp_cap)?;
            (feat, t_xfeat.elapsed().as_secs_f64() * 1000.0)
        } else {
            (SparseFeatures::empty(), 0.0)
        };

        let t_short = Instant::now();
        let prepared = PreparedKashinQuery::from_features(&projector, &qry_feat)?;
        let ranked = bank.shortlist_prepared(&prepared, top_k);
        let shortlist_us = t_short.elapsed().as_secs_f64() * 1_000_000.0;

        let t_match = Instant::now();
        let (mut best, mut attempts_log, mut lg_layers_total) = match_shortlisted_views(
            &bank,
            &projector,
            lighterglue.as_ref(),
            &qry_feat,
            &prepared,
            &ranked,
            (&k_render, render_size, kp_limit),
        )?;
        let bank_match_ms = t_match.elapsed().as_secs_f64() * 1000.0;

        let t_polish = Instant::now();
        let mut polish_rays = 0usize;
        if opts.polish
            && let Some((seed_idx, _, mut cur_best)) = best.clone()
        {
            let max_polish = if cur_best.score.geometry_accepted {
                1
            } else {
                2
            };
            for p_step in 1..=max_polish {
                let Some((polished, rays, layers)) = polish_pose_sparse(
                    &surface,
                    Some(&gpu_renderer),
                    &xfeat,
                    lighterglue.as_ref(),
                    &qry_feat,
                    (cur_best.eye, cur_best.rotation),
                    &k_render,
                    render_size,
                )?
                else {
                    break;
                };
                polish_rays += rays;
                lg_layers_total += layers;
                log_attempt(
                    &mut attempts_log,
                    "gpu-or-adaptive-polish+l1-gnc",
                    seed_idx,
                    p_step,
                    1.0,
                    polished.score.pairs,
                    Some(&polished),
                );
                let accepted_now = polished.score.geometry_accepted;
                if candidate_priority(&polished.score) >= candidate_priority(&cur_best.score) {
                    cur_best = polished.clone();
                    best = Some((seed_idx, p_step, polished));
                }
                if accepted_now {
                    break;
                }
            }
        }
        let polish_ms = t_polish.elapsed().as_secs_f64() * 1000.0;
        let total_query_ms = case_t0.elapsed().as_secs_f64() * 1000.0;

        let solved = best
            .as_ref()
            .is_some_and(|(_, _, c)| c.score.geometry_accepted);
        case_telemetry.push(json!({
            "id": case.id,
            "accepted": solved,
            "xfeat_runtime": xfeat.runtime_name(),
            "query_keypoints": qry_feat.len(),
            "shortlisted_views": ranked.len(),
            "lighterglue_layers": lg_layers_total,
            "polish_rays_cast": polish_rays,
            "query_xfeat_ms": query_xfeat_ms,
            "shortlist_us": shortlist_us,
            "bank_match_ms": bank_match_ms,
            "polish_ms": polish_ms,
            "total_case_ms": total_query_ms,
        }));

        write_case_report(
            &opts.output_dir,
            &surface,
            &case.id,
            "kashin_bank_match",
            attempts_log,
            best,
            &mut accepted,
        )?;
    }

    Ok(json!({
        "evaluated": evaluated,
        "accepted": accepted,
        "bank_views": bank.entries.len(),
        "xfeat_runtime": xfeat.runtime_name(),
        "shader_warmup_ms": shader_warmup_ms,
        "output": opts.output_dir.display().to_string(),
        "telemetry": case_telemetry,
    }))
}
