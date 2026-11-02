//! Terrain pose search stage (`photo_match`) and shared stage helpers.

pub mod focal;
mod followup;
mod gpu_match;
pub mod seeds;

pub use followup::{
    run_pipeline_recipe, run_refine_stage, run_terrain_bundle_stage, run_track_stage,
};
pub use seeds::generate_seeds;

use crate::{
    PipelineError,
    matcher_client::PipelineMatcher,
    pose::{PoseCandidate, load_candidate_pose, save_candidate_npz},
    surface::{ContextSurface, Surface, horizon_attitude},
    testset::PhotoCase,
    util::{read_json, write_json},
};
use gpu_match::StageTerrainRenderer;
use image::{GrayImage, RgbImage};
use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

/// Options controlling a single terrain pose-search or refinement stage.
#[derive(Clone, Debug)]
pub struct MatchStageOptions {
    /// Dataset root directory (`manifest.json`, `images/`, `masks/`).
    pub dataset_dir: PathBuf,
    /// Case profile JSON (`profiles/with-prior.json` or `no-prior.json`).
    pub profile_path: PathBuf,
    /// Primary `catalog.json` path.
    pub catalog_path: PathBuf,
    /// Optional regional context `catalog.json` path.
    pub context_catalog: Option<PathBuf>,
    /// Stage output directory (`run.json`, `{id}.json`, `{id}/candidate.npz`).
    pub output_dir: PathBuf,
    /// Maximum seed poses evaluated per photo.
    pub attempts: usize,
    /// Iterative re-render refinement steps per accepted seed.
    pub steps: usize,
    /// Heading sweep step in degrees (`0` uses reported direction or cardinal headings).
    pub heading_step_deg: f64,
    /// Multipliers applied to nominal height above ground.
    pub height_factors: Vec<f64>,
    /// Optional case ID allow-list (`unresolved_from` filter).
    pub only_cases: Option<BTreeSet<String>>,
}

/// Pass entry inside `pipeline.json`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PipelinePass {
    /// Unique stage identifier under `runs/{id}`.
    pub id: String,
    /// Stage kind (`"match"`, `"track"`, `"refine"`, `"terrain-bundle"`).
    #[serde(default = "default_pass_kind")]
    pub kind: String,
    /// Optional override catalog path relative to recipe parent.
    pub reference_catalog: Option<String>,
    /// Optional override backend name.
    pub backend: Option<String>,
    /// Optional attempt count override.
    pub attempts: Option<usize>,
    /// Optional heading step override in degrees.
    pub heading_step: Option<f64>,
    /// Optional height factors override.
    pub height_factors: Option<Vec<f64>>,
    /// Parent run IDs supplying anchor poses.
    #[serde(default)]
    pub parents: Vec<String>,
    /// Run IDs whose unresolved cases remain eligible.
    #[serde(default)]
    pub unresolved_from: Vec<String>,
}

fn default_pass_kind() -> String {
    "match".into()
}

/// Execute terrain pose search (`photo_match`) across cases in `opts.profile_path`.
///
/// # Errors
///
/// Returns [`PipelineError`] when catalog loading, image reading, matching, or writing fails.
pub fn run_match_stage(
    opts: &MatchStageOptions,
    matcher: &mut PipelineMatcher,
) -> Result<Value, PipelineError> {
    let primary = Surface::from_catalog(&opts.catalog_path)?;
    let context = opts
        .context_catalog
        .as_deref()
        .map(Surface::from_catalog)
        .transpose()?;
    let surface = ContextSurface { primary, context };
    let mut terrain_renderer = StageTerrainRenderer::new(&surface);
    fs::create_dir_all(&opts.output_dir).map_err(|source| PipelineError::Io {
        path: opts.output_dir.clone(),
        source,
    })?;
    write_json(
        &opts.output_dir.join("run.json"),
        &json!({ "catalog_sha256": surface.primary.catalog_sha256 }),
    )?;

    let cases = load_profile_cases(&opts.profile_path)?;
    let mut accepted_count = 0usize;
    let mut evaluated_count = 0usize;

    for case in &cases {
        if let Some(allow) = &opts.only_cases
            && !allow.contains(&case.id)
        {
            continue;
        }
        evaluated_count += 1;
        let query_rgb = load_case_rgb(&opts.dataset_dir, &case.image)?;
        let valid_mask = load_case_mask(&opts.dataset_dir, &case.valid_mask, case.size)?;
        let (attempts_log, best) = focal::match_case_with_focal_policy(
            opts,
            &surface,
            &mut terrain_renderer,
            matcher,
            case,
            &query_rgb,
            &valid_mask,
        )?;

        write_case_report(
            &opts.output_dir,
            &surface.primary,
            &case.id,
            "terrain_match",
            attempts_log,
            best,
            &mut accepted_count,
        )?;
    }

    Ok(json!({
        "evaluated": evaluated_count,
        "accepted": accepted_count,
        "output": opts.output_dir.display().to_string(),
    }))
}

pub(crate) fn load_profile_cases(profile_path: &Path) -> Result<Vec<PhotoCase>, PipelineError> {
    let doc: Value = read_json(profile_path)?;
    serde_json::from_value(doc["cases"].clone()).map_err(|source| PipelineError::Json {
        context: format!("{}:cases", profile_path.display()),
        source,
    })
}

pub(crate) fn load_case_rgb(dataset_dir: &Path, rel: &str) -> Result<RgbImage, PipelineError> {
    let p = dataset_dir.join(rel);
    Ok(image::open(&p)
        .map_err(|source| PipelineError::Image { path: p, source })?
        .to_rgb8())
}

pub(crate) fn load_case_mask(
    dataset_dir: &Path,
    rel: &str,
    size: [u32; 2],
) -> Result<GrayImage, PipelineError> {
    let p = dataset_dir.join(rel);
    if !p.is_file() {
        return Ok(GrayImage::from_pixel(
            size[0],
            size[1],
            image::Luma([255u8]),
        ));
    }
    Ok(image::open(&p)
        .map_err(|source| PipelineError::Image { path: p, source })?
        .to_luma8())
}

pub(super) fn mask_allows(mask: &GrayImage, size: [u32; 2], x: f64, y: f64) -> bool {
    if x < 0.0 || y < 0.0 || x >= f64::from(size[0]) || y >= f64::from(size[1]) {
        return false;
    }
    let mx = ((x / f64::from(size[0].max(1))) * f64::from(mask.width().max(1))).floor() as u32;
    let my = ((y / f64::from(size[1].max(1))) * f64::from(mask.height().max(1))).floor() as u32;
    mask.get_pixel(mx.min(mask.width() - 1), my.min(mask.height() - 1))
        .0[0]
        > 0
}

pub(crate) fn has_calibrated_focal_prior(case: &PhotoCase) -> bool {
    case.prior
        .as_ref()
        .and_then(|p| p.focal_35mm_mm)
        .is_some_and(|f35| f35 > 5.0)
}

pub(crate) fn estimate_intrinsics(case: &PhotoCase) -> Matrix3<f64> {
    estimate_intrinsics_with_estimator(case, &focal::FocalLengthEstimator::default())
}

pub(crate) fn estimate_intrinsics_with_estimator(
    case: &PhotoCase,
    estimator: &focal::FocalLengthEstimator,
) -> Matrix3<f64> {
    let f_px = if let Some(locked_px) = estimator.locked_focal_px.filter(|&px| px > 1.0) {
        locked_px
    } else if let Some(f35) = case
        .prior
        .as_ref()
        .and_then(|p| p.focal_35mm_mm)
        .filter(|&f35| f35 > 5.0)
    {
        f64::from(case.size[0]).hypot(f64::from(case.size[1])) * f35 / 43.266_615_3
    } else {
        estimator.baseline_focal_px(case.size[0].max(1))
    };
    Matrix3::new(
        f_px,
        0.0,
        (f64::from(case.size[0]) - 1.0) * 0.5,
        0.0,
        f_px,
        (f64::from(case.size[1]) - 1.0) * 0.5,
        0.0,
        0.0,
        1.0,
    )
}

pub(crate) fn load_horizon_attitude(
    dataset_dir: &Path,
    case: &PhotoCase,
    k: &Matrix3<f64>,
) -> Option<(f64, f64)> {
    let sem_rel = case.semantic_labels.as_deref()?;
    let img = image::open(dataset_dir.join(sem_rel)).ok()?.to_luma8();
    horizon_attitude(img.as_raw(), img.width(), img.height(), k)
}

pub(crate) fn scaled_size(size: [u32; 2], max_edge: u32) -> [u32; 2] {
    let m = size[0].max(size[1]).max(1);
    if m <= max_edge {
        return size;
    }
    let s = f64::from(max_edge) / f64::from(m);
    [
        ((f64::from(size[0]) * s).round() as u32).max(16),
        ((f64::from(size[1]) * s).round() as u32).max(16),
    ]
}

pub(crate) fn scale_k(k: &Matrix3<f64>, sx: f64, sy: f64) -> Matrix3<f64> {
    Matrix3::new(
        k[(0, 0)] * sx,
        0.0,
        k[(0, 2)] * sx,
        0.0,
        k[(1, 1)] * sy,
        k[(1, 2)] * sy,
        0.0,
        0.0,
        1.0,
    )
}

pub(super) type AnchorPoseMap = BTreeMap<String, ([f64; 3], Matrix3<f64>, Matrix3<f64>, [u32; 2])>;

pub(super) fn collect_parent_anchors(
    parent_dirs: &[PathBuf],
    surface: &Surface,
) -> Result<AnchorPoseMap, PipelineError> {
    let mut map = BTreeMap::new();
    for dir in parent_dirs {
        let same_cat = read_json::<Value>(&dir.join("run.json"))
            .ok()
            .and_then(|r| r["catalog_sha256"].as_str().map(str::to_owned))
            .as_deref()
            == Some(surface.catalog_sha256.as_str());
        let Ok(entries) = fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().and_then(|s| s.to_str()) != Some("json")
                || p.file_name().and_then(|s| s.to_str()) == Some("run.json")
            {
                continue;
            }
            let Ok(doc) = read_json::<Value>(&p) else {
                continue;
            };
            if doc.get("status").and_then(Value::as_str) != Some("candidate_requires_review") {
                continue;
            }
            let Some(id) = doc.get("id").and_then(Value::as_str) else {
                continue;
            };
            let npz = dir.join(id).join("candidate.npz");
            if npz.is_file()
                && let Ok(mut pose) = load_candidate_pose(&npz)
            {
                if !same_cat
                    && let Some(c) = doc.get("candidate")
                    && let (Some(lat), Some(lon), Some(alt)) = (
                        c["latitude"].as_f64(),
                        c["longitude"].as_f64(),
                        c["altitude_m"].as_f64(),
                    )
                {
                    pose.0 = surface.local(lat, lon, alt);
                }
                map.insert(id.to_owned(), pose);
            }
        }
    }
    Ok(map)
}

pub(crate) fn write_case_report(
    output_dir: &Path,
    surface: &Surface,
    case_id: &str,
    evidence_kind: &str,
    attempts: Vec<Value>,
    best: Option<(usize, usize, PoseCandidate)>,
    accepted_count: &mut usize,
) -> Result<(), PipelineError> {
    let report = if let Some((seed, step, cand)) = best
        && cand.score.geometry_accepted
    {
        *accepted_count += 1;
        let case_dir = output_dir.join(case_id);
        fs::create_dir_all(&case_dir).map_err(|source| PipelineError::Io {
            path: case_dir.clone(),
            source,
        })?;
        save_candidate_npz(&case_dir.join("candidate.npz"), &cand)?;
        let [lat, lon, alt] = surface.geodetic(cand.eye);
        json!({
            "id": case_id,
            "status": "candidate_requires_review",
            "attempts": attempts,
            "candidate": {
                "latitude": lat,
                "longitude": lon,
                "altitude_m": alt,
                "evidence_kind": evidence_kind,
                "seed": seed,
                "step": step,
                "score": cand.score,
            },
        })
    } else {
        json!({
            "id": case_id,
            "status": "unresolved",
            "attempts": attempts,
            "candidate": null,
        })
    };
    write_json(&output_dir.join(format!("{case_id}.json")), &report)
}

#[cfg(test)]
mod tests;
