//! Neighbor tracking, refinement, bundle adjustment, and recipe execution (`photo_stages`).

mod bundle;

pub use bundle::run_terrain_bundle_stage;

use super::{
    AnchorPoseMap, MatchStageOptions, PipelinePass, collect_parent_anchors, estimate_intrinsics,
    load_case_mask, load_case_rgb, load_profile_cases, mask_allows, run_match_stage, scale_k,
    scaled_size, write_case_report,
};
use crate::{
    PipelineError,
    matcher_client::PipelineMatcher,
    pose::{LoadedPose, PoseCandidate, candidate_priority, fit_pose_ransac},
    surface::{ContextSurface, Surface},
    testset::{FlightGroup, PhotoCase},
    util::{parse_iso8601_ms, read_json, write_json},
};
use image::{RgbImage, imageops::FilterType};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
};

type CachedAnchorRender = (RgbImage, Vec<f32>, Vec<[f64; 3]>, [u32; 2]);

pub(super) fn load_flight_group_map(dataset_dir: &Path) -> HashMap<String, usize> {
    let mut map = HashMap::new();
    let Ok(doc) = read_json::<Value>(&dataset_dir.join("manifest.json")) else {
        return map;
    };
    let groups_val = doc
        .get("capture_groups")
        .or_else(|| doc.get("flights"))
        .cloned();
    let Some(groups) = groups_val.and_then(|v| serde_json::from_value::<Vec<FlightGroup>>(v).ok())
    else {
        return map;
    };
    for (g_idx, grp) in groups.into_iter().enumerate() {
        for photo_id in grp.photos {
            map.insert(photo_id, g_idx);
        }
    }
    map
}

fn capture_date_prefix(case: &PhotoCase) -> Option<&str> {
    let s = case.prior.as_ref()?.capture_time.as_deref()?.trim();
    if s.len() >= 10
        && matches!(s.as_bytes()[4], b'-' | b':')
        && matches!(s.as_bytes()[7], b'-' | b':')
    {
        Some(&s[..10])
    } else {
        None
    }
}

fn capture_time_ms(case: &PhotoCase) -> Option<i64> {
    case.prior
        .as_ref()?
        .capture_time
        .as_deref()
        .and_then(parse_iso8601_ms)
}

pub(super) fn rank_neighbor_anchors<'a>(
    query: &PhotoCase,
    cases_by_id: &BTreeMap<&str, &'a PhotoCase>,
    group_map: &HashMap<String, usize>,
    anchors: &'a AnchorPoseMap,
    surface: &Surface,
) -> Vec<(&'a String, &'a LoadedPose, &'a PhotoCase)> {
    let q_group = group_map.get(&query.id).copied();
    let q_date = capture_date_prefix(query);
    let q_ms = capture_time_ms(query);
    let q_enu = query
        .prior
        .as_ref()
        .and_then(|p| p.position.as_ref())
        .map(|pos| surface.local(pos.latitude, pos.longitude, 0.0));

    let mut ranked = Vec::with_capacity(anchors.len());
    for (anchor_id, pose) in anchors {
        if anchor_id == &query.id {
            continue;
        }
        let Some(&anchor_case) = cases_by_id.get(anchor_id.as_str()) else {
            continue;
        };
        let same_grp = q_group.is_some() && q_group == group_map.get(anchor_id.as_str()).copied();
        let same_day = match (q_date, capture_date_prefix(anchor_case)) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        };
        let same_flight_date = same_grp || same_day;
        let abs_dt_s = match (q_ms, capture_time_ms(anchor_case)) {
            (Some(t0), Some(t1)) => (t0 - t1).unsigned_abs() as f64 / 1_000.0,
            _ => f64::INFINITY,
        };
        let horiz_dist_m = match q_enu {
            Some(qe) => (qe[0] - pose.0[0]).hypot(qe[1] - pose.0[1]),
            None => f64::INFINITY,
        };
        ranked.push((
            !same_flight_date,
            abs_dt_s,
            horiz_dist_m,
            anchor_id,
            pose,
            anchor_case,
        ));
    }
    ranked.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.total_cmp(&b.1))
            .then_with(|| a.2.total_cmp(&b.2))
            .then_with(|| a.3.cmp(b.3))
    });
    ranked
        .into_iter()
        .map(|(_, _, _, id, pose, ac)| (id, pose, ac))
        .collect()
}

/// Propagate poses from solved anchors in `parent_dirs` to unresolved photos (`photo_tracking`).
///
/// # Errors
///
/// Returns [`PipelineError`] when catalog, anchor, or image processing fails.
pub fn run_track_stage(
    opts: &MatchStageOptions,
    parent_dirs: &[PathBuf],
    matcher: &mut PipelineMatcher,
) -> Result<Value, PipelineError> {
    let primary = Surface::from_catalog(&opts.catalog_path)?;
    let surface = ContextSurface {
        primary,
        context: None,
    };
    fs::create_dir_all(&opts.output_dir).map_err(|source| PipelineError::Io {
        path: opts.output_dir.clone(),
        source,
    })?;
    write_json(
        &opts.output_dir.join("run.json"),
        &json!({ "catalog_sha256": surface.primary.catalog_sha256 }),
    )?;

    let cases = load_profile_cases(&opts.profile_path)?;
    let cases_by_id: BTreeMap<&str, &PhotoCase> =
        cases.iter().map(|c| (c.id.as_str(), c)).collect();
    let group_map = load_flight_group_map(&opts.dataset_dir);
    let anchors = collect_parent_anchors(parent_dirs, &surface.primary)?;
    let mut anchor_cache: HashMap<String, CachedAnchorRender> = HashMap::new();
    let mut accepted_count = 0usize;

    for case in &cases {
        if anchors.contains_key(&case.id) {
            continue;
        }
        if let Some(allow) = &opts.only_cases
            && !allow.contains(&case.id)
        {
            continue;
        }
        let qry_rgb = load_case_rgb(&opts.dataset_dir, &case.image)?;
        let valid_mask = load_case_mask(&opts.dataset_dir, &case.valid_mask, case.size)?;
        let k_qry = estimate_intrinsics(case);
        let q_sz = scaled_size(case.size, 640);
        let qx_s = f64::from(q_sz[0]) / f64::from(case.size[0].max(1));
        let qy_s = f64::from(q_sz[1]) / f64::from(case.size[1].max(1));
        let q_small = image::imageops::resize(&qry_rgb, q_sz[0], q_sz[1], FilterType::Triangle);

        let ranked_anchors =
            rank_neighbor_anchors(case, &cases_by_id, &group_map, &anchors, &surface.primary);
        let mut best: Option<(usize, usize, PoseCandidate)> = None;
        let mut attempts_log = Vec::new();

        for (a_idx, (anchor_id, (a_eye, a_rot, a_k, a_size), anchor_case)) in
            ranked_anchors.into_iter().take(6).enumerate()
        {
            if !anchor_cache.contains_key(anchor_id.as_str()) {
                let anchor_rgb = load_case_rgb(&opts.dataset_dir, &anchor_case.image)?;
                let render_sz = scaled_size(*a_size, 640);
                let sx = f64::from(render_sz[0]) / f64::from(a_size[0].max(1));
                let sy = f64::from(render_sz[1]) / f64::from(a_size[1].max(1));
                let (_, a_depth, a_pts) =
                    surface.render(*a_eye, a_rot, &scale_k(a_k, sx, sy), render_sz);
                let a_small = image::imageops::resize(
                    &anchor_rgb,
                    render_sz[0],
                    render_sz[1],
                    FilterType::Triangle,
                );
                anchor_cache.insert(anchor_id.clone(), (a_small, a_depth, a_pts, render_sz));
            }
            let (a_small, a_depth, a_pts, render_sz) = &anchor_cache[anchor_id.as_str()];

            let matches = matcher.match_rgb(a_small, &q_small)?;
            let mut world_pts = Vec::new();
            let mut img_px = Vec::new();
            for m in &matches {
                let rx =
                    (m.reference.x.round() as i64).clamp(0, i64::from(render_sz[0]) - 1) as u32;
                let ry =
                    (m.reference.y.round() as i64).clamp(0, i64::from(render_sz[1]) - 1) as u32;
                let idx = (ry * render_sz[0] + rx) as usize;
                if a_depth[idx] <= 0.5 {
                    continue;
                }
                let qx = m.query.x / qx_s;
                let qy = m.query.y / qy_s;
                if mask_allows(&valid_mask, case.size, qx, qy) {
                    world_pts.push(a_pts[idx]);
                    img_px.push([qx, qy]);
                }
            }
            if let Some(cand) = fit_pose_ransac(
                &world_pts,
                &img_px,
                &k_qry,
                case.size,
                Some((*a_eye, *a_rot)),
            ) {
                attempts_log.push(json!({
                    "anchor": anchor_id,
                    "inliers": cand.score.inliers,
                    "geometry_accepted": cand.score.geometry_accepted,
                }));
                let replace = match &best {
                    Some((_, _, prev)) => {
                        candidate_priority(&cand.score) > candidate_priority(&prev.score)
                    }
                    None => true,
                };
                if replace {
                    best = Some((a_idx, 0, cand));
                }
            }
        }

        write_case_report(
            &opts.output_dir,
            &surface.primary,
            &case.id,
            "neighbor_track",
            attempts_log,
            best,
            &mut accepted_count,
        )?;
    }
    Ok(json!({ "accepted": accepted_count, "output": opts.output_dir.display().to_string() }))
}

/// Re-render and refine existing parent candidates (`photo_refine`).
///
/// # Errors
///
/// Returns [`PipelineError`] when reading candidates or refining fails.
pub fn run_refine_stage(
    opts: &MatchStageOptions,
    parent_dirs: &[PathBuf],
    matcher: &mut PipelineMatcher,
) -> Result<Value, PipelineError> {
    let primary = Surface::from_catalog(&opts.catalog_path)?;
    let surface = ContextSurface {
        primary,
        context: None,
    };
    fs::create_dir_all(&opts.output_dir).map_err(|source| PipelineError::Io {
        path: opts.output_dir.clone(),
        source,
    })?;
    write_json(
        &opts.output_dir.join("run.json"),
        &json!({ "catalog_sha256": surface.primary.catalog_sha256 }),
    )?;
    let cases = load_profile_cases(&opts.profile_path)?;
    let anchors = collect_parent_anchors(parent_dirs, &surface.primary)?;
    let mut accepted_count = 0usize;

    for case in &cases {
        if let Some(allow) = &opts.only_cases
            && !allow.contains(&case.id)
        {
            continue;
        }
        let Some(&(mut eye, mut rot, a_k, a_size)) = anchors.get(&case.id) else {
            continue;
        };
        let qry_rgb = load_case_rgb(&opts.dataset_dir, &case.image)?;
        let valid_mask = load_case_mask(&opts.dataset_dir, &case.valid_mask, case.size)?;
        let mut best: Option<(usize, usize, PoseCandidate)> = None;
        let mut attempts_log = Vec::new();
        for step in 0..opts.steps.max(2) {
            let rsz = scaled_size(case.size, 800);
            let k_sx = f64::from(rsz[0]) / f64::from(a_size[0].max(1));
            let k_sy = f64::from(rsz[1]) / f64::from(a_size[1].max(1));
            let k_rsz = scale_k(&a_k, k_sx, k_sy);
            let q_sx = f64::from(rsz[0]) / f64::from(case.size[0].max(1));
            let q_sy = f64::from(rsz[1]) / f64::from(case.size[1].max(1));
            let (ref_rgb, ref_depth, ref_pts) = surface.render(eye, &rot, &k_rsz, rsz);
            let q_small = image::imageops::resize(&qry_rgb, rsz[0], rsz[1], FilterType::Triangle);
            let matches = matcher.match_rgb(&ref_rgb, &q_small)?;
            let mut wpts = Vec::new();
            let mut ipx = Vec::new();
            for m in &matches {
                let rx = (m.reference.x.round() as i64).clamp(0, i64::from(rsz[0]) - 1) as u32;
                let ry = (m.reference.y.round() as i64).clamp(0, i64::from(rsz[1]) - 1) as u32;
                let idx = (ry * rsz[0] + rx) as usize;
                let qx_full = m.query.x / q_sx;
                let qy_full = m.query.y / q_sy;
                if ref_depth[idx] > 0.5 && mask_allows(&valid_mask, case.size, qx_full, qy_full) {
                    wpts.push(ref_pts[idx]);
                    ipx.push([m.query.x, m.query.y]);
                }
            }
            let Some(cand) = fit_pose_ransac(&wpts, &ipx, &k_rsz, rsz, Some((eye, rot))) else {
                attempts_log.push(json!({
                    "step": step,
                    "raw_pairs": matches.len(),
                    "pairs": wpts.len(),
                    "inliers": 0,
                    "geometry_accepted": false,
                }));
                break;
            };
            eye = cand.eye;
            rot = cand.rotation;
            attempts_log.push(json!({
                "step": step,
                "pairs": cand.score.pairs,
                "inliers": cand.score.inliers,
                "holdout_inlier_fraction": cand.score.holdout_inlier_fraction,
                "holdout_median_px": cand.score.holdout_median_px,
                "geometry_accepted": cand.score.geometry_accepted,
            }));
            let replace = match &best {
                Some((_, _, prev)) => {
                    candidate_priority(&cand.score) > candidate_priority(&prev.score)
                }
                None => true,
            };
            if replace {
                best = Some((0, step, cand));
            }
        }
        write_case_report(
            &opts.output_dir,
            &surface.primary,
            &case.id,
            "terrain_refinement",
            attempts_log,
            best,
            &mut accepted_count,
        )?;
    }
    Ok(json!({ "accepted": accepted_count, "output": opts.output_dir.display().to_string() }))
}

pub(super) fn resolve_unresolved_allowlist(
    dataset_dir: &Path,
    profile_path: &Path,
    unresolved_from: &[String],
) -> Result<Option<BTreeSet<String>>, PipelineError> {
    if unresolved_from.is_empty() {
        return Ok(None);
    }
    let cases = load_profile_cases(profile_path)?;
    let mut allow = BTreeSet::new();
    for case in &cases {
        let resolved = unresolved_from.iter().any(|u_id| {
            let p = dataset_dir
                .join("runs")
                .join(u_id)
                .join(format!("{}.json", case.id));
            read_json::<Value>(&p).ok().is_some_and(|doc| {
                doc.get("status").and_then(Value::as_str) == Some("candidate_requires_review")
                    || doc.get("accepted").and_then(Value::as_bool) == Some(true)
            })
        });
        if !resolved {
            allow.insert(case.id.clone());
        }
    }
    Ok(Some(allow))
}

pub(super) fn build_pass_match_options(
    dataset_dir: &Path,
    profile_path: &Path,
    catalog_path: &Path,
    output_dir: &Path,
    pass: &PipelinePass,
) -> Result<MatchStageOptions, PipelineError> {
    let only_cases =
        resolve_unresolved_allowlist(dataset_dir, profile_path, &pass.unresolved_from)?;
    Ok(MatchStageOptions {
        dataset_dir: dataset_dir.to_path_buf(),
        profile_path: profile_path.to_path_buf(),
        catalog_path: catalog_path.to_path_buf(),
        context_catalog: None,
        output_dir: output_dir.to_path_buf(),
        attempts: pass.attempts.unwrap_or(12),
        steps: 2,
        heading_step_deg: pass.heading_step.unwrap_or(0.0),
        height_factors: pass.height_factors.clone().unwrap_or_else(|| vec![1.0]),
        only_cases,
    })
}

/// Execute a declarative multi-pass `pipeline.json` recipe (`photo_pipeline`).
///
/// # Errors
///
/// Returns [`PipelineError`] when recipe parsing or any stage fails.
pub fn run_pipeline_recipe(
    recipe_path: &Path,
    matcher: &mut PipelineMatcher,
) -> Result<Value, PipelineError> {
    let root = recipe_path.parent().unwrap_or_else(|| Path::new("."));
    let doc: Value = read_json(recipe_path)?;
    let dataset_rel = doc
        .get("dataset")
        .and_then(Value::as_str)
        .unwrap_or("dataset");
    let dataset_dir = root.join(dataset_rel);
    let default_cat = doc
        .get("reference_catalog")
        .and_then(Value::as_str)
        .unwrap_or("dataset/reference-region/catalog.json");
    let profile_path = dataset_dir.join("profiles").join("with-prior.json");
    let passes: Vec<PipelinePass> = doc
        .get("passes")
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(|source| PipelineError::Json {
            context: "pipeline.passes".into(),
            source,
        })?
        .unwrap_or_default();

    let mut reports = Vec::new();
    for pass in &passes {
        let out_dir = dataset_dir.join("runs").join(&pass.id);
        let cat = root.join(pass.reference_catalog.as_deref().unwrap_or(default_cat));
        let parent_dirs: Vec<PathBuf> = pass
            .parents
            .iter()
            .map(|p| dataset_dir.join("runs").join(p))
            .collect();
        let opts = build_pass_match_options(&dataset_dir, &profile_path, &cat, &out_dir, pass)?;
        let rep = match pass.kind.as_str() {
            "track" => run_track_stage(&opts, &parent_dirs, matcher)?,
            "refine" => run_refine_stage(&opts, &parent_dirs, matcher)?,
            "terrain-bundle" => run_terrain_bundle_stage(&cat, &parent_dirs, &out_dir)?,
            _ => run_match_stage(&opts, matcher)?,
        };
        reports.push(json!({ "id": pass.id, "kind": pass.kind, "result": rep }));
    }
    Ok(json!({ "passes": reports }))
}

#[cfg(test)]
mod tests;
