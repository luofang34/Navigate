//! Helper routines for [`super::build_reference_bank`] and [`super::run_realtime_bank_match`].

use crate::{
    PipelineError,
    pose::{PoseCandidate, candidate_priority, fit_pose_hybrid, refine_seed_eye_2pt},
    surface::{Surface, SurfaceGpuRenderer, SurfaceRayHit},
    xfeat_runtime::LoadedXFeat,
};
use image::DynamicImage;
use nalgebra::Matrix3;
use navigate_match::{
    FastMatchConfig, HadamardProjector, IndexedMatch, LighterGlue, PrecomputedMatch,
    PrecomputedReferenceBank, PrecomputedViewEntry, PreparedKashinQuery, SparseFeatures, Weights,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path, sync::Mutex};

#[expect(
    clippy::too_many_arguments,
    reason = "sparse pose polish accepts surface, optional GPU renderer, feature models, query keypoints, and camera geometry"
)]
pub(crate) fn polish_pose_sparse(
    surface: &Surface,
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    xfeat: &LoadedXFeat,
    lighterglue: Option<&LighterGlue>,
    qry_feat: &SparseFeatures,
    pose: ([f64; 3], Matrix3<f64>),
    k_render: &Matrix3<f64>,
    render_size: [u32; 2],
) -> Result<Option<(PoseCandidate, usize, usize)>, PipelineError> {
    if !pose.0.iter().all(|v| v.is_finite()) || pose.0[0].hypot(pose.0[1]) > 95_000.0 {
        return Ok(None);
    }
    let kp_cap = qry_feat.len().clamp(1024, 1536);
    let mut gpu_view_opt = None;
    if let Some(lock) = gpu_renderer
        && let Ok(mut guard) = lock.lock()
    {
        if guard.is_none() {
            *guard = SurfaceGpuRenderer::try_new(surface, k_render, render_size);
        }
        if let Some(gpu) = guard.as_mut() {
            gpu_view_opt = gpu
                .render_view(surface, pose.0, &pose.1, k_render, render_size)
                .ok();
        }
    }

    let (ref_gray, pre_rays) = if let Some(ref gv) = gpu_view_opt {
        (gv.image.clone(), 0_usize)
    } else {
        let ((ref_rgb, _, _), render_stats) =
            surface.render_adaptive(pose.0, &pose.1, k_render, render_size, 4, 8, 0.025);
        (
            DynamicImage::ImageRgb8(ref_rgb).into_luma8(),
            render_stats.rays_cast,
        )
    };

    let raw_ref = xfeat.extract(&ref_gray, kp_cap)?;
    let (ref_feat, pre_world) = if let Some(ref gv) = gpu_view_opt {
        let unproj_all = gv.unproject_pixels(surface, &raw_ref.pixels);
        let (filtered, world) = filter_terrain_world_points(&raw_ref, &unproj_all);
        if filtered.len() < 16 {
            return Ok(None);
        }
        (filtered, Some(world))
    } else {
        (raw_ref, None)
    };
    let (matches, layers) = if let Some(lg) = lighterglue {
        let cfg = FastMatchConfig {
            max_keypoints: kp_cap.min(1024),
            min_layers: 6,
            max_layers: 6,
            exit_check_layers: 0,
            prune_after_layer: None,
            ..FastMatchConfig::default()
        };
        let mut batched =
            pollster::block_on(lg.match_features_batch_indexed_fast(&[&ref_feat], qry_feat, &cfg))?;
        let (idx_m, stats) = batched.pop().unwrap_or_default();
        let m = idx_m.into_iter().map(|im| im.pixel_match).collect();
        (m, stats.layers_executed)
    } else {
        (ref_feat.mutual_matches(qry_feat), 0)
    };
    let ref_pixels: Vec<[f64; 2]> = matches
        .iter()
        .map(|m| [m.reference.x, m.reference.y])
        .collect();

    let mut world_pts = Vec::with_capacity(matches.len());
    let mut img_px = Vec::with_capacity(matches.len());
    let rays_used = if let Some(gv) = gpu_view_opt {
        let _ = pre_world;
        let unproj = gv.unproject_pixels(surface, &ref_pixels);
        for (m, w_opt) in matches.iter().zip(unproj) {
            if let Some(w) = w_opt {
                world_pts.push(w);
                img_px.push([m.query.x, m.query.y]);
            }
        }
        0_usize
    } else {
        let hits = surface.raycast_pixels(pose.0, &pose.1, k_render, &ref_pixels);
        for (m, hit) in matches.iter().zip(hits) {
            if let Some(h) = hit {
                world_pts.push(h.world_enu);
                img_px.push([m.query.x, m.query.y]);
            }
        }
        pre_rays + ref_pixels.len()
    };

    let mut cand = fit_pose_hybrid(&world_pts, &img_px, k_render, render_size, Some(pose));
    if !cand.as_ref().is_some_and(|c| c.score.geometry_accepted)
        && let Some(tri_init) = refine_seed_eye_2pt(&world_pts, &img_px, pose, k_render)
        && let Some(tri_cand) =
            fit_pose_hybrid(&world_pts, &img_px, k_render, render_size, Some(tri_init))
        && cand.as_ref().is_none_or(|prev| {
            candidate_priority(&tri_cand.score) > candidate_priority(&prev.score)
        })
    {
        cand = Some(tri_cand);
    }
    Ok(cand.map(|c| (c, rays_used, layers)))
}

pub(crate) fn filter_terrain_features(
    raw: &SparseFeatures,
    hits: &[Option<SurfaceRayHit>],
) -> (SparseFeatures, Vec<[f64; 3]>) {
    filter_terrain_world_points(
        raw,
        &hits
            .iter()
            .map(|h| h.map(|x| x.world_enu))
            .collect::<Vec<_>>(),
    )
}

pub(crate) fn filter_terrain_world_points(
    raw: &SparseFeatures,
    world_opts: &[Option<[f64; 3]>],
) -> (SparseFeatures, Vec<[f64; 3]>) {
    let mut out = SparseFeatures::empty();
    let mut world = Vec::new();
    for (i, w_opt) in world_opts.iter().enumerate() {
        if let Some(w) = *w_opt {
            out.pixels.push(raw.pixels[i]);
            out.model_pixels.push(raw.model_pixels[i]);
            out.scores.push(raw.scores[i]);
            out.descriptors
                .extend_from_slice(&raw.descriptors[i * 64..(i + 1) * 64]);
            world.push(w);
        }
    }
    (out, world)
}

pub(super) fn collect_bank_correspondences(
    matches: &[PrecomputedMatch],
) -> (Vec<[f64; 3]>, Vec<[f64; 2]>) {
    let mut world = Vec::with_capacity(matches.len());
    let mut px = Vec::with_capacity(matches.len());
    for m in matches {
        if let Some(w) = m.world_point_enu {
            world.push(w);
            px.push(m.query_pixel);
        }
    }
    (world, px)
}

pub(super) fn lookup_indexed_3d_matches(
    entry: &PrecomputedViewEntry,
    matches: &[IndexedMatch],
) -> (Vec<[f64; 3]>, Vec<[f64; 2]>) {
    let mut world = Vec::with_capacity(matches.len());
    let mut px = Vec::with_capacity(matches.len());
    for m in matches {
        if let Some(&w) = entry.world_points_enu.get(m.ref_index) {
            world.push(w);
            px.push([m.pixel_match.query.x, m.pixel_match.query.y]);
        }
    }
    (world, px)
}

pub(crate) fn log_attempt(
    log: &mut Vec<Value>,
    backend: &str,
    seed: usize,
    step: usize,
    sketch_score: f32,
    pairs: usize,
    cand: Option<&PoseCandidate>,
) {
    if let Some(c) = cand {
        log.push(json!({
            "backend": backend,
            "seed": seed,
            "step": step,
            "sketch_score": sketch_score,
            "pairs": c.score.pairs,
            "inliers": c.score.inliers,
            "train_inliers": c.score.train_inliers,
            "train_rms_px": c.score.train_rms_px,
            "holdout_count": c.score.holdout_count,
            "holdout_inlier_fraction": c.score.holdout_inlier_fraction,
            "holdout_median_px": c.score.holdout_median_px,
            "geometry_accepted": c.score.geometry_accepted,
        }));
    } else {
        log.push(json!({
            "backend": backend,
            "seed": seed,
            "step": step,
            "sketch_score": sketch_score,
            "pairs": pairs,
            "inliers": 0,
            "geometry_accepted": false,
        }));
    }
}

pub(crate) fn raw_file_sha256(path: &Path) -> Result<[u8; 32], PipelineError> {
    let bytes = fs::read(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(Sha256::digest(&bytes).into())
}

/// Check a bank `encoder_contract` stamp against the loaded `XFeat` extractor.
pub(crate) fn check_encoder_contract(
    bank_kind: &str,
    stamp: &[u8; 32],
    xfeat: &LoadedXFeat,
) -> Result<(), PipelineError> {
    if xfeat.accepts_encoder_contract(stamp) {
        return Ok(());
    }
    Err(PipelineError::Invalid(format!(
        "{bank_kind} encoder_contract {} is neither the XFeat encoder contract {} nor the --model file digest {}",
        hex(stamp),
        hex(&xfeat.encoder_contract()),
        hex(xfeat.weights_sha256()),
    )))
}

/// Lowercase hex form of a SHA-256 digest for reports and error messages.
pub(crate) fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

pub(crate) fn load_lighterglue_model(
    path: &Path,
    device: &burn::tensor::Device,
) -> Result<LighterGlue, PipelineError> {
    let bytes = fs::read(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let weights = Weights::from_safetensors(&bytes)?;
    Ok(LighterGlue::load(weights, device)?)
}

pub(crate) fn rot_to_row_major(r: &Matrix3<f64>) -> [f64; 9] {
    [
        r[(0, 0)],
        r[(0, 1)],
        r[(0, 2)],
        r[(1, 0)],
        r[(1, 1)],
        r[(1, 2)],
        r[(2, 0)],
        r[(2, 1)],
        r[(2, 2)],
    ]
}

pub(crate) fn rot_from_row_major(m: [f64; 9]) -> Matrix3<f64> {
    Matrix3::new(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8])
}

pub(super) type ShortlistMatchResult = (Option<(usize, usize, PoseCandidate)>, Vec<Value>, usize);

pub(super) fn match_shortlisted_views(
    bank: &PrecomputedReferenceBank,
    projector: &HadamardProjector,
    lighterglue: Option<&LighterGlue>,
    qry_feat: &SparseFeatures,
    prepared: &PreparedKashinQuery,
    ranked: &[(usize, f32)],
    (k_render, render_size, kp_limit): (&Matrix3<f64>, [u32; 2], usize),
) -> Result<ShortlistMatchResult, PipelineError> {
    let kashin_limit = if lighterglue.is_some() {
        640
    } else {
        usize::MAX
    };
    let mut attempts_log = Vec::new();
    let mut best: Option<(usize, usize, PoseCandidate)> = None;
    let mut kashin_cands = Vec::with_capacity(ranked.len());

    for &(entry_idx, sketch_score) in ranked {
        let entry = &bank.entries[entry_idx];
        let init_eye = entry.eye_enu;
        let init_rot = rot_from_row_major(entry.world_to_cam);
        let kashin_matches =
            bank.match_prepared_limit(entry_idx, prepared, kashin_limit, 24, 0.60, 0.92)?;
        let (world_pts, img_px) = collect_bank_correspondences(&kashin_matches);
        let cand = fit_pose_hybrid(
            &world_pts,
            &img_px,
            k_render,
            render_size,
            Some((init_eye, init_rot)),
        );
        if let Some(ref c) = cand
            && c.score.geometry_accepted
        {
            log_attempt(
                &mut attempts_log,
                "kashin-i8",
                entry_idx,
                0,
                sketch_score,
                world_pts.len(),
                cand.as_ref(),
            );
            return Ok((Some((entry_idx, 0, c.clone())), attempts_log, 0));
        }
        kashin_cands.push((entry_idx, sketch_score, world_pts.len(), cand));
    }

    let mut lg_layers_total = 0usize;
    if let Some(lg) = lighterglue
        && !kashin_cands.is_empty()
    {
        let ref_owned: Vec<SparseFeatures> = kashin_cands
            .iter()
            .map(|&(idx, _, _, _)| bank.entries[idx].to_sparse_features(projector))
            .collect();
        let ref_refs: Vec<&SparseFeatures> = ref_owned.iter().collect();
        let fast_cfg = FastMatchConfig {
            max_keypoints: kp_limit,
            ..FastMatchConfig::balanced_search()
        };
        let batched = pollster::block_on(
            lg.match_features_batch_indexed_fast(&ref_refs, qry_feat, &fast_cfg),
        )?;
        if let Some((_, first_stats)) = batched.first() {
            lg_layers_total += first_stats.layers_executed;
        }
        for ((entry_idx, sketch_score, mut pairs_n, mut cand), (lg_matches, _)) in
            kashin_cands.into_iter().zip(batched)
        {
            let entry = &bank.entries[entry_idx];
            let init_eye = entry.eye_enu;
            let init_rot = rot_from_row_major(entry.world_to_cam);
            let mut label = "kashin-i8";
            let (lg_world, lg_px) = lookup_indexed_3d_matches(entry, &lg_matches);
            if let Some(lg_cand) = fit_pose_hybrid(
                &lg_world,
                &lg_px,
                k_render,
                render_size,
                Some((init_eye, init_rot)),
            ) && cand.as_ref().is_none_or(|prev| {
                candidate_priority(&lg_cand.score) > candidate_priority(&prev.score)
            }) {
                pairs_n = lg_world.len();
                label = "kashin-bank+lighterglue-batch";
                cand = Some(lg_cand);
            }
            log_attempt(
                &mut attempts_log,
                label,
                entry_idx,
                0,
                sketch_score,
                pairs_n,
                cand.as_ref(),
            );
            if let Some(c) = cand {
                let is_acc = c.score.geometry_accepted;
                if best.as_ref().is_none_or(|(_, _, prev)| {
                    candidate_priority(&c.score) > candidate_priority(&prev.score)
                }) {
                    best = Some((entry_idx, 0, c));
                }
                if is_acc {
                    break;
                }
            }
        }
    } else {
        for (entry_idx, sketch_score, pairs_n, cand) in kashin_cands {
            log_attempt(
                &mut attempts_log,
                "kashin-i8",
                entry_idx,
                0,
                sketch_score,
                pairs_n,
                cand.as_ref(),
            );
            if let Some(c) = cand
                && best.as_ref().is_none_or(|(_, _, prev)| {
                    candidate_priority(&c.score) > candidate_priority(&prev.score)
                })
            {
                best = Some((entry_idx, 0, c));
            }
        }
    }
    Ok((best, attempts_log, lg_layers_total))
}
