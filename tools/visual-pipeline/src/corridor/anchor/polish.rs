//! GPU and sparse terrain polish for candidate anchor seed pools.

use super::{
    is_plausible_anchor_candidate, is_sane_anchor_seed,
    nadir_screen::{SeedPose, select_nadir_polish_fallbacks},
};
use crate::{
    PipelineError,
    pose::{PoseCandidate, PoseScore, candidate_priority},
    realtime::helpers::{log_attempt, polish_pose_sparse},
    surface::{Surface, SurfaceGpuRenderer},
    xfeat_runtime::LoadedXFeat,
};
use nalgebra::Matrix3;
use navigate_match::{LighterGlue, SparseFeatures};
use std::sync::Mutex;

fn seed_only_candidate(
    eye: [f64; 3],
    rotation: Matrix3<f64>,
    k_render: &Matrix3<f64>,
    render_size: [u32; 2],
) -> PoseCandidate {
    PoseCandidate {
        eye,
        rotation,
        k: *k_render,
        size: render_size,
        points: Vec::new(),
        pixels: Vec::new(),
        inliers: Vec::new(),
        score: PoseScore {
            focal_px: k_render[(0, 0)],
            geometry_accepted: false,
            holdout_count: 0,
            holdout_inlier_fraction: 0.0,
            holdout_median_px: 1e6,
            image_support_fraction: 0.0,
            inliers: 0,
            pairs: 0,
            positive_depth_fraction: 0.0,
            train_inliers: 0,
            train_rms_px: 1e6,
        },
    }
}

fn same_pose_neighborhood(
    a_eye: [f64; 3],
    a_rot: &Matrix3<f64>,
    b_eye: [f64; 3],
    b_rot: &Matrix3<f64>,
) -> bool {
    let horiz = (a_eye[0] - b_eye[0]).hypot(a_eye[1] - b_eye[1]);
    let cos_ang = (((a_rot * b_rot.transpose()).trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
    horiz <= 25.0 && cos_ang >= 0.94
}

/// Rank intermediate seed candidates for reproject and GPU polish by holdout coherence first.
pub(super) fn seed_refinement_priority(
    score: &PoseScore,
) -> (bool, bool, bool, usize, usize, (bool, usize, i64, i64)) {
    let hold_inl = ((score.holdout_count as f64) * score.holdout_inlier_fraction).round() as usize;
    let coherent_holdout = score.inliers >= 7
        && (hold_inl >= 1 || score.holdout_median_px <= 45.0 || score.inliers >= 14);
    let tight_train = score.inliers >= 7 && score.train_rms_px <= 7.8;
    (
        score.geometry_accepted,
        coherent_holdout,
        tight_train,
        score.inliers,
        hold_inl,
        candidate_priority(score),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "anchor polish loop shares surface, GPU renderer, feature models, intrinsics, and best-candidate state"
)]
pub(super) fn polish_anchor_seed_pool(
    surface: &Surface,
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    xfeat: &LoadedXFeat,
    lighterglue: Option<&LighterGlue>,
    qry_feat: &SparseFeatures,
    k_render: &Matrix3<f64>,
    render_size: [u32; 2],
    focal_mult: f64,
    mut seed_pool: Vec<(usize, PoseCandidate)>,
    (screened_seeds, closest_nadir): (Option<&[SeedPose]>, Option<SeedPose>),
    best: &mut Option<(usize, usize, PoseCandidate)>,
    attempts_log: &mut Vec<serde_json::Value>,
) -> Result<(usize, usize), PipelineError> {
    seed_pool.sort_by_key(|a| std::cmp::Reverse(seed_refinement_priority(&a.1.score)));
    let best_accepted = best.as_ref().is_some_and(|(_, _, c)| {
        c.score.geometry_accepted && is_plausible_anchor_candidate(c, surface)
    });
    let mut selected: Vec<(usize, PoseCandidate)> =
        if best_accepted {
            seed_pool.into_iter().take(1).collect()
        } else if let Some(scr) = screened_seeds {
            let mut top: Vec<(usize, PoseCandidate)> = Vec::with_capacity(3);
            let mut has_coherent = false;
            for (idx, cand) in seed_pool {
                if cand.score.inliers < 6 {
                    continue;
                }
                if !top.iter().any(|(_, c)| {
                    same_pose_neighborhood(c.eye, &c.rotation, cand.eye, &cand.rotation)
                }) {
                    has_coherent |= seed_refinement_priority(&cand.score).1;
                    top.push((idx, cand));
                    if top.len() >= 2 {
                        break;
                    }
                }
            }
            let fb_cap = if has_coherent { 2 } else { 3 };
            for (idx, (s_eye, s_rot)) in select_nadir_polish_fallbacks(scr, closest_nadir) {
                if top.len() >= fb_cap {
                    break;
                }
                if !top
                    .iter()
                    .any(|(_, c)| same_pose_neighborhood(c.eye, &c.rotation, s_eye, &s_rot))
                {
                    top.push((
                        idx,
                        seed_only_candidate(s_eye, s_rot, k_render, render_size),
                    ));
                }
            }
            top
        } else {
            let mut top: Vec<(usize, PoseCandidate)> = Vec::with_capacity(2);
            for (idx, cand) in seed_pool {
                if !top.iter().any(|(_, c)| {
                    same_pose_neighborhood(c.eye, &c.rotation, cand.eye, &cand.rotation)
                }) {
                    top.push((idx, cand));
                    if top.len() >= 2 {
                        break;
                    }
                }
            }
            top
        };

    let mut visited: Vec<([f64; 3], Matrix3<f64>)> = Vec::with_capacity(8);
    let mut polish_rays = 0_usize;
    let mut lg_layers = 0_usize;
    for (seed_idx, mut cur_cand) in selected.drain(..) {
        if visited
            .iter()
            .any(|&(ve, ref vr)| same_pose_neighborhood(cur_cand.eye, &cur_cand.rotation, ve, vr))
        {
            continue;
        }
        visited.push((cur_cand.eye, cur_cand.rotation));
        let steps = if cur_cand.score.geometry_accepted {
            1
        } else {
            3
        };
        for p_step in 1..=steps {
            let Some((polished, rays, layers)) = polish_pose_sparse(
                surface,
                gpu_renderer,
                xfeat,
                lighterglue,
                qry_feat,
                (cur_cand.eye, cur_cand.rotation),
                k_render,
                render_size,
            )?
            else {
                break;
            };
            polish_rays += rays;
            lg_layers += layers;
            log_attempt(
                attempts_log,
                "corridor-polish-gpu-or-sparse+l1-gnc",
                seed_idx,
                p_step + 1,
                focal_mult as f32,
                polished.score.pairs,
                Some(&polished),
            );
            let polished_inliers = polished.score.inliers;
            let acc_now = polished.score.geometry_accepted
                && is_plausible_anchor_candidate(&polished, surface);
            let mut improved_cur = false;
            if is_sane_anchor_seed(&polished, surface) {
                let beat_cur = candidate_priority(&polished.score)
                    > candidate_priority(&cur_cand.score)
                    || (p_step == 1 && cur_cand.score.inliers < 12 && polished_inliers >= 8);
                if beat_cur {
                    cur_cand = polished.clone();
                    improved_cur = true;
                }
                if best.as_ref().is_none_or(|(_, _, prev)| {
                    let prev_acc = prev.score.geometry_accepted
                        && is_plausible_anchor_candidate(prev, surface);
                    (acc_now, candidate_priority(&polished.score))
                        >= (prev_acc, candidate_priority(&prev.score))
                }) {
                    *best = Some((seed_idx, p_step + 1, polished));
                }
            }
            if acc_now
                || !improved_cur
                || (p_step == 1 && polished_inliers < 8)
                || (p_step == 2 && polished_inliers < 18)
            {
                break;
            }
        }
        visited.push((cur_cand.eye, cur_cand.rotation));
        if best.as_ref().is_some_and(|(_, _, c)| {
            c.score.geometry_accepted && is_plausible_anchor_candidate(c, surface)
        }) {
            break;
        }
    }
    Ok((polish_rays, lg_layers))
}

#[expect(
    clippy::too_many_arguments,
    reason = "stage-end top-seed polish shares surface, GPU renderer, feature models, intrinsics, and best-candidate state"
)]
pub(super) fn polish_stage_top_seed(
    surface: &Surface,
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, lighterglue): (&LoadedXFeat, Option<&LighterGlue>),
    qry_feat: &SparseFeatures,
    (k_render, render_size, focal_mult): (&Matrix3<f64>, [u32; 2], f64),
    seed_pool: &[(usize, PoseCandidate)],
    best: &mut Option<(usize, usize, PoseCandidate)>,
    lg_layers: &mut usize,
    attempts_log: &mut Vec<serde_json::Value>,
) -> Result<bool, PipelineError> {
    if gpu_renderer.is_none() {
        return Ok(false);
    }
    let Some(top_seed) = seed_pool
        .iter()
        .filter(|(_, c)| c.score.inliers >= 6)
        .max_by_key(|(_, c)| seed_refinement_priority(&c.score))
        .cloned()
    else {
        return Ok(false);
    };
    let (_, p_layers) = polish_anchor_seed_pool(
        surface,
        gpu_renderer,
        xfeat,
        lighterglue,
        qry_feat,
        k_render,
        render_size,
        focal_mult,
        vec![top_seed],
        (None, None),
        best,
        attempts_log,
    )?;
    *lg_layers += p_layers;
    Ok(best.as_ref().is_some_and(|(_, _, c)| {
        c.score.geometry_accepted && is_plausible_anchor_candidate(c, surface)
    }))
}
