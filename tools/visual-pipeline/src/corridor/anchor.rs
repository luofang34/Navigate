//! Virtual-view map-anchor acquisition, focal-multiplier sweeps, and global place fallback.

mod eval;
pub mod multitier;
mod nadir_screen;
pub mod plausibility;
mod sweep;

use super::CorridorTrackOptions;
use crate::{
    PipelineError,
    pose::{PoseCandidate, candidate_priority},
    realtime::helpers::log_attempt,
    stages::{MatchStageOptions, has_calibrated_focal_prior, load_horizon_attitude},
    surface::{Surface, SurfaceGpuRenderer},
    testset::PhotoCase,
    xfeat_runtime::LoadedXFeat,
};
use eval::{SeedBatchFlags, eval_global_fallback_seeds, eval_seed_batch, polish_anchor_seed_pool};
use image::GrayImage;
use nadir_screen::{
    SeedPose, build_anchor_seeds, closest_prior_nadir_seed, maybe_screen_low_agl_nadir_seeds,
};
use nalgebra::{Matrix3, Vector3};
use navigate_match::{
    CorridorTileStore, HadamardProjector, LighterGlue, PreparedKashinQuery, SparseFeatures,
    VirtualCorridorView,
};
use plausibility::{PhysicalPlausibilityConfig, is_plausible_candidate, is_sane_seed_candidate};
use serde_json::{Value, json};
use std::{path::PathBuf, sync::Mutex};

const ANCHOR_BATCH_SIZE: usize = 4;

pub(super) struct AnchorAcquisition {
    pub best: Option<(usize, usize, PoseCandidate)>,
    pub lg_layers: usize,
    pub polish_rays: usize,
    pub best_view: Option<VirtualCorridorView>,
    pub locked_focal_px: Option<f64>,
}

pub(super) struct AnchorQueryAux<'a> {
    pub pred_pose: Option<([f64; 3], Matrix3<f64>)>,
    pub live_attitude: Option<(f64, f64)>,
    pub mask_small: Option<&'a GrayImage>,
    pub search_tier_matcher: Option<&'a multitier::SearchTierMatcher>,
    pub focal_locked: bool,
}

pub(super) fn scale_focal_matrix(k: &Matrix3<f64>, mult: f64) -> Matrix3<f64> {
    Matrix3::new(
        k[(0, 0)] * mult,
        0.0,
        k[(0, 2)],
        0.0,
        k[(1, 1)] * mult,
        k[(1, 2)],
        0.0,
        0.0,
        1.0,
    )
}

pub(super) fn scale_horizon_attitude(att: Option<(f64, f64)>, mult: f64) -> Option<(f64, f64)> {
    att.map(|(down_deg, roll_deg)| {
        let scaled_down = (down_deg.to_radians().tan() / mult.max(0.1))
            .atan()
            .to_degrees()
            .clamp(-45.0, 89.0);
        (scaled_down, roll_deg)
    })
}

pub(super) fn stage_match_options(opts: &CorridorTrackOptions) -> MatchStageOptions {
    MatchStageOptions {
        dataset_dir: opts.dataset_dir.clone(),
        profile_path: opts.profile_path.clone(),
        catalog_path: opts.catalog_path.clone(),
        context_catalog: None,
        output_dir: PathBuf::new(),
        attempts: opts.anchor_attempts.max(1),
        steps: 1,
        heading_step_deg: opts.heading_step_deg,
        height_factors: if opts.height_factors.is_empty() {
            vec![1.0]
        } else {
            opts.height_factors.clone()
        },
        only_cases: None,
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "anchor acquisition borrows terrain, tile store, optional GPU renderer, feature models, query buffers, and camera intrinsics"
)]
pub(super) fn acquire_corridor_anchor(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, lighterglue): (&LoadedXFeat, Option<&LighterGlue>),
    (case, opts): (&PhotoCase, &CorridorTrackOptions),
    (qry_feat, prepared): (&SparseFeatures, &PreparedKashinQuery),
    (k_full, k_render, render_size): (&Matrix3<f64>, &Matrix3<f64>, [u32; 2]),
    aux: AnchorQueryAux<'_>,
    attempts_log: &mut Vec<Value>,
) -> Result<AnchorAcquisition, PipelineError> {
    let run_focal_sweep =
        !has_calibrated_focal_prior(case) && aux.pred_pose.is_none() && !aux.focal_locked;
    if !run_focal_sweep {
        return acquire_corridor_anchor_single(
            (surface, store, projector),
            gpu_renderer,
            (xfeat, lighterglue),
            (case, opts),
            (qry_feat, prepared),
            (k_full, k_render, render_size),
            &aux,
            1.0,
            attempts_log,
        );
    }

    sweep::run_uncalibrated_focal_sweep(
        (surface, store, projector),
        gpu_renderer,
        (xfeat, lighterglue),
        (case, opts),
        (qry_feat, prepared),
        (k_full, k_render, render_size),
        &aux,
        attempts_log,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "single-focal anchor pass shares query buffers and optional global retrieval state"
)]
fn acquire_corridor_anchor_single(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, lighterglue): (&LoadedXFeat, Option<&LighterGlue>),
    (case, opts): (&PhotoCase, &CorridorTrackOptions),
    (qry_feat, prepared): (&SparseFeatures, &PreparedKashinQuery),
    (k_full, k_render, render_size): (&Matrix3<f64>, &Matrix3<f64>, [u32; 2]),
    aux: &AnchorQueryAux<'_>,
    focal_mult: f64,
    attempts_log: &mut Vec<Value>,
) -> Result<AnchorAcquisition, PipelineError> {
    let stage_opts = stage_match_options(opts);
    let scaled_live = scale_horizon_attitude(aux.live_attitude, focal_mult);
    let attitude = load_horizon_attitude(&opts.dataset_dir, case, k_full).or(scaled_live);
    let intrinsics = [
        k_render[(0, 0)],
        k_render[(1, 1)],
        k_render[(0, 2)],
        k_render[(1, 2)],
    ];
    let screened = if aux.pred_pose.is_none() && attitude.is_none() {
        maybe_screen_low_agl_nadir_seeds(
            (surface, store, projector),
            case,
            prepared,
            intrinsics,
            render_size,
            opts.anchor_attempts,
        )?
    } else {
        None
    };
    let closest_nadir = screened
        .as_ref()
        .and_then(|s| closest_prior_nadir_seed(surface, case, s.first().copied()));
    let seeds = screened.as_ref().cloned().unwrap_or_else(|| {
        build_anchor_seeds(
            surface,
            case,
            &stage_opts,
            attitude,
            aux.pred_pose,
            opts.anchor_attempts,
        )
    });
    let max_lm = opts.query_keypoints.clamp(256, 2048);
    let mut best: Option<(usize, usize, PoseCandidate)> = None;
    let mut best_view: Option<VirtualCorridorView> = None;
    let mut seed_pool: Vec<(usize, PoseCandidate)> = Vec::new();
    let mut lg_layers = 0_usize;
    let mut base_idx = 0_usize;

    eval_seed_chunks(
        (surface, store, projector),
        gpu_renderer,
        (xfeat, lighterglue),
        (qry_feat, prepared),
        &seeds,
        (intrinsics, k_render, render_size, max_lm),
        (aux.pred_pose.is_some(), false, true, focal_mult),
        (&mut base_idx, &mut lg_layers),
        (&mut seed_pool, &mut best, &mut best_view),
        attempts_log,
    )?;

    eval_global_fallback_if_needed(
        (surface, store, projector),
        gpu_renderer,
        (xfeat, lighterglue),
        (case, opts),
        (qry_feat, prepared),
        (intrinsics, k_render, render_size, max_lm),
        aux,
        attitude,
        (&mut base_idx, &mut lg_layers),
        (&mut seed_pool, &mut best, &mut best_view),
    )?;

    let mut polish_rays = 0_usize;
    if opts.polish_anchors {
        let (rays, p_layers) = polish_anchor_seed_pool(
            surface,
            gpu_renderer,
            xfeat,
            lighterglue,
            qry_feat,
            k_render,
            render_size,
            focal_mult,
            seed_pool,
            screened.as_deref(),
            closest_nadir,
            &mut best,
            attempts_log,
        )?;
        polish_rays += rays;
        lg_layers += p_layers;
    }
    if let Some((_, _, ref mut cand)) = best
        && !is_plausible_anchor_candidate(cand, surface)
    {
        cand.score.geometry_accepted = false;
    }
    Ok(AnchorAcquisition {
        best,
        lg_layers,
        polish_rays,
        best_view,
        locked_focal_px: None,
    })
}

type SeedStateMut<'a> = (
    &'a mut Vec<(usize, PoseCandidate)>,
    &'a mut Option<(usize, usize, PoseCandidate)>,
    &'a mut Option<VirtualCorridorView>,
);

#[expect(
    clippy::too_many_arguments,
    reason = "seed-chunk evaluation updates shared pool, best candidate, and telemetry counters"
)]
pub(super) fn eval_seed_chunks(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, lighterglue): (&LoadedXFeat, Option<&LighterGlue>),
    (qry_feat, prepared): (&SparseFeatures, &PreparedKashinQuery),
    seeds: &[SeedPose],
    (intrinsics, k_render, render_size, max_lm): ([f64; 4], &Matrix3<f64>, [u32; 2], usize),
    (has_pred_lead, coarse_screen, allow_horizon_gpu, focal_mult): (bool, bool, bool, f64),
    (base_idx, lg_layers): (&mut usize, &mut usize),
    (seed_pool, best, best_view): SeedStateMut<'_>,
    attempts_log: &mut Vec<Value>,
) -> Result<usize, PipelineError> {
    if seeds.is_empty() {
        return Ok(0);
    }
    let (lead_count, rest_offset) = if has_pred_lead && *base_idx == 0 {
        (1_usize, 1_usize)
    } else {
        (0_usize, 0_usize)
    };
    let mut chunks: Vec<&[SeedPose]> = Vec::new();
    if lead_count == 1 {
        chunks.push(&seeds[..1]);
    }
    for ch in seeds[rest_offset..].chunks(ANCHOR_BATCH_SIZE) {
        chunks.push(ch);
    }

    let max_re_steps = if coarse_screen { 1_usize } else { 2_usize };
    let mut max_pairs = 0_usize;
    for chunk in chunks {
        let fast_exit = has_pred_lead && *base_idx == 0 && chunk.len() == 1;
        let evals = eval_seed_batch(
            (surface, store, projector),
            gpu_renderer,
            (xfeat, lighterglue),
            (qry_feat, prepared),
            chunk,
            (intrinsics, k_render, render_size, max_lm),
            SeedBatchFlags {
                fast_exit,
                coarse_screen,
                allow_horizon_gpu,
            },
        )?;
        let mut accepted_early = false;
        for (local_i, (cand_opt, virt, layers)) in evals.into_iter().enumerate() {
            let seed_idx = *base_idx + local_i;
            *lg_layers += layers;
            let pairs = cand_opt.as_ref().map_or(0, |c| c.score.pairs);
            max_pairs = max_pairs.max(pairs);
            log_attempt(
                attempts_log,
                "corridor-virtual-view+l1-gnc",
                seed_idx,
                0,
                focal_mult as f32,
                pairs,
                cand_opt.as_ref(),
            );
            if let Some(last) = attempts_log.last_mut().and_then(Value::as_object_mut) {
                last.insert("virtual_landmarks".into(), json!(virt.features.len()));
                last.insert("visible_tiles".into(), json!(virt.visible_tiles));
                last.insert("focal_multiplier".into(), json!(focal_mult));
            }

            let hyp_eye = chunk[local_i].0;
            let mut cur_cand = cand_opt.filter(|c| is_sane_anchor_seed(c, surface));
            let mut cur_virt = virt;
            let mut prev_eye = hyp_eye;
            for re_step in 1..=max_re_steps {
                let Some(ref c_prev) = cur_cand else {
                    break;
                };
                let shift_m = (Vector3::from(c_prev.eye) - Vector3::from(prev_eye)).norm();
                let prev_plausible = c_prev.score.geometry_accepted
                    && is_plausible_anchor_candidate(c_prev, surface);
                let weak_seven = c_prev.score.inliers < 8 && c_prev.score.train_rms_px > 7.0;
                if c_prev.score.inliers < 7 || weak_seven || (prev_plausible && shift_m <= 120.0) {
                    break;
                }
                prev_eye = c_prev.eye;
                let mut single = eval_seed_batch(
                    (surface, store, projector),
                    gpu_renderer,
                    (xfeat, lighterglue),
                    (qry_feat, prepared),
                    &[(c_prev.eye, c_prev.rotation)],
                    (intrinsics, k_render, render_size, max_lm),
                    SeedBatchFlags {
                        fast_exit: false,
                        coarse_screen: false,
                        allow_horizon_gpu,
                    },
                )?;
                let (re_cand, re_virt, re_layers) =
                    single.pop().unwrap_or((None, cur_virt.clone(), 0));
                *lg_layers += re_layers;
                let re_cand = re_cand.filter(|c| is_sane_anchor_seed(c, surface));
                let re_pairs = re_cand.as_ref().map_or(0, |c| c.score.pairs);
                max_pairs = max_pairs.max(re_pairs);
                log_attempt(
                    attempts_log,
                    "corridor-virtual-reproject+l1-gnc",
                    seed_idx,
                    re_step,
                    focal_mult as f32,
                    re_pairs,
                    re_cand.as_ref(),
                );
                if let Some(rc) = re_cand
                    && candidate_priority(&rc.score) > candidate_priority(&c_prev.score)
                {
                    cur_cand = Some(rc);
                    cur_virt = re_virt;
                } else {
                    break;
                }
            }

            if let Some(c) = cur_cand {
                let plausible = is_plausible_anchor_candidate(&c, surface);
                let is_acc = c.score.geometry_accepted && plausible;
                let strong_coarse = coarse_screen
                    && c.score.inliers >= 28
                    && c.score.holdout_inlier_fraction >= 0.50
                    && plausible;
                let take = best.as_ref().is_none_or(|(_, _, prev)| {
                    let prev_acc = prev.score.geometry_accepted
                        && is_plausible_anchor_candidate(prev, surface);
                    (is_acc, candidate_priority(&c.score))
                        > (prev_acc, candidate_priority(&prev.score))
                });
                seed_pool.push((seed_idx, c.clone()));
                if take {
                    *best = Some((seed_idx, 0, c));
                    *best_view = Some(cur_virt);
                }
                if is_acc || strong_coarse {
                    accepted_early = true;
                    break;
                }
            }
        }
        *base_idx += chunk.len();
        if accepted_early {
            break;
        }
    }
    Ok(max_pairs)
}

#[expect(
    clippy::too_many_arguments,
    reason = "global retrieval fallback conditionally runs SearchTierMatcher when virtual-view search is unaccepted"
)]
pub(super) fn eval_global_fallback_if_needed(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, lighterglue): (&LoadedXFeat, Option<&LighterGlue>),
    (case, opts): (&PhotoCase, &CorridorTrackOptions),
    (qry_feat, prepared): (&SparseFeatures, &PreparedKashinQuery),
    (intrinsics, k_render, render_size, max_lm): ([f64; 4], &Matrix3<f64>, [u32; 2], usize),
    aux: &AnchorQueryAux<'_>,
    attitude: Option<(f64, f64)>,
    (base_idx, lg_layers): (&mut usize, &mut usize),
    (seed_pool, best, best_view): SeedStateMut<'_>,
) -> Result<(), PipelineError> {
    if best.as_ref().is_none_or(|(_, _, c)| {
        !(c.score.geometry_accepted && is_plausible_anchor_candidate(c, surface))
    }) && let Some(matcher) = aux.search_tier_matcher
        && let Ok(rgb_img) = crate::stages::load_case_rgb(&opts.dataset_dir, &case.image)
    {
        let small_rgb = image::imageops::resize(
            &rgb_img,
            render_size[0],
            render_size[1],
            image::imageops::FilterType::Triangle,
        );
        let dyn_img = image::DynamicImage::ImageRgb8(small_rgb);
        let mask_bools = aux.mask_small.map(multitier::gray_mask_to_bools);
        let search_prior =
            multitier::build_search_prior_from_hint(aux.pred_pose, opts.anchor_uncertainty_m);
        if let Ok(fb_seeds) = multitier::resolve_global_fallback_seeds_with_mask(
            matcher,
            &search_prior,
            &dyn_img,
            mask_bools.as_deref(),
            surface,
            store,
            480.0,
            attitude,
        ) && !fb_seeds.is_empty()
        {
            eval_global_fallback_seeds(
                (surface, store, projector),
                gpu_renderer,
                (xfeat, lighterglue),
                (qry_feat, prepared),
                &fb_seeds,
                (intrinsics, k_render, render_size, max_lm),
                base_idx,
                lg_layers,
                seed_pool,
                best,
                best_view,
            )?;
        }
    }
    Ok(())
}

pub(super) fn is_sane_anchor_seed(cand: &PoseCandidate, surface: &Surface) -> bool {
    is_sane_seed_candidate(cand, surface, 5.0)
}

pub(super) fn is_plausible_anchor_candidate(cand: &PoseCandidate, surface: &Surface) -> bool {
    let config = PhysicalPlausibilityConfig::default();
    is_plausible_candidate(cand, surface, None, false, &config)
}

#[cfg(test)]
mod tests;
