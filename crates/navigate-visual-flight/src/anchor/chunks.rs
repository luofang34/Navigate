//! Priority lead-slice and broad sweep chunk evaluation with virtual reprojection.

use super::{
    SeedStateMut,
    basin::{basin_priority_key, is_unambiguously_confirmed},
    eval::{SeedBatchFlags, SeedChunkOpts, eval_seed_batch},
    is_plausible_anchor_candidate, is_sane_anchor_seed,
    polish::{is_fragile_virtual_anchor, polish_anchor_seed_pool, seed_refinement_priority},
    record_attempt,
    seeds::SeedPose,
};
use crate::{
    diagnostics::AttemptRecord,
    error::FlightVisualError,
    runtime::LoadedXFeat,
    surface::{Surface, SurfaceGpuRenderer},
};
use image::GrayImage;
use nalgebra::{Matrix3, Vector3};
use navigate_match::{
    CorridorTileStore, HadamardProjector, PreparedKashinQuery, SparseFeatures, SparseMatcher,
    VirtualCorridorView,
};
use navigate_visual::{PoseCandidate, holdout_pnp::candidate_priority};
use std::{cmp::Reverse, sync::Mutex};

const MAX_REPROJECT_SEEDS: usize = 3;

#[expect(
    clippy::too_many_arguments,
    reason = "seed-chunk evaluation updates shared pool, best candidate, and optional attempt telemetry"
)]
pub(crate) fn eval_seed_chunks(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, matcher): (&LoadedXFeat, Option<&dyn SparseMatcher>),
    (qry_feat, prepared): (&SparseFeatures, &PreparedKashinQuery),
    seeds: &[SeedPose],
    (intrinsics, k_render, render_size, max_lm): ([f64; 4], &Matrix3<f64>, [u32; 2], usize),
    chunk_opts: SeedChunkOpts<'_>,
    (base_idx, lg_layers): (&mut usize, &mut usize),
    (seed_pool, best, best_view): SeedStateMut<'_>,
    mut attempts: Option<&mut Vec<AttemptRecord>>,
) -> Result<usize, FlightVisualError> {
    if seeds.is_empty() {
        return Ok(0);
    }
    let lead_n = if *base_idx == 0 {
        chunk_opts.lead_count.min(seeds.len())
    } else {
        0_usize
    };
    let mut slices: Vec<(bool, &[SeedPose])> = Vec::with_capacity(2);
    if lead_n > 0 {
        slices.push((true, &seeds[..lead_n]));
    }
    if lead_n < seeds.len() {
        slices.push((false, &seeds[lead_n..]));
    }

    let max_re_steps = 1_usize;
    let focal_mult = chunk_opts.focal_mult;
    let mut max_pairs = 0_usize;

    for (is_lead_slice, chunk) in slices {
        let evals = eval_seed_batch(
            (surface, store, projector),
            gpu_renderer,
            (xfeat, matcher),
            (qry_feat, prepared),
            chunk,
            (intrinsics, k_render, render_size, max_lm),
            SeedBatchFlags {
                priority_lead: is_lead_slice,
                coarse_screen: chunk_opts.coarse_screen,
                allow_horizon_gpu: chunk_opts.allow_horizon_gpu,
                pace_realtime: chunk_opts.pace_realtime,
                mask_small: chunk_opts.mask_small,
            },
        )?;
        let all_steep_nadir = chunk.iter().all(|&(_, r)| -r[(2, 2)] > 0.85);
        let mut accepted_in_slice = false;
        let mut reproject_queue: Vec<(usize, [f64; 3], PoseCandidate, VirtualCorridorView)> =
            Vec::new();

        for (local_i, (cand_opt, virt, layers)) in evals.into_iter().enumerate() {
            let seed_idx = *base_idx + local_i;
            *lg_layers += layers;
            let pairs = cand_opt.as_ref().map_or(0, |c| c.score.pairs);
            max_pairs = max_pairs.max(pairs);
            if let Some(log) = attempts.as_deref_mut() {
                record_attempt(
                    log,
                    "corridor-virtual-view+l1-gnc",
                    seed_idx,
                    0,
                    focal_mult as f32,
                    pairs,
                    cand_opt.as_ref(),
                    Some((virt.features.len(), virt.visible_tiles, focal_mult)),
                );
            }

            if let Some(c) = cand_opt.filter(|c| is_sane_anchor_seed(c, surface)) {
                let plausible = is_plausible_anchor_candidate(&c, surface);
                let is_acc = c.score.geometry_accepted && plausible;
                let tight_acc = is_acc
                    && (c.score.holdout_median_px <= 4.5
                        || (!all_steep_nadir
                            && !is_fragile_virtual_anchor(&c.score)
                            && c.score.inliers >= 64
                            && c.score.train_rms_px <= 5.5));
                let paced_lead_acc = is_lead_slice
                    && chunk_opts.pace_realtime
                    && is_acc
                    && c.score.inliers >= 32
                    && c.score.holdout_inlier_fraction >= 0.35
                    && c.score.holdout_median_px <= 8.5;
                let strong_coarse = chunk_opts.coarse_screen
                    && c.score.inliers >= 28
                    && c.score.holdout_inlier_fraction >= 0.50
                    && c.score.holdout_median_px <= 4.5
                    && plausible;
                let take = best.as_ref().is_none_or(|(_, _, prev)| {
                    let prev_acc = prev.score.geometry_accepted
                        && is_plausible_anchor_candidate(prev, surface);
                    basin_priority_key(is_acc, &c) > basin_priority_key(prev_acc, prev)
                });
                seed_pool.push((seed_idx, c.clone()));
                if take {
                    *best = Some((seed_idx, 0, c.clone()));
                    *best_view = Some(virt.clone());
                }
                if tight_acc || paced_lead_acc || strong_coarse {
                    accepted_in_slice = true;
                } else {
                    let weak_steep_nadir = -c.rotation[(2, 2)] > 0.85 && c.score.inliers < 12;
                    let can_reproject = c.score.inliers >= 7
                        && !weak_steep_nadir
                        && !(c.score.inliers < 8 && c.score.train_rms_px > 7.0);
                    if can_reproject {
                        reproject_queue.push((seed_idx, chunk[local_i].0, c, virt));
                    }
                }
            }
        }

        let best_accepted = best.as_ref().is_some_and(|(_, _, c)| {
            c.score.geometry_accepted && is_plausible_anchor_candidate(c, surface)
        });
        let skip_reproject = all_steep_nadir || (chunk_opts.pace_realtime && best_accepted);
        if !accepted_in_slice && !skip_reproject && !reproject_queue.is_empty() {
            reproject_queue.sort_by_key(|(_, _, c, _)| Reverse(seed_refinement_priority(&c.score)));
            reproject_queue.truncate(MAX_REPROJECT_SEEDS);
            for (seed_idx, mut prev_eye, mut cur_cand, mut cur_virt) in reproject_queue {
                let mut improved_any = false;
                for re_step in 1..=max_re_steps {
                    let shift_m = (Vector3::from(cur_cand.eye) - Vector3::from(prev_eye)).norm();
                    let prev_plausible = cur_cand.score.geometry_accepted
                        && is_plausible_anchor_candidate(&cur_cand, surface);
                    let weak_seven =
                        cur_cand.score.inliers < 8 && cur_cand.score.train_rms_px > 7.0;
                    if cur_cand.score.inliers < 7
                        || weak_seven
                        || (prev_plausible && shift_m <= 120.0)
                    {
                        break;
                    }
                    prev_eye = cur_cand.eye;
                    let mut single = eval_seed_batch(
                        (surface, store, projector),
                        gpu_renderer,
                        (xfeat, matcher),
                        (qry_feat, prepared),
                        &[(cur_cand.eye, cur_cand.rotation)],
                        (intrinsics, k_render, render_size, max_lm),
                        SeedBatchFlags {
                            priority_lead: true,
                            coarse_screen: false,
                            allow_horizon_gpu: chunk_opts.allow_horizon_gpu,
                            pace_realtime: chunk_opts.pace_realtime,
                            mask_small: chunk_opts.mask_small,
                        },
                    )?;
                    let (re_cand, re_virt, re_layers) =
                        single.pop().unwrap_or((None, cur_virt.clone(), 0));
                    *lg_layers += re_layers;
                    let re_cand = re_cand.filter(|c| is_sane_anchor_seed(c, surface));
                    let re_pairs = re_cand.as_ref().map_or(0, |c| c.score.pairs);
                    max_pairs = max_pairs.max(re_pairs);
                    if let Some(log) = attempts.as_deref_mut() {
                        record_attempt(
                            log,
                            "corridor-virtual-reproject+l1-gnc",
                            seed_idx,
                            re_step,
                            focal_mult as f32,
                            re_pairs,
                            re_cand.as_ref(),
                            None,
                        );
                    }
                    if let Some(rc) = re_cand
                        && candidate_priority(&rc.score) > candidate_priority(&cur_cand.score)
                    {
                        cur_cand = rc;
                        cur_virt = re_virt;
                        improved_any = true;
                    } else {
                        break;
                    }
                }
                if improved_any {
                    let plausible = is_plausible_anchor_candidate(&cur_cand, surface);
                    let is_acc = cur_cand.score.geometry_accepted && plausible;
                    let tight_acc = is_acc
                        && (cur_cand.score.holdout_median_px <= 4.5
                            || (!all_steep_nadir
                                && !is_fragile_virtual_anchor(&cur_cand.score)
                                && cur_cand.score.inliers >= 64
                                && cur_cand.score.train_rms_px <= 5.5));
                    let strong_coarse = chunk_opts.coarse_screen
                        && cur_cand.score.inliers >= 28
                        && cur_cand.score.holdout_inlier_fraction >= 0.50
                        && cur_cand.score.holdout_median_px <= 4.5
                        && plausible;
                    let take = best.as_ref().is_none_or(|(_, _, prev)| {
                        let prev_acc = prev.score.geometry_accepted
                            && is_plausible_anchor_candidate(prev, surface);
                        basin_priority_key(is_acc, &cur_cand) > basin_priority_key(prev_acc, prev)
                    });
                    seed_pool.push((seed_idx, cur_cand.clone()));
                    if take {
                        *best = Some((seed_idx, 1, cur_cand));
                        *best_view = Some(cur_virt);
                    }
                    if tight_acc || strong_coarse {
                        accepted_in_slice = true;
                        break;
                    }
                }
            }
        }

        if !accepted_in_slice && !chunk_opts.coarse_screen && is_lead_slice && lead_n < seeds.len()
        {
            accepted_in_slice = polish_stage_top_seed(
                surface,
                gpu_renderer,
                (xfeat, matcher),
                qry_feat,
                (
                    k_render,
                    render_size,
                    focal_mult,
                    chunk_opts.pace_realtime,
                    chunk_opts.mask_small,
                ),
                seed_pool,
                seeds,
                best,
                lg_layers,
                attempts.as_deref_mut(),
            )?;
        }

        *base_idx += chunk.len();
        if accepted_in_slice {
            break;
        }
    }
    Ok(max_pairs)
}

#[expect(
    clippy::too_many_arguments,
    reason = "stage-end top-seed polish borrows surface, GPU renderer, models, intrinsics, and best-candidate state"
)]
fn polish_stage_top_seed(
    surface: &Surface,
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, matcher): (&LoadedXFeat, Option<&dyn SparseMatcher>),
    qry_feat: &SparseFeatures,
    (k_render, render_size, focal_mult, pace_realtime, mask_small): (
        &Matrix3<f64>,
        [u32; 2],
        f64,
        bool,
        Option<&GrayImage>,
    ),
    seed_pool: &[(usize, PoseCandidate)],
    all_seeds: &[SeedPose],
    best: &mut Option<(usize, usize, PoseCandidate)>,
    lg_layers: &mut usize,
    attempts: Option<&mut Vec<AttemptRecord>>,
) -> Result<bool, FlightVisualError> {
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
    let out = polish_anchor_seed_pool(
        surface,
        gpu_renderer,
        xfeat,
        matcher,
        qry_feat,
        (k_render, render_size, focal_mult, pace_realtime, mask_small),
        vec![top_seed],
        (None, None, &[], all_seeds),
        best,
        attempts,
    )?;
    *lg_layers += out.lg_layers;
    Ok(best.as_ref().is_some_and(|(_, _, c)| {
        c.score.geometry_accepted
            && is_plausible_anchor_candidate(c, surface)
            && (is_unambiguously_confirmed(&c.score) || !is_fragile_virtual_anchor(&c.score))
    }))
}
