//! Seed-batch virtual-view projection, Kashin screening, and `SparseMatcher` refinement.

use super::{
    ANCHOR_BATCH_SIZE, AnchorQuery, AnchorQueryAux, SeedStateMut,
    basin::{basin_priority_key, is_unambiguously_confirmed},
    is_plausible_anchor_candidate, is_sane_anchor_seed,
    nadir_screen::rot_to_row_major,
    polish::is_fragile_virtual_anchor,
    seeds::{AnchorSearchOpts, SeedPose, is_heading_known},
    sparse_polish::polish_pose_sparse_masked,
    virt_mask::mask_valid_fraction,
};
use crate::{
    error::FlightVisualError,
    runtime::LoadedXFeat,
    surface::{Surface, SurfaceGpuRenderer},
};
use image::{DynamicImage, GrayImage};
use nalgebra::Matrix3;
use navigate_locate::corridor::{
    kinematics::match_virtual_kashin_scored,
    multitier::{
        build_search_prior_from_hint, gray_mask_to_bools, resolve_global_fallback_seeds_with_mask,
    },
};
use navigate_match::{
    CorridorTileStore, FastMatchConfig, HadamardProjector, IndexedMatch, PreparedKashinQuery,
    SparseFeatures, SparseMatcher, VirtualCorridorView, VirtualViewCamera,
};
use navigate_visual::{
    PoseCandidate,
    holdout_pnp::{candidate_priority, fit_pose_hybrid, refine_seed_eye_2pt},
};
use std::{cmp::Reverse, sync::Mutex};

pub(super) type SeedEvalTuple = (Option<PoseCandidate>, VirtualCorridorView, usize);

#[derive(Clone, Copy)]
pub(super) struct SeedBatchFlags<'a> {
    pub priority_lead: bool,
    pub coarse_screen: bool,
    pub allow_horizon_gpu: bool,
    pub pace_realtime: bool,
    pub mask_small: Option<&'a GrayImage>,
}

#[derive(Clone, Copy)]
pub(crate) struct SeedChunkOpts<'a> {
    pub lead_count: usize,
    pub coarse_screen: bool,
    pub allow_horizon_gpu: bool,
    pub pace_realtime: bool,
    pub focal_mult: f64,
    pub mask_small: Option<&'a GrayImage>,
}

impl<'a> SeedChunkOpts<'a> {
    pub(crate) const fn new(
        lead_count: usize,
        coarse_screen: bool,
        allow_horizon_gpu: bool,
        pace_realtime: bool,
        focal_mult: f64,
        mask_small: Option<&'a GrayImage>,
    ) -> Self {
        Self {
            lead_count,
            coarse_screen,
            allow_horizon_gpu,
            pace_realtime,
            focal_mult,
            mask_small,
        }
    }
}

pub(crate) fn lead_seed_count(
    query: &AnchorQuery,
    opts: &AnchorSearchOpts,
    has_pred: bool,
) -> usize {
    if is_heading_known(query.prior, opts) {
        ANCHOR_BATCH_SIZE
    } else {
        usize::from(has_pred)
    }
}

const PACED_NADIR_LG_SEED_CAP: usize = 4;
const COARSE_LG_SEED_CAP: usize = 12;
const COARSE_NADIR_OR_COREML_LG_SEED_CAP: usize = 16;
const FULL_BATCH_LG_SEED_CAP: usize = 24;

pub(super) fn eval_seed_batch(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, matcher): (&LoadedXFeat, Option<&dyn SparseMatcher>),
    (qry_feat, prepared): (&SparseFeatures, &PreparedKashinQuery),
    poses: &[SeedPose],
    (intrinsics, k_render, render_size, max_lm): ([f64; 4], &Matrix3<f64>, [u32; 2], usize),
    flags: SeedBatchFlags<'_>,
) -> Result<Vec<SeedEvalTuple>, FlightVisualError> {
    let virt_lm_cap = max_lm.min(1024);
    let active_mask = flags.mask_small.filter(|m| mask_valid_fraction(m) < 0.85);
    let proj_lm_cap = if active_mask.is_some() {
        virt_lm_cap.saturating_mul(2).min(2048)
    } else {
        virt_lm_cap
    };
    let k_cap = if matcher.is_some() { 640 } else { virt_lm_cap };
    let inline_gpu_cap = if flags.pace_realtime {
        1_usize
    } else if flags.coarse_screen {
        0_usize
    } else if active_mask.is_some() {
        12_usize
    } else {
        4_usize
    };
    let mut inline_gpu_used = 0_usize;
    let mut results: Vec<SeedEvalTuple> = Vec::with_capacity(poses.len());
    let mut need_lg: Vec<(usize, usize, usize, f32)> = Vec::new();

    for (i, &(hyp_eye, hyp_rot)) in poses.iter().enumerate() {
        let sin_down = -hyp_rot[(2, 2)];
        let steep_nadir = sin_down > 0.85;
        let min_cos_orientation = if sin_down > 0.95 { 0.94 } else { 0.80 };
        let min_cos_incidence = if steep_nadir {
            0.45
        } else if sin_down < 0.25 {
            0.72
        } else {
            0.84
        };
        let w2c = rot_to_row_major(&hyp_rot);
        let proj_view = |min_cos_incidence: f32, min_cos_orientation: f32| {
            store.project_virtual_view(
                &VirtualViewCamera {
                    eye_enu: hyp_eye,
                    world_to_cam: w2c,
                    intrinsics,
                    image_size: render_size,
                    max_landmarks: proj_lm_cap,
                    min_cos_incidence,
                    min_cos_orientation,
                },
                projector,
            )
        };
        let mut virt = proj_view(min_cos_incidence, min_cos_orientation)?;
        if sin_down > 0.95 && virt.world_points_enu.len() < 48 {
            virt = proj_view(min_cos_incidence, 0.80)?;
        } else if !steep_nadir && virt.world_points_enu.len() < 48 {
            let (fb_inc, fb_ori) = if sin_down < 0.15 {
                (0.05, 0.35)
            } else {
                (0.55, 0.65)
            };
            virt = proj_view(fb_inc, fb_ori)?;
        }

        if virt.world_points_enu.len() < 12
            || (flags.allow_horizon_gpu && sin_down < 0.02 && virt.world_points_enu.len() < 32)
        {
            let stride_ok = flags.priority_lead
                || poses.len() <= 12
                || (!flags.pace_realtime && active_mask.is_some())
                || i % 2 == 0;
            let can_inline_gpu = virt.visible_tiles > 0
                && inline_gpu_used < inline_gpu_cap
                && (!steep_nadir || (!flags.pace_realtime && poses.len() <= 4))
                && stride_ok;
            if can_inline_gpu
                && let Some(gpu_lock) = gpu_renderer
                && let Some((mut polished, _, mut layers)) = polish_pose_sparse_masked(
                    surface,
                    Some(gpu_lock),
                    xfeat,
                    matcher,
                    qry_feat,
                    (hyp_eye, hyp_rot),
                    k_render,
                    render_size,
                    (active_mask, flags.pace_realtime, false),
                )?
                && is_sane_anchor_seed(&polished, surface)
            {
                inline_gpu_used += 1;
                let mut acc_gpu = polished.score.geometry_accepted
                    && is_plausible_anchor_candidate(&polished, surface);
                if !acc_gpu
                    && polished.score.inliers >= 10
                    && let Some((pol2, _, layers2)) = polish_pose_sparse_masked(
                        surface,
                        Some(gpu_lock),
                        xfeat,
                        matcher,
                        qry_feat,
                        (polished.eye, polished.rotation),
                        k_render,
                        render_size,
                        (active_mask, flags.pace_realtime, false),
                    )?
                    && is_sane_anchor_seed(&pol2, surface)
                    && candidate_priority(&pol2.score) >= candidate_priority(&polished.score)
                {
                    layers += layers2;
                    acc_gpu = pol2.score.geometry_accepted
                        && is_plausible_anchor_candidate(&pol2, surface);
                    polished = pol2;
                }
                let tight_gpu = acc_gpu && polished.score.holdout_median_px <= 4.5;
                results.push((Some(polished), virt, layers));
                if tight_gpu {
                    return Ok(results);
                }
                continue;
            }
            if virt.world_points_enu.len() < 12 {
                results.push((None, virt, 0));
                continue;
            }
        }

        let (k_world, k_px, kashin_sim) = match_virtual_kashin_scored(&virt, prepared, k_cap);
        let cand = fit_pose_hybrid(
            &k_world,
            &k_px,
            k_render,
            render_size,
            Some((hyp_eye, hyp_rot)),
        )
        .filter(|c| is_sane_anchor_seed(c, surface));
        let tight_accepted = cand.as_ref().is_some_and(|c| {
            c.score.geometry_accepted
                && is_plausible_anchor_candidate(c, surface)
                && is_unambiguously_confirmed(&c.score)
        });
        let k_inliers = cand.as_ref().map_or(0, |c| c.score.inliers);
        results.push((cand, virt, 0));
        if tight_accepted {
            return Ok(results);
        }
        need_lg.push((i, k_inliers, k_world.len(), kashin_sim));
    }

    if let Some(m) = matcher
        && !need_lg.is_empty()
    {
        let all_steep_nadir = poses.iter().all(|&(_, r)| -r[(2, 2)] > 0.85);
        let is_coreml = xfeat.runtime_name() == "coreml";
        let paced_steep_nadir = flags.pace_realtime
            && all_steep_nadir
            && (!flags.priority_lead || poses.len() >= ANCHOR_BATCH_SIZE)
            && !is_coreml;
        let paced_oblique = flags.pace_realtime && !all_steep_nadir && !flags.priority_lead;
        if !flags.priority_lead {
            need_lg.sort_by_key(|&(i, k_inliers, k_matches, kashin_sim)| {
                let geom_k_inl = if k_inliers >= 6 { k_inliers } else { 0 };
                let center_lead = usize::from(paced_steep_nadir && flags.coarse_screen && i < 4);
                let lead_nadir = usize::from(all_steep_nadir && (i < 4 || (6..10).contains(&i)));
                let primary_nadir = usize::from(all_steep_nadir && i < 12);
                Reverse((
                    geom_k_inl,
                    center_lead,
                    lead_nadir,
                    primary_nadir,
                    k_inliers,
                    (f64::from(kashin_sim) * 1_000.0).round() as i64,
                    k_matches,
                    Reverse(i),
                ))
            });
            let lg_cap = if paced_steep_nadir {
                if flags.coarse_screen {
                    8
                } else {
                    PACED_NADIR_LG_SEED_CAP
                }
            } else if paced_oblique || (!is_coreml && (all_steep_nadir || flags.coarse_screen)) {
                COARSE_LG_SEED_CAP
            } else if all_steep_nadir || flags.coarse_screen {
                COARSE_NADIR_OR_COREML_LG_SEED_CAP
            } else {
                FULL_BATCH_LG_SEED_CAP
            };
            need_lg.truncate(lg_cap);
        }

        let fast_warm_lead = flags.pace_realtime && flags.priority_lead && poses.len() <= 3;
        let max_kp = if fast_warm_lead {
            max_lm.min(512)
        } else if paced_oblique || (flags.coarse_screen && !paced_steep_nadir) {
            max_lm.min(768)
        } else {
            max_lm.min(1024)
        };
        let base_layers =
            if fast_warm_lead || paced_oblique || (flags.coarse_screen && !all_steep_nadir) {
                4
            } else {
                6
            };

        for (batch_idx, lg_chunk) in need_lg.chunks(ANCHOR_BATCH_SIZE).enumerate() {
            let batch_max_kp = if paced_steep_nadir && !flags.coarse_screen && batch_idx >= 1 {
                max_lm.min(512)
            } else if paced_steep_nadir && flags.coarse_screen && batch_idx >= 1 {
                max_lm.min(768)
            } else {
                max_kp
            };
            let layers = if (paced_steep_nadir && batch_idx >= 1)
                || (!all_steep_nadir
                    && !flags.priority_lead
                    && !flags.coarse_screen
                    && batch_idx >= 3)
            {
                4
            } else {
                base_layers
            };
            let cfg = FastMatchConfig {
                max_keypoints: batch_max_kp,
                min_layers: layers,
                max_layers: layers,
                exit_check_layers: 0,
                prune_after_layer: None,
                ..FastMatchConfig::default()
            };
            let mut ref_slices: Vec<&SparseFeatures> = lg_chunk
                .iter()
                .map(|&(idx, _, _, _)| &results[idx].1.features)
                .collect();
            if (2..ANCHOR_BATCH_SIZE).contains(&ref_slices.len())
                && let Some(&pad_ref) = ref_slices.iter().find(|r| !r.is_empty())
            {
                ref_slices.resize(ANCHOR_BATCH_SIZE, pad_ref);
            }
            let batched = m.match_batch_indexed_blocking(&ref_slices, qry_feat, &cfg)?;
            let mut batch_confirmed = false;
            for (chunk_pos, (&(slot_pos, _, _, _), (idx_matches, stats))) in
                lg_chunk.iter().zip(batched).enumerate()
            {
                let (hyp_eye, hyp_rot) = poses[slot_pos];
                let (ref mut cand, ref virt, ref mut slot_layers) = results[slot_pos];
                *slot_layers = if chunk_pos == 0 {
                    stats.layers_executed
                } else {
                    0
                };
                let (w_pts, px) = collect_virtual_indexed(virt, &idx_matches);
                let mut lg_best =
                    fit_pose_hybrid(&w_pts, &px, k_render, render_size, Some((hyp_eye, hyp_rot)))
                        .filter(|c| is_sane_anchor_seed(c, surface));
                if !lg_best.as_ref().is_some_and(|c| {
                    c.score.geometry_accepted && is_plausible_anchor_candidate(c, surface)
                }) && let Some(tri_init) =
                    refine_seed_eye_2pt(&w_pts, &px, (hyp_eye, hyp_rot), k_render)
                    && let Some(tri_cand) =
                        fit_pose_hybrid(&w_pts, &px, k_render, render_size, Some(tri_init))
                            .filter(|c| is_sane_anchor_seed(c, surface))
                    && lg_best.as_ref().is_none_or(|prev| {
                        candidate_priority(&tri_cand.score) > candidate_priority(&prev.score)
                    })
                {
                    lg_best = Some(tri_cand);
                }
                if let Some(lg_cand) = lg_best
                    && cand.as_ref().is_none_or(|prev| {
                        candidate_priority(&lg_cand.score) > candidate_priority(&prev.score)
                    })
                {
                    *cand = Some(lg_cand);
                }
                if let Some(c) = cand.as_ref() {
                    let plausible = is_plausible_anchor_candidate(c, surface);
                    let tight_acc = c.score.geometry_accepted
                        && plausible
                        && ((c.score.holdout_median_px <= 4.5 && c.score.train_rms_px <= 4.5)
                            || (!all_steep_nadir
                                && !is_fragile_virtual_anchor(&c.score)
                                && c.score.inliers >= 64
                                && c.score.train_rms_px <= 5.5));
                    let strong_coarse = flags.coarse_screen
                        && c.score.inliers >= 28
                        && c.score.holdout_inlier_fraction >= 0.50
                        && c.score.holdout_median_px <= 4.5
                        && plausible;
                    if tight_acc || strong_coarse {
                        batch_confirmed = true;
                    }
                }
            }
            if batch_confirmed {
                break;
            }
        }
    }
    Ok(results)
}

#[expect(
    clippy::too_many_arguments,
    reason = "global retrieval fallback conditionally runs SearchTierMatcher on in-memory RGB buffers"
)]
pub(crate) fn eval_global_fallback_if_needed(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, matcher): (&LoadedXFeat, Option<&dyn SparseMatcher>),
    opts: &AnchorSearchOpts,
    (qry_feat, prepared): (&SparseFeatures, &PreparedKashinQuery),
    (intrinsics, k_render, render_size, max_lm): ([f64; 4], &Matrix3<f64>, [u32; 2], usize),
    aux: &AnchorQueryAux<'_>,
    attitude: Option<(f64, f64)>,
    (base_idx, lg_layers): (&mut usize, &mut usize),
    (seed_pool, best, best_view): SeedStateMut<'_>,
) -> Result<(), FlightVisualError> {
    let already_accepted = best.as_ref().is_some_and(|(_, _, c)| {
        c.score.geometry_accepted && is_plausible_anchor_candidate(c, surface)
    });
    if already_accepted {
        return Ok(());
    }
    let (Some(tier_matcher), Some(small_rgb)) = (aux.search_tier_matcher, aux.small_rgb) else {
        return Ok(());
    };
    let dyn_img = DynamicImage::ImageRgb8(small_rgb.clone());
    let mask_bools = aux.mask_small.map(gray_mask_to_bools);
    let search_prior = build_search_prior_from_hint(aux.pred_pose, opts.anchor_uncertainty_m);
    let Ok(fb_seeds) = resolve_global_fallback_seeds_with_mask(
        tier_matcher,
        &search_prior,
        &dyn_img,
        mask_bools.as_deref(),
        surface,
        store,
        480.0,
        attitude,
    ) else {
        return Ok(());
    };
    if fb_seeds.is_empty() {
        return Ok(());
    }
    let fb_evals = eval_seed_batch(
        (surface, store, projector),
        gpu_renderer,
        (xfeat, matcher),
        (qry_feat, prepared),
        &fb_seeds,
        (intrinsics, k_render, render_size, max_lm),
        SeedBatchFlags {
            priority_lead: fb_seeds.len() <= ANCHOR_BATCH_SIZE,
            coarse_screen: false,
            allow_horizon_gpu: true,
            pace_realtime: opts.pace_realtime,
            mask_small: aux.mask_small,
        },
    )?;
    let evaluated_n = fb_evals.len();
    for (local_i, (cand_opt, virt, layers)) in fb_evals.into_iter().enumerate() {
        *lg_layers += layers;
        if let Some(c) = cand_opt.filter(|c| is_sane_anchor_seed(c, surface)) {
            let seed_idx = *base_idx + local_i;
            let is_acc = c.score.geometry_accepted && is_plausible_anchor_candidate(&c, surface);
            let take = best.as_ref().is_none_or(|(_, _, prev)| {
                let prev_acc =
                    prev.score.geometry_accepted && is_plausible_anchor_candidate(prev, surface);
                basin_priority_key(is_acc, &c) > basin_priority_key(prev_acc, prev)
            });
            seed_pool.push((seed_idx, c.clone()));
            if take {
                *best = Some((seed_idx, 0, c));
                *best_view = Some(virt);
            }
        }
    }
    *base_idx += evaluated_n;
    Ok(())
}

fn collect_virtual_indexed(
    virt: &VirtualCorridorView,
    matches: &[IndexedMatch],
) -> (Vec<[f64; 3]>, Vec<[f64; 2]>) {
    let mut w = Vec::with_capacity(matches.len());
    let mut p = Vec::with_capacity(matches.len());
    for m in matches {
        if let Some(&pt) = virt.world_points_enu.get(m.ref_index) {
            w.push(pt);
            p.push([m.pixel_match.query.x, m.pixel_match.query.y]);
        }
    }
    (w, p)
}
