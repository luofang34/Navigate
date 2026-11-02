//! Seed-batch virtual-view projection, Kashin screening, and LighterGlue refinement.

use super::{
    AnchorQueryAux, SeedStateMut, is_plausible_anchor_candidate, is_sane_anchor_seed, multitier,
    nadir_screen::SeedPose,
};
use crate::{
    PipelineError,
    corridor::{CorridorTrackOptions, kinematics::match_virtual_kashin_scored},
    pose::{PoseCandidate, candidate_priority, fit_pose_hybrid, refine_seed_eye_2pt},
    realtime::helpers::{polish_pose_sparse, rot_to_row_major},
    surface::{Surface, SurfaceGpuRenderer},
    testset::PhotoCase,
    xfeat_runtime::LoadedXFeat,
};
use nalgebra::Matrix3;
use navigate_match::{
    CorridorTileStore, FastMatchConfig, HadamardProjector, IndexedMatch, LighterGlue,
    PreparedKashinQuery, SparseFeatures, VirtualCorridorView, VirtualViewCamera,
};
use std::{cmp::Reverse, sync::Mutex};

pub(super) type SeedEvalTuple = (Option<PoseCandidate>, VirtualCorridorView, usize);

#[derive(Clone, Copy)]
pub(super) struct SeedBatchFlags {
    pub fast_exit: bool,
    pub coarse_screen: bool,
    pub allow_horizon_gpu: bool,
}

const COARSE_LG_SEED_CAP: usize = 8;
const COARSE_NADIR_LG_SEED_CAP: usize = 12;
const FULL_BATCH_LG_SEED_CAP: usize = 12;

pub(super) fn eval_seed_batch(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, lighterglue): (&LoadedXFeat, Option<&LighterGlue>),
    (qry_feat, prepared): (&SparseFeatures, &PreparedKashinQuery),
    poses: &[([f64; 3], Matrix3<f64>)],
    (intrinsics, k_render, render_size, max_lm): ([f64; 4], &Matrix3<f64>, [u32; 2], usize),
    flags: SeedBatchFlags,
) -> Result<Vec<SeedEvalTuple>, PipelineError> {
    let virt_lm_cap = max_lm.min(1024);
    let k_cap = if lighterglue.is_some() {
        640
    } else {
        virt_lm_cap
    };
    let inline_gpu_cap = if flags.coarse_screen {
        2_usize
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
        let mut virt = store.project_virtual_view(
            &VirtualViewCamera {
                eye_enu: hyp_eye,
                world_to_cam: rot_to_row_major(&hyp_rot),
                intrinsics,
                image_size: render_size,
                max_landmarks: virt_lm_cap,
                min_cos_incidence,
                min_cos_orientation,
            },
            projector,
        )?;
        if sin_down > 0.95 && virt.world_points_enu.len() < 48 {
            virt = store.project_virtual_view(
                &VirtualViewCamera {
                    eye_enu: hyp_eye,
                    world_to_cam: rot_to_row_major(&hyp_rot),
                    intrinsics,
                    image_size: render_size,
                    max_landmarks: virt_lm_cap,
                    min_cos_incidence,
                    min_cos_orientation: 0.80,
                },
                projector,
            )?;
        }
        if virt.world_points_enu.len() < 12 || (flags.allow_horizon_gpu && sin_down < 0.02) {
            let can_inline_gpu =
                inline_gpu_used < inline_gpu_cap && (!steep_nadir || poses.len() <= 4);
            if can_inline_gpu
                && let Some(gpu_lock) = gpu_renderer
                && let Some((mut polished, _, mut layers)) = polish_pose_sparse(
                    surface,
                    Some(gpu_lock),
                    xfeat,
                    lighterglue,
                    qry_feat,
                    (hyp_eye, hyp_rot),
                    k_render,
                    render_size,
                )?
                && is_sane_anchor_seed(&polished, surface)
            {
                inline_gpu_used += 1;
                let mut acc_gpu = polished.score.geometry_accepted
                    && is_plausible_anchor_candidate(&polished, surface);
                if !acc_gpu
                    && polished.score.inliers >= 20
                    && let Some((pol2, _, layers2)) = polish_pose_sparse(
                        surface,
                        Some(gpu_lock),
                        xfeat,
                        lighterglue,
                        qry_feat,
                        (polished.eye, polished.rotation),
                        k_render,
                        render_size,
                    )?
                    && is_sane_anchor_seed(&pol2, surface)
                    && candidate_priority(&pol2.score) >= candidate_priority(&polished.score)
                {
                    layers += layers2;
                    acc_gpu = pol2.score.geometry_accepted
                        && is_plausible_anchor_candidate(&pol2, surface);
                    polished = pol2;
                }
                results.push((Some(polished), virt, layers));
                if acc_gpu {
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
        let accepted = cand.as_ref().is_some_and(|c| {
            c.score.geometry_accepted && is_plausible_anchor_candidate(c, surface)
        });
        let k_inliers = cand.as_ref().map_or(0, |c| c.score.inliers);
        results.push((cand, virt, 0));
        if accepted {
            return Ok(results);
        }
        need_lg.push((i, k_inliers, k_world.len(), kashin_sim));
    }

    if let Some(lg) = lighterglue
        && !need_lg.is_empty()
    {
        let all_steep_nadir = poses.iter().all(|&(_, r)| -r[(2, 2)] > 0.85);
        need_lg.sort_by_key(|&(i, k_inliers, k_matches, kashin_sim)| {
            Reverse((
                k_inliers,
                k_matches,
                (f64::from(kashin_sim) * 1_000.0).round() as i64,
                Reverse(i),
            ))
        });
        let lg_cap = if flags.coarse_screen {
            if all_steep_nadir {
                COARSE_NADIR_LG_SEED_CAP
            } else {
                COARSE_LG_SEED_CAP
            }
        } else {
            FULL_BATCH_LG_SEED_CAP
        };
        need_lg.truncate(lg_cap);

        let max_kp = if flags.fast_exit || flags.coarse_screen || poses.len() > 12 {
            max_lm.min(768)
        } else {
            max_lm.min(1024)
        };
        let layers = if flags.coarse_screen
            || flags.fast_exit
            || (poses.len() > super::ANCHOR_BATCH_SIZE && !all_steep_nadir)
        {
            4
        } else {
            6
        };
        let cfg = FastMatchConfig {
            max_keypoints: max_kp,
            min_layers: layers,
            max_layers: layers,
            exit_check_layers: 0,
            prune_after_layer: None,
            ..FastMatchConfig::default()
        };

        for lg_chunk in need_lg.chunks(super::ANCHOR_BATCH_SIZE) {
            let mut ref_slices: Vec<&SparseFeatures> = lg_chunk
                .iter()
                .map(|&(idx, _, _, _)| &results[idx].1.features)
                .collect();
            if (2..super::ANCHOR_BATCH_SIZE).contains(&ref_slices.len())
                && let Some(&pad_ref) = ref_slices.iter().find(|r| !r.is_empty())
            {
                ref_slices.resize(super::ANCHOR_BATCH_SIZE, pad_ref);
            }
            let batched = pollster::block_on(lg.match_features_batch_indexed_fast(
                &ref_slices,
                qry_feat,
                &cfg,
            ))?;
            let mut batch_accepted = false;
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
                    let is_acc = c.score.geometry_accepted && plausible;
                    let strong_coarse = flags.coarse_screen
                        && c.score.inliers >= 28
                        && c.score.holdout_inlier_fraction >= 0.50
                        && plausible;
                    if is_acc || strong_coarse {
                        batch_accepted = true;
                    }
                }
            }
            if batch_accepted {
                break;
            }
        }
    }
    Ok(results)
}

#[expect(
    clippy::too_many_arguments,
    reason = "global fallback batch evaluation shares terrain, tile store, models, query features, and best-candidate state"
)]
pub(super) fn eval_global_fallback_seeds(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, lighterglue): (&LoadedXFeat, Option<&LighterGlue>),
    (qry_feat, prepared): (&SparseFeatures, &PreparedKashinQuery),
    fb_seeds: &[SeedPose],
    (intrinsics, k_render, render_size, max_lm): ([f64; 4], &Matrix3<f64>, [u32; 2], usize),
    base_idx: &mut usize,
    lg_layers: &mut usize,
    seed_pool: &mut Vec<(usize, PoseCandidate)>,
    best: &mut Option<(usize, usize, PoseCandidate)>,
    best_view: &mut Option<VirtualCorridorView>,
) -> Result<(), PipelineError> {
    if fb_seeds.is_empty() {
        return Ok(());
    }
    let fb_evals = eval_seed_batch(
        (surface, store, projector),
        gpu_renderer,
        (xfeat, lighterglue),
        (qry_feat, prepared),
        fb_seeds,
        (intrinsics, k_render, render_size, max_lm),
        SeedBatchFlags {
            fast_exit: false,
            coarse_screen: false,
            allow_horizon_gpu: true,
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
                (is_acc, candidate_priority(&c.score)) > (prev_acc, candidate_priority(&prev.score))
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

#[expect(
    clippy::too_many_arguments,
    reason = "global retrieval fallback conditionally runs SearchTierMatcher when virtual-view search is unaccepted"
)]
pub(in crate::corridor::anchor) fn eval_global_fallback_if_needed(
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
