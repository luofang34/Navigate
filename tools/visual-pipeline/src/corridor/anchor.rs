//! Virtual-view map-anchor acquisition against [`CorridorTileStore`].

pub mod multitier;
pub mod plausibility;

use super::{CorridorTrackOptions, kinematics::match_virtual_kashin};
use crate::{
    PipelineError,
    pose::{PoseCandidate, candidate_priority, fit_pose_hybrid, refine_seed_eye_2pt},
    realtime::helpers::{log_attempt, polish_pose_sparse, rot_to_row_major},
    stages::{MatchStageOptions, generate_seeds, load_horizon_attitude},
    surface::{Surface, SurfaceGpuRenderer},
    testset::PhotoCase,
    xfeat_runtime::LoadedXFeat,
};
use nalgebra::{Matrix3, Vector3};
use navigate_match::{
    CorridorTileStore, FastMatchConfig, HadamardProjector, IndexedMatch, LighterGlue,
    PreparedKashinQuery, SparseFeatures, VirtualCorridorView, VirtualViewCamera,
};
use plausibility::{PhysicalPlausibilityConfig, is_plausible_candidate};
use serde_json::{Value, json};
use std::{path::PathBuf, sync::Mutex};

const ANCHOR_BATCH_SIZE: usize = 4;

pub(super) struct AnchorAcquisition {
    pub best: Option<(usize, usize, PoseCandidate)>,
    pub lg_layers: usize,
    pub polish_rays: usize,
    pub best_view: Option<VirtualCorridorView>,
}

pub(super) type AnchorSeedHint = (Option<([f64; 3], Matrix3<f64>)>, Option<(f64, f64)>);

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
    (pred_pose, live_attitude): AnchorSeedHint,
    attempts_log: &mut Vec<Value>,
) -> Result<AnchorAcquisition, PipelineError> {
    let stage_opts = MatchStageOptions {
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
    };
    let attitude = load_horizon_attitude(&opts.dataset_dir, case, k_full).or(live_attitude);
    let seeds = build_anchor_seeds(
        surface,
        case,
        &stage_opts,
        attitude,
        pred_pose,
        opts.anchor_attempts,
    );
    let intrinsics = [
        k_render[(0, 0)],
        k_render[(1, 1)],
        k_render[(0, 2)],
        k_render[(1, 2)],
    ];
    let max_lm = opts.query_keypoints.clamp(256, 2048);
    let mut best: Option<(usize, usize, PoseCandidate)> = None;
    let mut best_view: Option<VirtualCorridorView> = None;
    let mut seed_pool: Vec<(usize, PoseCandidate)> = Vec::new();
    let mut lg_layers = 0_usize;

    let (lead_count, rest_offset) = if pred_pose.is_some() && !seeds.is_empty() {
        (1_usize, 1_usize)
    } else {
        (0_usize, 0_usize)
    };
    let mut chunks: Vec<&[([f64; 3], Matrix3<f64>)]> = Vec::new();
    if lead_count == 1 {
        chunks.push(&seeds[..1]);
    }
    for ch in seeds[rest_offset..].chunks(ANCHOR_BATCH_SIZE) {
        chunks.push(ch);
    }

    let mut base_idx = 0_usize;
    for chunk in chunks {
        let fast_exit = pred_pose.is_some() && base_idx == 0 && chunk.len() == 1;
        let evals = eval_seed_batch(
            (surface, store, projector),
            gpu_renderer,
            (xfeat, lighterglue),
            (qry_feat, prepared),
            chunk,
            (intrinsics, k_render, render_size, max_lm),
            fast_exit,
        )?;
        let mut accepted_early = false;
        for (local_i, (cand_opt, virt, layers)) in evals.into_iter().enumerate() {
            let seed_idx = base_idx + local_i;
            lg_layers += layers;
            log_attempt(
                attempts_log,
                "corridor-virtual-view+l1-gnc",
                seed_idx,
                0,
                1.0,
                cand_opt.as_ref().map_or(0, |c| c.score.pairs),
                cand_opt.as_ref(),
            );
            if let Some(last) = attempts_log.last_mut().and_then(Value::as_object_mut) {
                last.insert("virtual_landmarks".into(), json!(virt.features.len()));
                last.insert("visible_tiles".into(), json!(virt.visible_tiles));
            }

            let hyp_eye = chunk[local_i].0;
            let mut cur_cand = cand_opt.filter(|c| is_plausible_anchor_candidate(c, surface));
            let mut cur_virt = virt;
            let mut prev_eye = hyp_eye;
            for re_step in 1..=2_usize {
                let Some(ref c_prev) = cur_cand else {
                    break;
                };
                let shift_m = (Vector3::from(c_prev.eye) - Vector3::from(prev_eye)).norm();
                if c_prev.score.inliers < 10 || (c_prev.score.geometry_accepted && shift_m <= 120.0)
                {
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
                    false,
                )?;
                let (re_cand, re_virt, re_layers) =
                    single.pop().unwrap_or((None, cur_virt.clone(), 0));
                lg_layers += re_layers;
                let re_cand = re_cand.filter(|c| is_plausible_anchor_candidate(c, surface));
                log_attempt(
                    attempts_log,
                    "corridor-virtual-reproject+l1-gnc",
                    seed_idx,
                    re_step,
                    1.0,
                    re_cand.as_ref().map_or(0, |c| c.score.pairs),
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
                let is_acc = c.score.geometry_accepted;
                let take = best.as_ref().is_none_or(|(_, _, prev)| {
                    candidate_priority(&c.score) > candidate_priority(&prev.score)
                });
                seed_pool.push((seed_idx, c.clone()));
                if take {
                    best = Some((seed_idx, 0, c));
                    best_view = Some(cur_virt);
                }
                if is_acc {
                    accepted_early = true;
                    break;
                }
            }
        }
        base_idx += chunk.len();
        if accepted_early {
            break;
        }
    }

    if best
        .as_ref()
        .is_none_or(|(_, _, c)| !c.score.geometry_accepted)
        && let Ok(rgb_img) = crate::stages::load_case_rgb(&opts.dataset_dir, &case.image)
    {
        let dyn_img = image::DynamicImage::ImageRgb8(rgb_img);
        let search_prior =
            multitier::build_search_prior_from_hint(pred_pose, opts.anchor_uncertainty_m);
        let matcher =
            multitier::SearchTierMatcher::new(multitier::MultiTierSearchConfig::default());
        if let Ok(fb_seeds) = multitier::resolve_global_fallback_seeds(
            &matcher,
            &search_prior,
            &dyn_img,
            surface,
            store,
            480.0,
        ) && !fb_seeds.is_empty()
        {
            let fb_evals = eval_seed_batch(
                (surface, store, projector),
                gpu_renderer,
                (xfeat, lighterglue),
                (qry_feat, prepared),
                &fb_seeds,
                (intrinsics, k_render, render_size, max_lm),
                false,
            )?;
            for (local_i, (cand_opt, virt, layers)) in fb_evals.into_iter().enumerate() {
                lg_layers += layers;
                if let Some(c) = cand_opt.filter(|c| is_plausible_anchor_candidate(c, surface)) {
                    let seed_idx = base_idx + local_i;
                    let take = best.as_ref().is_none_or(|(_, _, prev)| {
                        candidate_priority(&c.score) > candidate_priority(&prev.score)
                    });
                    seed_pool.push((seed_idx, c.clone()));
                    if take {
                        best = Some((seed_idx, 0, c.clone()));
                        best_view = Some(virt);
                    }
                    if c.score.geometry_accepted {
                        break;
                    }
                }
            }
        }
    }

    let mut polish_rays = 0_usize;
    if opts.polish_anchors {
        seed_pool.sort_by(|a, b| {
            (
                b.1.score.geometry_accepted,
                b.1.score.inliers,
                b.1.score.pairs,
            )
                .cmp(&(
                    a.1.score.geometry_accepted,
                    a.1.score.inliers,
                    a.1.score.pairs,
                ))
        });
        let polish_cap = if seed_pool
            .first()
            .is_some_and(|(_, c)| c.score.inliers >= 12)
        {
            1
        } else {
            2
        };
        for (seed_idx, mut cur_cand) in seed_pool.into_iter().take(polish_cap) {
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
                    1.0,
                    polished.score.pairs,
                    Some(&polished),
                );
                let acc_now = polished.score.geometry_accepted;
                if candidate_priority(&polished.score) >= candidate_priority(&cur_cand.score) {
                    cur_cand = polished.clone();
                }
                if best.as_ref().is_none_or(|(_, _, prev)| {
                    candidate_priority(&polished.score) >= candidate_priority(&prev.score)
                }) {
                    best = Some((seed_idx, p_step + 1, polished));
                }
                if acc_now {
                    break;
                }
            }
            if best
                .as_ref()
                .is_some_and(|(_, _, c)| c.score.geometry_accepted)
            {
                break;
            }
        }
    }
    Ok(AnchorAcquisition {
        best,
        lg_layers,
        polish_rays,
        best_view,
    })
}

type SeedEvalTuple = (Option<PoseCandidate>, VirtualCorridorView, usize);

fn eval_seed_batch(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, lighterglue): (&LoadedXFeat, Option<&LighterGlue>),
    (qry_feat, prepared): (&SparseFeatures, &PreparedKashinQuery),
    poses: &[([f64; 3], Matrix3<f64>)],
    (intrinsics, k_render, render_size, max_lm): ([f64; 4], &Matrix3<f64>, [u32; 2], usize),
    fast_exit: bool,
) -> Result<Vec<SeedEvalTuple>, PipelineError> {
    let k_cap = if lighterglue.is_some() { 640 } else { max_lm };
    let mut results: Vec<SeedEvalTuple> = Vec::with_capacity(poses.len());
    let mut need_lg: Vec<usize> = Vec::new();

    for (i, &(hyp_eye, hyp_rot)) in poses.iter().enumerate() {
        let min_cos_orientation = if -hyp_rot[(2, 2)] > 0.95 { 0.80 } else { 0.90 };
        let virt = store.project_virtual_view(
            &VirtualViewCamera {
                eye_enu: hyp_eye,
                world_to_cam: rot_to_row_major(&hyp_rot),
                intrinsics,
                image_size: render_size,
                max_landmarks: max_lm,
                min_cos_incidence: 0.84,
                min_cos_orientation,
            },
            projector,
        )?;
        if virt.world_points_enu.len() < 12 {
            if let Some(gpu_lock) = gpu_renderer
                && let Some((polished, layers, _)) = polish_pose_sparse(
                    surface,
                    Some(gpu_lock),
                    xfeat,
                    lighterglue,
                    qry_feat,
                    (hyp_eye, hyp_rot),
                    k_render,
                    render_size,
                )?
            {
                results.push((Some(polished), virt, layers));
                continue;
            }
            results.push((None, virt, 0));
            continue;
        }
        let (k_world, k_px) = match_virtual_kashin(&virt, prepared, k_cap);
        let cand = fit_pose_hybrid(
            &k_world,
            &k_px,
            k_render,
            render_size,
            Some((hyp_eye, hyp_rot)),
        )
        .filter(|c| is_plausible_anchor_candidate(c, surface));
        let accepted = cand.as_ref().is_some_and(|c| c.score.geometry_accepted);
        results.push((cand, virt, 0));
        if accepted {
            return Ok(results);
        }
        need_lg.push(i);
    }

    if let Some(lg) = lighterglue
        && !need_lg.is_empty()
    {
        let base_cfg = if fast_exit {
            FastMatchConfig::realtime_tracking()
        } else {
            FastMatchConfig::balanced_search()
        };
        let cfg = FastMatchConfig {
            max_keypoints: max_lm.min(1024),
            ..base_cfg
        };
        let ref_slices: Vec<&SparseFeatures> = need_lg
            .iter()
            .map(|&idx| &results[idx].1.features)
            .collect();
        let batched =
            pollster::block_on(lg.match_features_batch_indexed_fast(&ref_slices, qry_feat, &cfg))?;
        for (slot_pos, (idx_matches, stats)) in need_lg.into_iter().zip(batched) {
            let (hyp_eye, hyp_rot) = poses[slot_pos];
            let (ref mut cand, ref virt, ref mut layers) = results[slot_pos];
            *layers = if slot_pos == 0 {
                stats.layers_executed
            } else {
                0
            };
            let (w_pts, px) = collect_virtual_indexed(virt, &idx_matches);
            let mut lg_best =
                fit_pose_hybrid(&w_pts, &px, k_render, render_size, Some((hyp_eye, hyp_rot)))
                    .filter(|c| is_plausible_anchor_candidate(c, surface));
            if !lg_best.as_ref().is_some_and(|c| c.score.geometry_accepted)
                && let Some(tri_init) =
                    refine_seed_eye_2pt(&w_pts, &px, (hyp_eye, hyp_rot), k_render)
                && let Some(tri_cand) =
                    fit_pose_hybrid(&w_pts, &px, k_render, render_size, Some(tri_init))
                        .filter(|c| is_plausible_anchor_candidate(c, surface))
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
        }
    }
    Ok(results)
}

fn is_plausible_anchor_candidate(cand: &PoseCandidate, surface: &Surface) -> bool {
    let config = PhysicalPlausibilityConfig::default();
    is_plausible_candidate(cand, surface, None, false, &config)
}

fn build_anchor_seeds(
    surface: &Surface,
    case: &PhotoCase,
    stage_opts: &MatchStageOptions,
    attitude: Option<(f64, f64)>,
    pred_pose: Option<([f64; 3], Matrix3<f64>)>,
    anchor_attempts: usize,
) -> Vec<([f64; 3], Matrix3<f64>)> {
    pred_pose
        .into_iter()
        .chain(
            generate_seeds(surface, case, stage_opts, attitude)
                .into_iter()
                .take(anchor_attempts.max(1)),
        )
        .collect()
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

#[cfg(test)]
mod tests;
