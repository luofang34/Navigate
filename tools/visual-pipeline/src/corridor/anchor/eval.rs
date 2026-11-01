//! Seed-batch virtual-view projection, Kashin matching, and LighterGlue refinement.

use super::{is_plausible_anchor_candidate, is_sane_anchor_seed, nadir_screen::SeedPose};
use crate::{
    PipelineError,
    corridor::kinematics::match_virtual_kashin,
    pose::{PoseCandidate, PoseScore, candidate_priority, fit_pose_hybrid, refine_seed_eye_2pt},
    realtime::helpers::{polish_pose_sparse, rot_to_row_major},
    surface::{Surface, SurfaceGpuRenderer},
    xfeat_runtime::LoadedXFeat,
};
use nalgebra::{Matrix3, Vector3};
use navigate_match::{
    CorridorTileStore, FastMatchConfig, HadamardProjector, IndexedMatch, LighterGlue,
    PreparedKashinQuery, SparseFeatures, VirtualCorridorView, VirtualViewCamera,
};
use std::sync::Mutex;

pub(super) type SeedEvalTuple = (Option<PoseCandidate>, VirtualCorridorView, usize);

#[derive(Clone, Copy)]
pub(super) struct SeedBatchFlags {
    pub fast_exit: bool,
    pub coarse_screen: bool,
    pub allow_horizon_gpu: bool,
}

pub(super) fn eval_seed_batch(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, lighterglue): (&LoadedXFeat, Option<&LighterGlue>),
    (qry_feat, prepared): (&SparseFeatures, &PreparedKashinQuery),
    poses: &[([f64; 3], Matrix3<f64>)],
    (intrinsics, k_render, render_size, max_lm): ([f64; 4], &Matrix3<f64>, [u32; 2], usize),
    flags: SeedBatchFlags,
) -> Result<Vec<SeedEvalTuple>, PipelineError> {
    let virt_lm_cap = if flags.coarse_screen {
        max_lm.min(1024)
    } else {
        max_lm
    };
    let k_cap = if lighterglue.is_some() {
        640
    } else {
        virt_lm_cap
    };
    let mut results: Vec<SeedEvalTuple> = Vec::with_capacity(poses.len());
    let mut need_lg: Vec<usize> = Vec::new();

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
            if let Some(gpu_lock) = gpu_renderer
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
        let (k_world, k_px) = match_virtual_kashin(&virt, prepared, k_cap);
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
        results.push((cand, virt, 0));
        if accepted {
            return Ok(results);
        }
        need_lg.push(i);
    }

    if let Some(lg) = lighterglue
        && !need_lg.is_empty()
    {
        let dense_coarse = flags.coarse_screen
            && need_lg
                .iter()
                .all(|&idx| results[idx].1.features.len() >= 512);
        let max_kp = if flags.fast_exit {
            max_lm.min(512)
        } else if dense_coarse {
            max_lm.min(768)
        } else {
            max_lm.min(1024)
        };
        let layers = if dense_coarse { 4 } else { 6 };
        let cfg = FastMatchConfig {
            max_keypoints: max_kp,
            min_layers: layers,
            max_layers: layers,
            exit_check_layers: 0,
            prune_after_layer: None,
            ..FastMatchConfig::default()
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
        }
    }
    Ok(results)
}

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

fn same_seed_neighborhood(a: &PoseCandidate, b_eye: [f64; 3], b_rot: &Matrix3<f64>) -> bool {
    let dist = (Vector3::from(a.eye) - Vector3::from(b_eye)).norm();
    let cos_ang = (((a.rotation * b_rot.transpose()).trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
    dist <= 20.0 && cos_ang >= 0.96
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
    for ch in fb_seeds.chunks(super::ANCHOR_BATCH_SIZE) {
        let fb_evals = eval_seed_batch(
            (surface, store, projector),
            gpu_renderer,
            (xfeat, lighterglue),
            (qry_feat, prepared),
            ch,
            (intrinsics, k_render, render_size, max_lm),
            SeedBatchFlags {
                fast_exit: false,
                coarse_screen: false,
                allow_horizon_gpu: true,
            },
        )?;
        let mut accepted_fb = false;
        for (local_i, (cand_opt, virt, layers)) in fb_evals.into_iter().enumerate() {
            *lg_layers += layers;
            if let Some(c) = cand_opt.filter(|c| is_sane_anchor_seed(c, surface)) {
                let seed_idx = *base_idx + local_i;
                let is_acc =
                    c.score.geometry_accepted && is_plausible_anchor_candidate(&c, surface);
                let take = best.as_ref().is_none_or(|(_, _, prev)| {
                    let prev_acc = prev.score.geometry_accepted
                        && is_plausible_anchor_candidate(prev, surface);
                    (is_acc, candidate_priority(&c.score))
                        > (prev_acc, candidate_priority(&prev.score))
                });
                seed_pool.push((seed_idx, c.clone()));
                if take {
                    *best = Some((seed_idx, 0, c));
                    *best_view = Some(virt);
                }
                if is_acc {
                    accepted_fb = true;
                    break;
                }
            }
        }
        *base_idx += ch.len();
        if accepted_fb {
            break;
        }
    }
    Ok(())
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
    screened_seeds: Option<&[SeedPose]>,
    closest_nadir: Option<SeedPose>,
    best: &mut Option<(usize, usize, PoseCandidate)>,
    attempts_log: &mut Vec<serde_json::Value>,
) -> Result<(usize, usize), PipelineError> {
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
    let best_accepted = best.as_ref().is_some_and(|(_, _, c)| {
        c.score.geometry_accepted && is_plausible_anchor_candidate(c, surface)
    });
    let has_strong_seed = seed_pool
        .first()
        .is_some_and(|(_, c)| c.score.inliers >= 16);
    let mut selected: Vec<(usize, PoseCandidate)> = if best_accepted {
        seed_pool.into_iter().take(1).collect()
    } else if has_strong_seed {
        seed_pool.into_iter().take(2).collect()
    } else if let Some(scr) = screened_seeds {
        for (idx, &(s_eye, s_rot)) in scr.iter().take(2).enumerate() {
            if !seed_pool
                .iter()
                .any(|(_, c)| same_seed_neighborhood(c, s_eye, &s_rot))
            {
                seed_pool.push((
                    idx,
                    seed_only_candidate(s_eye, s_rot, k_render, render_size),
                ));
            }
        }
        let mut top: Vec<(usize, PoseCandidate)> = seed_pool.into_iter().take(2).collect();
        if top.first().is_none_or(|(_, c)| c.score.inliers < 8) {
            if let Some((n_eye, n_rot)) = closest_nadir
                && !top
                    .iter()
                    .any(|(_, c)| same_seed_neighborhood(c, n_eye, &n_rot))
            {
                top.insert(
                    0,
                    (0, seed_only_candidate(n_eye, n_rot, k_render, render_size)),
                );
            }
            top.truncate(1);
        }
        top
    } else {
        let polish_cap = if seed_pool
            .first()
            .is_some_and(|(_, c)| c.score.geometry_accepted)
        {
            1
        } else {
            3
        };
        seed_pool.into_iter().take(polish_cap).collect()
    };
    let mut polish_rays = 0_usize;
    let mut lg_layers = 0_usize;
    for (seed_idx, mut cur_cand) in selected.drain(..) {
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
            crate::realtime::helpers::log_attempt(
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
                if candidate_priority(&polished.score) > candidate_priority(&cur_cand.score) {
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
                || (p_step == 2 && polished_inliers < 32)
            {
                break;
            }
        }
        if best.as_ref().is_some_and(|(_, _, c)| {
            c.score.geometry_accepted && is_plausible_anchor_candidate(c, surface)
        }) {
            break;
        }
    }
    Ok((polish_rays, lg_layers))
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
