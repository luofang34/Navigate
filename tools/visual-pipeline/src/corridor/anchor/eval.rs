//! Seed-batch virtual-view projection, Kashin matching, and LighterGlue refinement.

use super::{is_plausible_anchor_candidate, is_sane_anchor_seed};
use crate::{
    PipelineError,
    corridor::{
        kinematics::{approx_dot, match_virtual_kashin},
        nadir::DENSE_NADIR_YAW_BANDS_DEG,
    },
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
use std::sync::Mutex;

pub(super) type SeedEvalTuple = (Option<PoseCandidate>, VirtualCorridorView, usize);

pub(super) fn eval_seed_batch(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, lighterglue): (&LoadedXFeat, Option<&LighterGlue>),
    (qry_feat, prepared): (&SparseFeatures, &PreparedKashinQuery),
    poses: &[([f64; 3], Matrix3<f64>)],
    (intrinsics, k_render, render_size, max_lm): ([f64; 4], &Matrix3<f64>, [u32; 2], usize),
    (fast_exit, allow_horizon_gpu): (bool, bool),
) -> Result<Vec<SeedEvalTuple>, PipelineError> {
    let k_cap = if lighterglue.is_some() { 640 } else { max_lm };
    let mut results: Vec<SeedEvalTuple> = Vec::with_capacity(poses.len());
    let mut need_lg: Vec<usize> = Vec::new();

    for (i, &(hyp_eye, hyp_rot)) in poses.iter().enumerate() {
        let sin_down = -hyp_rot[(2, 2)];
        let is_nadir = sin_down > 0.95;
        let min_cos_orientation = if is_nadir { 0.94 } else { 0.80 };
        let min_cos_incidence = if is_nadir {
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
                max_landmarks: max_lm,
                min_cos_incidence,
                min_cos_orientation,
            },
            projector,
        )?;
        if is_nadir && virt.world_points_enu.len() < 48 {
            virt = store.project_virtual_view(
                &VirtualViewCamera {
                    eye_enu: hyp_eye,
                    world_to_cam: rot_to_row_major(&hyp_rot),
                    intrinsics,
                    image_size: render_size,
                    max_landmarks: max_lm,
                    min_cos_incidence,
                    min_cos_orientation: 0.80,
                },
                projector,
            )?;
        }
        if virt.world_points_enu.len() < 12 || (allow_horizon_gpu && sin_down < 0.02) {
            if let Some(gpu_lock) = gpu_renderer
                && let Some((polished, _, layers)) = polish_pose_sparse(
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
                let acc_gpu = polished.score.geometry_accepted
                    && is_plausible_anchor_candidate(&polished, surface);
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

#[allow(clippy::too_many_arguments)]
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
    let polish_cap = if seed_pool
        .first()
        .is_some_and(|(_, c)| c.score.inliers >= 12)
    {
        1
    } else {
        2
    };
    let mut polish_rays = 0_usize;
    let mut lg_layers = 0_usize;
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
            crate::realtime::helpers::log_attempt(
                attempts_log,
                "corridor-polish-gpu-or-sparse+l1-gnc",
                seed_idx,
                p_step + 1,
                focal_mult as f32,
                polished.score.pairs,
                Some(&polished),
            );
            let acc_now = polished.score.geometry_accepted
                && is_plausible_anchor_candidate(&polished, surface);
            if is_sane_anchor_seed(&polished, surface) {
                if candidate_priority(&polished.score) >= candidate_priority(&cur_cand.score) {
                    cur_cand = polished.clone();
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
            if acc_now {
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

type SeedPose = ([f64; 3], Matrix3<f64>);
type RankedSeedPose = ((bool, i64), SeedPose);

pub(super) fn maybe_screen_low_agl_nadir_seeds(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    case: &PhotoCase,
    prepared: &PreparedKashinQuery,
    intrinsics: [f64; 4],
    render_size: [u32; 2],
    limit: usize,
) -> Result<Option<Vec<SeedPose>>, PipelineError> {
    let Some(pos) = case.prior.as_ref().and_then(|p| p.position.as_ref()) else {
        return Ok(None);
    };
    let Some(alt_m) = pos.altitude_m else {
        return Ok(None);
    };
    let Some(dem_h) = surface.sample_dem(pos.latitude, pos.longitude) else {
        return Ok(None);
    };
    let agl = (alt_m - dem_h).max(0.0);
    if agl > 250.0 {
        return Ok(None);
    }
    let center = surface.local(pos.latitude, pos.longitude, 0.0);
    let agl_center = (agl * 1.50).clamp(130.0, 175.0);
    let center_eye = surface.local(pos.latitude, pos.longitude, dem_h + agl_center);
    let cap = limit.max(28);
    let mut out: Vec<SeedPose> = Vec::with_capacity(cap);
    for &hdg in &DENSE_NADIR_YAW_BANDS_DEG {
        out.push((
            center_eye,
            surface.camera_rotation(pos.latitude, pos.longitude, hdg, 88.0, 0.0),
        ));
    }

    let step_xy = (agl * 0.70).clamp(58.0, 85.0);
    let agl_ring = (agl * 1.32).clamp(115.0, 155.0);
    let ring_offsets = [
        (1_i32, 0_i32),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ];
    let mut ring_scored: Vec<RankedSeedPose> = Vec::with_capacity(16);
    for (gx, gy) in ring_offsets {
        let [glat, glon, _] = surface.geodetic([
            center[0] + f64::from(gx) * step_xy,
            center[1] + f64::from(gy) * step_xy,
            0.0,
        ]);
        let Some(gdem) = surface.sample_dem(glat, glon) else {
            continue;
        };
        let geye = surface.local(glat, glon, gdem + agl_ring);
        let mut cell_yaws: Vec<RankedSeedPose> = Vec::with_capacity(12);
        for &hdg in &DENSE_NADIR_YAW_BANDS_DEG {
            let grot = surface.camera_rotation(glat, glon, hdg, 88.0, 0.0);
            let virt = store.project_virtual_view(
                &VirtualViewCamera {
                    eye_enu: geye,
                    world_to_cam: rot_to_row_major(&grot),
                    intrinsics,
                    image_size: render_size,
                    max_landmarks: 512,
                    min_cos_incidence: 0.45,
                    min_cos_orientation: 0.94,
                },
                projector,
            )?;
            if virt.world_points_enu.len() < 16 {
                continue;
            }
            let strict_cnt = match_virtual_kashin(&virt, prepared, 512).0.len();
            let smooth = kashin_mutual_top_sum(&virt, prepared, 512, 32);
            let rank = (
                strict_cnt >= 8,
                (smooth * 1_000.0).round() as i64 + (strict_cnt as i64) * 150,
            );
            cell_yaws.push((rank, (geye, grot)));
        }
        cell_yaws.sort_by_key(|a| std::cmp::Reverse(a.0));
        ring_scored.extend(cell_yaws.into_iter().take(2));
    }
    ring_scored.sort_by_key(|a| std::cmp::Reverse(a.0));
    out.extend(
        ring_scored
            .into_iter()
            .take(cap.saturating_sub(out.len()))
            .map(|(_, pose)| pose),
    );
    out.truncate(cap);
    Ok(Some(out))
}

fn kashin_mutual_top_sum(
    virt: &VirtualCorridorView,
    query: &PreparedKashinQuery,
    cap: usize,
    top_k: usize,
) -> f64 {
    let n_r = virt.sign_bits.len().min(cap);
    let n_q = query.valid_indices().len().min(cap);
    if n_r == 0 || n_q == 0 {
        return 0.0;
    }
    let q_signs = query.sign_bits();
    let q_rots = query.rotated();
    let mut fwd = vec![(usize::MAX, -2.0_f32); n_r];
    let mut rev = vec![(usize::MAX, -2.0_f32); n_q];
    for (r, slot) in fwd.iter_mut().enumerate() {
        let rs = virt.sign_bits[r];
        let rc = &virt.kashin_i8[r];
        for q in 0..n_q {
            if (rs ^ q_signs[q]).count_ones() > 26 {
                continue;
            }
            let dot = approx_dot(rc, &q_rots[q]);
            if dot > slot.1 {
                *slot = (q, dot);
            }
            if dot > rev[q].1 {
                rev[q] = (r, dot);
            }
        }
    }
    let mut dots: Vec<f32> = fwd
        .into_iter()
        .enumerate()
        .filter_map(|(r, (q, d))| (q < n_q && rev[q].0 == r && d >= 0.42).then_some(d))
        .collect();
    dots.sort_by(|a, b| b.total_cmp(a));
    dots.into_iter().take(top_k).map(f64::from).sum()
}
