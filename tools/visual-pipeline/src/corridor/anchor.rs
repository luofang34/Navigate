//! Virtual-view map-anchor acquisition against [`CorridorLandmarkBank`].

use super::CorridorTrackOptions;
use crate::{
    PipelineError,
    pose::{PoseCandidate, candidate_priority, fit_pose_hybrid, refine_seed_eye_2pt},
    realtime::helpers::{log_attempt, polish_pose_sparse, rot_to_row_major},
    stages::{MatchStageOptions, generate_seeds, load_horizon_attitude},
    surface::Surface,
    testset::PhotoCase,
};
use nalgebra::{Matrix3, Vector3};
use navigate_match::{
    ActiveTrackSet, CorridorLandmarkBank, FastMatchConfig, HadamardProjector, IndexedMatch,
    LighterGlue, PreparedKashinQuery, SparseFeatures, VirtualCorridorView, VirtualViewCamera,
    XFeat,
};
use serde_json::{Value, json};
use std::path::PathBuf;

pub(super) struct AnchorAcquisition {
    pub best: Option<(usize, usize, PoseCandidate)>,
    pub lg_layers: usize,
    pub polish_rays: usize,
    pub best_view: Option<VirtualCorridorView>,
}

pub(super) type AnchorSeedHint = (Option<([f64; 3], Matrix3<f64>)>, Option<(f64, f64)>);

pub(super) fn acquire_corridor_anchor(
    (surface, bank, projector): (&Surface, &CorridorLandmarkBank, &HadamardProjector),
    (xfeat, lighterglue): (&XFeat, Option<&LighterGlue>),
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
    let seeds: Vec<_> = pred_pose
        .into_iter()
        .chain(generate_seeds(surface, case, &stage_opts, attitude))
        .take(opts.anchor_attempts.max(1))
        .collect();
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

    for (seed_idx, &(hyp_eye, hyp_rot)) in seeds.iter().enumerate() {
        let (cand_opt, virt, layers) = eval_virtual_hypothesis(
            bank,
            projector,
            lighterglue,
            (qry_feat, prepared),
            (hyp_eye, hyp_rot),
            (intrinsics, k_render, render_size, max_lm),
            pred_pose.is_some() && seed_idx == 0,
        )?;
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

        let mut cur_cand = cand_opt.filter(|c| is_plausible_corridor_pose(c, hyp_eye));
        let mut cur_virt = virt;
        let mut prev_eye = hyp_eye;
        for re_step in 1..=2_usize {
            let Some(ref c_prev) = cur_cand else {
                break;
            };
            let shift_m = (Vector3::from(c_prev.eye) - Vector3::from(prev_eye)).norm();
            if c_prev.score.inliers < 10 || (c_prev.score.geometry_accepted && shift_m <= 120.0) {
                break;
            }
            prev_eye = c_prev.eye;
            let (re_cand, re_virt, re_layers) = eval_virtual_hypothesis(
                bank,
                projector,
                lighterglue,
                (qry_feat, prepared),
                (c_prev.eye, c_prev.rotation),
                (intrinsics, k_render, render_size, max_lm),
                false,
            )?;
            lg_layers += re_layers;
            let re_cand = re_cand.filter(|c| is_plausible_corridor_pose(c, hyp_eye));
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
                break;
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
        for (seed_idx, mut cur_cand) in seed_pool.into_iter().take(2) {
            let steps = if cur_cand.score.geometry_accepted {
                1
            } else {
                2
            };
            for p_step in 1..=steps {
                let Some((polished, rays, layers)) = polish_pose_sparse(
                    surface,
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
                    "corridor-polish-sparse+l1-gnc",
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

fn is_plausible_corridor_pose(c: &PoseCandidate, hyp_eye: [f64; 3]) -> bool {
    let horiz = (c.eye[0] - hyp_eye[0]).hypot(c.eye[1] - hyp_eye[1]);
    let dz = (c.eye[2] - hyp_eye[2]).abs();
    horiz.is_finite() && horiz <= 2_200.0 && dz <= 750.0 && c.score.image_support_fraction >= 0.025
}

fn eval_virtual_hypothesis(
    bank: &CorridorLandmarkBank,
    projector: &HadamardProjector,
    lighterglue: Option<&LighterGlue>,
    (qry_feat, prepared): (&SparseFeatures, &PreparedKashinQuery),
    (hyp_eye, hyp_rot): ([f64; 3], Matrix3<f64>),
    (intrinsics, k_render, render_size, max_lm): ([f64; 4], &Matrix3<f64>, [u32; 2], usize),
    fast_exit: bool,
) -> Result<(Option<PoseCandidate>, VirtualCorridorView, usize), PipelineError> {
    let virt = bank.project_virtual_view(
        &VirtualViewCamera {
            eye_enu: hyp_eye,
            world_to_cam: rot_to_row_major(&hyp_rot),
            intrinsics,
            image_size: render_size,
            max_landmarks: max_lm,
            min_cos_incidence: 0.84,
            min_cos_orientation: 0.90,
        },
        projector,
    );
    if virt.world_points_enu.len() < 16 {
        return Ok((None, virt, 0));
    }

    let k_cap = if lighterglue.is_some() { 640 } else { max_lm };
    let (k_world, k_px) = match_virtual_kashin(&virt, prepared, k_cap);
    let mut cand = fit_pose_hybrid(
        &k_world,
        &k_px,
        k_render,
        render_size,
        Some((hyp_eye, hyp_rot)),
    )
    .filter(|c| is_plausible_corridor_pose(c, hyp_eye));
    let mut layers = 0_usize;

    if !cand.as_ref().is_some_and(|c| c.score.geometry_accepted)
        && let Some(lg) = lighterglue
    {
        let base_cfg = if fast_exit {
            FastMatchConfig::realtime_tracking()
        } else {
            FastMatchConfig::balanced_search()
        };
        let cfg = FastMatchConfig {
            max_keypoints: max_lm,
            ..base_cfg
        };
        let (idx_matches, stats) =
            pollster::block_on(lg.match_features_indexed_fast(&virt.features, qry_feat, &cfg))?;
        layers = stats.layers_executed;
        let (w_pts, px) = collect_virtual_indexed(&virt, &idx_matches);
        let mut lg_best =
            fit_pose_hybrid(&w_pts, &px, k_render, render_size, Some((hyp_eye, hyp_rot)))
                .filter(|c| is_plausible_corridor_pose(c, hyp_eye));
        if !lg_best.as_ref().is_some_and(|c| c.score.geometry_accepted)
            && let Some(tri_init) = refine_seed_eye_2pt(&w_pts, &px, (hyp_eye, hyp_rot), k_render)
            && let Some(tri_cand) =
                fit_pose_hybrid(&w_pts, &px, k_render, render_size, Some(tri_init))
                    .filter(|c| is_plausible_corridor_pose(c, hyp_eye))
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
            cand = Some(lg_cand);
        }
    }
    Ok((cand, virt, layers))
}

fn match_virtual_kashin(
    virt: &VirtualCorridorView,
    query: &PreparedKashinQuery,
    cap: usize,
) -> (Vec<[f64; 3]>, Vec<[f64; 2]>) {
    let mut probe = ActiveTrackSet {
        world_points_enu: virt.world_points_enu.iter().take(cap).copied().collect(),
        sign_bits: virt.sign_bits.iter().take(cap).copied().collect(),
        kashin_i8: virt.kashin_i8.iter().take(cap).copied().collect(),
    };
    let n_r = probe.len();
    let q_valid = query.valid_indices();
    let n_q = q_valid.len().min(cap);
    if n_r == 0 || n_q == 0 {
        probe.world_points_enu.clear();
        return (Vec::new(), Vec::new());
    }
    let q_signs = query.sign_bits();
    let q_rots = query.rotated();
    let q_px = query.pixels();
    let mut fwd = vec![(usize::MAX, -2.0_f32, -2.0_f32); n_r];
    let mut rev = vec![(usize::MAX, -2.0_f32); n_q];

    for (r, fwd_slot) in fwd.iter_mut().enumerate() {
        let rs = probe.sign_bits[r];
        let rc = &probe.kashin_i8[r];
        for q in 0..n_q {
            if (rs ^ q_signs[q]).count_ones() > 24 {
                continue;
            }
            let dot = approx_dot(rc, &q_rots[q]);
            if dot > fwd_slot.1 {
                *fwd_slot = (q, dot, fwd_slot.1);
            } else if dot > fwd_slot.2 {
                fwd_slot.2 = dot;
            }
            if dot > rev[q].1 {
                rev[q] = (r, dot);
            }
        }
    }
    let mut world = Vec::new();
    let mut px = Vec::new();
    for (r, &(q, b1, b2)) in fwd.iter().enumerate() {
        if q < n_q
            && rev[q].0 == r
            && b1 >= 0.62
            && (b2 <= -1.5 || (1.0 - b1).max(0.0) <= 0.85 * (1.0 - b2).max(0.0))
        {
            world.push(probe.world_points_enu[r]);
            px.push(q_px[q_valid[q]]);
        }
    }
    (world, px)
}

fn approx_dot(code: &[u8; 64], rot: &[f32; 64]) -> f32 {
    let mut dot = 0.0_f32;
    let mut norm_sq = 0_i32;
    for (&raw, &q) in code.iter().zip(rot) {
        let v = raw as i8;
        let vi = i32::from(v);
        norm_sq += vi * vi;
        dot += f32::from(v) * q;
    }
    if norm_sq <= 0 {
        return 0.0;
    }
    (dot / (norm_sq as f32).sqrt()).clamp(-1.0, 1.0)
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
