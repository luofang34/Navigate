//! Multi-hypothesis GPU and sparse terrain polish pool selection and basin ranking.

pub(crate) use super::basin::{is_fragile_virtual_anchor, seed_refinement_priority};
use super::{
    basin::{
        COMPETING_BASIN_SEPARATION_M, POLISH_DIVERSITY_RADIUS_M, basin_priority_key,
        clamp_screened_step1_seed, classify_anchor_confidence, is_unambiguously_confirmed,
        seed_only_candidate, veto_unverified_virtual_best, within_pose_neighborhood,
    },
    is_plausible_anchor_candidate, is_sane_anchor_seed,
    nadir_screen::select_nadir_polish_fallbacks,
    record_attempt,
    seeds::SeedPose,
    sparse_polish::polish_pose_sparse_masked,
};
use crate::{
    diagnostics::AttemptRecord,
    error::FlightVisualError,
    runtime::LoadedXFeat,
    surface::{Surface, SurfaceGpuRenderer},
    types::AnchorConfidence,
};
use image::GrayImage;
use nalgebra::Matrix3;
use navigate_match::{SparseFeatures, SparseMatcher};
use navigate_visual::{PoseCandidate, holdout_pnp::candidate_priority};
use std::{cmp::Reverse, sync::Mutex};

/// Summary of a [`polish_anchor_seed_pool`] pass.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PolishPoolOutcome {
    pub polish_rays: usize,
    pub lg_layers: usize,
    pub was_polished: bool,
    pub confidence: Option<AnchorConfidence>,
}

/// Seed lists and fallback poses for [`polish_anchor_seed_pool`].
pub(crate) type PolishSeedRefs<'a> = (
    Option<&'a [SeedPose]>,
    Option<SeedPose>,
    &'a [SeedPose],
    &'a [SeedPose],
);

#[expect(
    clippy::too_many_arguments,
    reason = "multi-hypothesis anchor polish coordinates surface, GPU renderer, models, fallbacks, and attempt telemetry"
)]
pub(crate) fn polish_anchor_seed_pool(
    surface: &Surface,
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    xfeat: &LoadedXFeat,
    matcher: Option<&dyn SparseMatcher>,
    qry_feat: &SparseFeatures,
    (k_render, render_size, focal_mult, pace_realtime, mask_small): (
        &Matrix3<f64>,
        [u32; 2],
        f64,
        bool,
        Option<&GrayImage>,
    ),
    mut seed_pool: Vec<(usize, PoseCandidate)>,
    (screened_seeds, closest_nadir, oblique_fallbacks, all_seeds): PolishSeedRefs<'_>,
    best: &mut Option<(usize, usize, PoseCandidate)>,
    mut attempts: Option<&mut Vec<AttemptRecord>>,
) -> Result<PolishPoolOutcome, FlightVisualError> {
    seed_pool.sort_by_key(|a| Reverse(seed_refinement_priority(&a.1.score)));
    let max_pool_inliers = seed_pool
        .iter()
        .map(|(_, c)| c.score.inliers)
        .max()
        .unwrap_or(0);
    let best_unambiguous = best.as_ref().is_some_and(|(_, _, c)| {
        is_plausible_anchor_candidate(c, surface) && is_unambiguously_confirmed(&c.score)
    });
    let warm_single_seed = pace_realtime && screened_seeds.is_none() && all_seeds.len() == 1;
    if let Some((_, step, ref c)) = *best
        && (step >= 2 || warm_single_seed)
        && c.score.geometry_accepted
        && is_plausible_anchor_candidate(c, surface)
        && (!is_fragile_virtual_anchor(&c.score) || best_unambiguous)
    {
        return Ok(PolishPoolOutcome {
            polish_rays: 0,
            lg_layers: 0,
            was_polished: step >= 2,
            confidence: Some(classify_anchor_confidence(c, &[])),
        });
    }

    let subwindow_mask =
        mask_small.is_some_and(|m| super::virt_mask::mask_valid_fraction(m) < 0.70);
    let narrow_mask = mask_small.is_some_and(|m| super::virt_mask::mask_valid_fraction(m) < 0.35);
    let mut selected = select_diverse_polish_candidates(
        (best_unambiguous, pace_realtime, narrow_mask),
        max_pool_inliers,
        seed_pool,
        (screened_seeds, closest_nadir, oblique_fallbacks),
        k_render,
        render_size,
    );

    let mut visited: Vec<SeedPose> = Vec::with_capacity(8);
    let mut evaluated_basins: Vec<(usize, bool, PoseCandidate)> = Vec::with_capacity(6);
    if let Some((pre_seed, _, ref pre)) = *best {
        let pre_acc = pre.score.geometry_accepted && is_plausible_anchor_candidate(pre, surface);
        if pre_acc || pre.score.inliers >= 20 {
            evaluated_basins.push((pre_seed, pre_acc, pre.clone()));
        }
    }

    let mut polish_rays = 0_usize;
    let mut lg_layers = 0_usize;
    let mut was_polished = false;
    let paced_nadir = pace_realtime && screened_seeds.is_some();
    let mut polish_calls = 0_usize;

    for (seed_idx, mut cur_cand) in selected.drain(..) {
        if paced_nadir && polish_calls >= 2 {
            break;
        }
        if visited.iter().any(|&(ve, ref vr)| {
            within_pose_neighborhood(
                cur_cand.eye,
                &cur_cand.rotation,
                ve,
                vr,
                POLISH_DIVERSITY_RADIUS_M,
            )
        }) {
            continue;
        }
        let (mut orig_eye, mut orig_rot) = (cur_cand.eye, cur_cand.rotation);
        let fragile_virtual = is_fragile_virtual_anchor(&cur_cand.score);
        let mut max_polished_inliers = 0_usize;
        visited.push((orig_eye, orig_rot));
        let steps = if cur_cand.score.geometry_accepted || warm_single_seed {
            1
        } else {
            3
        };
        let clean_seed = screened_seeds
            .and_then(|s| s.get(seed_idx).copied())
            .or_else(|| all_seeds.get(seed_idx).copied());
        for p_step in 1..=steps {
            let allow_converge = p_step >= 2
                && polish_calls < 4
                && cur_cand.score.inliers >= 14
                && (cur_cand.score.holdout_median_px <= 14.0
                    || (cur_cand.score.inliers >= 24 && cur_cand.score.holdout_median_px <= 18.0));
            if paced_nadir && polish_calls >= 2 && !allow_converge {
                break;
            }
            polish_calls += 1;
            let barren_subwindow = !pace_realtime
                && screened_seeds.is_none()
                && subwindow_mask
                && polish_calls >= 2
                && p_step == 1
                && best.as_ref().is_none_or(|(_, _, c)| c.score.inliers < 8);
            let use_compact = (paced_nadir && polish_calls >= 2 && p_step == 1) || barren_subwindow;
            let mut pol_res = polish_pose_sparse_masked(
                surface,
                gpu_renderer,
                xfeat,
                matcher,
                qry_feat,
                (cur_cand.eye, cur_cand.rotation),
                k_render,
                render_size,
                (mask_small, pace_realtime, use_compact),
            )?;
            let allow_clean_retry = (!pace_realtime
                && (screened_seeds.is_some()
                    || cur_cand.score.pairs > 0
                    || oblique_fallbacks.len() > 4))
                || (pace_realtime
                    && cur_cand.score.pairs > 0
                    && (screened_seeds.is_none() || polish_calls < 3));
            if p_step == 1
                && allow_clean_retry
                && pol_res.as_ref().is_none_or(|(c, _, _)| {
                    c.score.inliers < 14 || (!pace_realtime && !is_sane_anchor_seed(c, surface))
                })
                && let Some((se, sr)) = clean_seed
                && !within_pose_neighborhood(cur_cand.eye, &cur_cand.rotation, se, &sr, 8.0)
            {
                polish_calls += 1;
                visited.push((se, sr));
                if let Some((sc, srays, slayers)) = polish_pose_sparse_masked(
                    surface,
                    gpu_renderer,
                    xfeat,
                    matcher,
                    qry_feat,
                    (se, sr),
                    k_render,
                    render_size,
                    (mask_small, pace_realtime, false),
                )? {
                    polish_rays += srays;
                    lg_layers += slayers;
                    if pol_res.as_ref().is_none_or(|(prev, _, _)| {
                        candidate_priority(&sc.score) > candidate_priority(&prev.score)
                    }) {
                        if !pace_realtime {
                            (orig_eye, orig_rot) = (se, sr);
                        }
                        pol_res = Some((sc, 0, 0));
                    }
                }
            }
            let Some((polished, rays, layers)) = pol_res else {
                break;
            };
            polish_rays += rays;
            lg_layers += layers;
            if let Some(log) = attempts.as_deref_mut() {
                record_attempt(
                    log,
                    "corridor-polish-gpu-or-sparse+l1-gnc",
                    seed_idx,
                    p_step + 1,
                    focal_mult as f32,
                    polished.score.pairs,
                    Some(&polished),
                    None,
                );
            }
            let polished_inliers = polished.score.inliers;
            let polished_pairs = polished.score.pairs;
            let polished_support = polished.score.image_support_fraction;
            let polished_hm = polished.score.holdout_median_px;
            let acc_now = polished.score.geometry_accepted
                && is_plausible_anchor_candidate(&polished, surface);
            let mut improved_cur = false;
            if is_sane_anchor_seed(&polished, surface) {
                max_polished_inliers = max_polished_inliers.max(polished_inliers);
                evaluated_basins.push((seed_idx, acc_now, polished.clone()));
                let beat_cur = candidate_priority(&polished.score)
                    > candidate_priority(&cur_cand.score)
                    || (p_step == 1
                        && !cur_cand.score.geometry_accepted
                        && (polished_inliers >= 10
                            || (cur_cand.score.inliers < 12 && polished_inliers >= 8)));
                if beat_cur {
                    cur_cand = polished.clone();
                    let min_clamp_hm = if pace_realtime { 18.0 } else { 10.0 };
                    if screened_seeds.is_some()
                        && p_step == 1
                        && !acc_now
                        && polished_inliers < 14
                        && polished.score.holdout_median_px > min_clamp_hm
                    {
                        clamp_screened_step1_seed(&mut cur_cand, orig_eye, orig_rot, pace_realtime);
                    }
                    improved_cur = true;
                }
                if best.as_ref().is_none_or(|(_, _, prev)| {
                    let prev_acc = prev.score.geometry_accepted
                        && is_plausible_anchor_candidate(prev, surface);
                    basin_priority_key(acc_now, &polished) >= basin_priority_key(prev_acc, prev)
                }) {
                    *best = Some((seed_idx, p_step + 1, polished));
                    was_polished = true;
                }
            }
            let step1_min = if !pace_realtime {
                if screened_seeds.is_some() && polished_hm > 40.0 {
                    24
                } else if screened_seeds.is_some() && polished_hm > 25.0 {
                    16
                } else {
                    8
                }
            } else if paced_nadir {
                if polished_hm <= 14.0 { 10 } else { 16 }
            } else {
                12
            };
            let step2_min = if paced_nadir {
                if polished_hm <= 14.0 { 14 } else { 22 }
            } else {
                18
            };
            let refine_narrow_acc = !pace_realtime
                && (subwindow_mask || polished_support < 0.15)
                && p_step == 1
                && polished_inliers < 32
                && polished_pairs >= 36;
            if (acc_now && !refine_narrow_acc)
                || !improved_cur
                || (p_step == 1 && polished_inliers < step1_min)
                || (p_step == 2 && polished_inliers < step2_min)
            {
                break;
            }
        }
        if fragile_virtual && max_polished_inliers < 16 {
            veto_unverified_virtual_best(best, seed_idx, orig_eye, &orig_rot);
        }
        if pace_realtime || cur_cand.score.geometry_accepted {
            visited.push((cur_cand.eye, cur_cand.rotation));
        }
        let fm_scale = if pace_realtime {
            1.0
        } else {
            focal_mult.clamp(1.0, 1.5)
        };
        if best.as_ref().is_some_and(|(_, _, c)| {
            let s = &c.score;
            s.geometry_accepted
                && is_plausible_anchor_candidate(c, surface)
                && (is_unambiguously_confirmed(s)
                    || (screened_seeds.is_some()
                        && s.inliers >= 26
                        && s.holdout_inlier_fraction >= 0.45
                        && s.holdout_median_px <= 11.5
                        && s.train_rms_px <= 6.5)
                    || (screened_seeds.is_none()
                        && ((!is_fragile_virtual_anchor(s)
                            && s.inliers >= 48
                            && s.train_rms_px <= 5.5)
                            || (s.inliers >= 28
                                && (pace_realtime
                                    || s.inliers >= 30
                                    || s.image_support_fraction >= 0.042
                                    || (s.holdout_inlier_fraction >= 0.85
                                        && s.holdout_median_px <= 4.0))
                                && s.holdout_inlier_fraction >= 0.35
                                && s.holdout_median_px <= 9.5 * fm_scale
                                && s.train_rms_px <= 5.5 * fm_scale))))
        }) {
            break;
        }
    }

    let confidence = best.as_ref().and_then(|(win_seed, _, win)| {
        (win.score.geometry_accepted && is_plausible_anchor_candidate(win, surface)).then(|| {
            let competing: Vec<(bool, PoseCandidate)> = evaluated_basins
                .into_iter()
                .filter(|(s_idx, _, _)| *s_idx != *win_seed)
                .map(|(_, is_acc, c)| (is_acc, c))
                .collect();
            classify_anchor_confidence(win, &competing)
        })
    });
    Ok(PolishPoolOutcome {
        polish_rays,
        lg_layers,
        was_polished,
        confidence,
    })
}

fn select_diverse_polish_candidates(
    (best_unambiguous, pace_realtime, narrow_mask): (bool, bool, bool),
    max_pool_inliers: usize,
    seed_pool: Vec<(usize, PoseCandidate)>,
    (screened_seeds, closest_nadir, oblique_fallbacks): (
        Option<&[SeedPose]>,
        Option<SeedPose>,
        &[SeedPose],
    ),
    k_render: &Matrix3<f64>,
    render_size: [u32; 2],
) -> Vec<(usize, PoseCandidate)> {
    if best_unambiguous {
        return seed_pool.into_iter().take(1).collect();
    }
    if let Some(scr) = screened_seeds {
        let mut top: Vec<(usize, PoseCandidate)> = Vec::with_capacity(4);
        for (idx, mut cand) in seed_pool {
            let coherent_virt = (!pace_realtime && cand.score.inliers >= 7)
                || (cand.score.inliers >= 8 && cand.score.holdout_median_px <= 55.0)
                || (cand.score.inliers >= 6 && cand.score.holdout_median_px <= 42.0);
            if !coherent_virt {
                continue;
            }
            if !cand.score.geometry_accepted
                && let Some(&(se, sr)) = scr.get(idx)
                && (cand.score.inliers < 6
                    || (cand.score.inliers == 6 && cand.score.holdout_inlier_fraction < 0.25)
                    || (pace_realtime
                        && cand.score.inliers < 8
                        && cand.score.holdout_median_px > 42.0)
                    || -cand.rotation[(2, 2)] < 0.88
                    || (cand.eye[0] - se[0]).hypot(cand.eye[1] - se[1]) > 95.0
                    || (cand.eye[2] - se[2]).abs() > 65.0)
            {
                cand.eye = se;
                cand.rotation = sr;
            }
            if !top.iter().any(|(_, c)| {
                within_pose_neighborhood(
                    c.eye,
                    &c.rotation,
                    cand.eye,
                    &cand.rotation,
                    POLISH_DIVERSITY_RADIUS_M,
                )
            }) {
                top.push((idx, cand));
                if top.len() >= 2 {
                    break;
                }
            }
        }
        let fb_cap = 4;
        let total_cap = if pace_realtime { 2 } else { 3 };
        for (idx, (s_eye, s_rot)) in select_nadir_polish_fallbacks(scr, closest_nadir, fb_cap) {
            if top.len() >= total_cap {
                break;
            }
            let sep = if idx == 0 && pace_realtime {
                COMPETING_BASIN_SEPARATION_M
            } else {
                POLISH_DIVERSITY_RADIUS_M
            };
            if !top
                .iter()
                .any(|(_, c)| within_pose_neighborhood(c.eye, &c.rotation, s_eye, &s_rot, sep))
            {
                top.push((
                    idx,
                    seed_only_candidate(s_eye, s_rot, k_render, render_size),
                ));
            }
        }
        return top;
    }

    let mut top: Vec<(usize, PoseCandidate)> = Vec::with_capacity(5);
    for (idx, cand) in seed_pool {
        if cand.score.inliers < 6 || (cand.score.inliers == 6 && cand.score.pairs < 16) {
            continue;
        }
        if !top.iter().any(|(_, c)| {
            within_pose_neighborhood(
                c.eye,
                &c.rotation,
                cand.eye,
                &cand.rotation,
                POLISH_DIVERSITY_RADIUS_M,
            )
        }) {
            top.push((idx, cand));
            if top.len() >= 3 {
                break;
            }
        }
    }
    if max_pool_inliers < 16 && !oblique_fallbacks.is_empty() {
        let base_ob = if pace_realtime { 2 } else { 4 };
        let ob_cap = (if narrow_mask && oblique_fallbacks.len() > 4 {
            base_ob * 2
        } else {
            base_ob
        })
        .max(top.len() + 1);
        for (idx, &(f_eye, f_rot)) in oblique_fallbacks.iter().enumerate() {
            if top.len() >= ob_cap {
                break;
            }
            if !top.iter().any(|(_, c)| {
                within_pose_neighborhood(
                    c.eye,
                    &c.rotation,
                    f_eye,
                    &f_rot,
                    POLISH_DIVERSITY_RADIUS_M,
                )
            }) {
                top.push((
                    idx,
                    seed_only_candidate(f_eye, f_rot, k_render, render_size),
                ));
            }
        }
    }
    top
}
