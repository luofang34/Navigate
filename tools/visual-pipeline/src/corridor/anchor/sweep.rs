//! Two-stage coarse-to-fine focal-multiplier sweep for uncalibrated cold anchors.

use super::{
    AnchorAcquisition, AnchorQueryAux,
    eval::eval_global_fallback_if_needed,
    eval_seed_chunks, is_plausible_anchor_candidate,
    nadir_screen::{SeedPose, closest_prior_nadir_seed, maybe_screen_low_agl_nadir_seeds},
    plausibility::compute_holdout_rmse_px,
    polish::polish_anchor_seed_pool,
    scale_focal_matrix, scale_horizon_attitude, stage_match_options,
};
use crate::{
    PipelineError,
    corridor::CorridorTrackOptions,
    pose::{PoseCandidate, candidate_priority},
    stages::{
        focal::FocalLengthEstimator, load_horizon_attitude, seeds::build_stratified_coarse_seeds,
    },
    surface::{Surface, SurfaceGpuRenderer},
    testset::PhotoCase,
    xfeat_runtime::LoadedXFeat,
};
use nalgebra::Matrix3;
use navigate_match::{
    CorridorTileStore, HadamardProjector, LighterGlue, PreparedKashinQuery, SparseFeatures,
    VirtualCorridorView,
};
use serde_json::Value;
use std::{cmp::Reverse, sync::Mutex};

const COARSE_FOCAL_SEED_CAP: usize = 12;

pub(super) fn is_early_exit_focal_zero(cand: &PoseCandidate, surface: &Surface) -> bool {
    cand.score.geometry_accepted
        && is_plausible_anchor_candidate(cand, surface)
        && cand.score.inliers >= 48
        && cand.score.holdout_inlier_fraction >= 0.40
}

struct CoarseMultiplierState {
    mult: f64,
    k_render_m: Matrix3<f64>,
    attitude: Option<(f64, f64)>,
    screened: Option<Vec<SeedPose>>,
    closest_nadir: Option<SeedPose>,
    remaining_seeds: Vec<SeedPose>,
    base_idx: usize,
    best: Option<(usize, usize, PoseCandidate)>,
    best_view: Option<VirtualCorridorView>,
    seed_pool: Vec<(usize, PoseCandidate)>,
    max_pairs: usize,
    polished_in_coarse: bool,
}

impl CoarseMultiplierState {
    fn intrinsics(&self) -> [f64; 4] {
        [
            self.k_render_m[(0, 0)],
            self.k_render_m[(1, 1)],
            self.k_render_m[(0, 2)],
            self.k_render_m[(1, 2)],
        ]
    }

    fn max_inliers(&self) -> usize {
        let best_inl = self.best.as_ref().map_or(0, |(_, _, c)| c.score.inliers);
        let pool_inl = self.seed_pool.iter().map(|(_, c)| c.score.inliers).max();
        best_inl.max(pool_inl.unwrap_or(0))
    }

    fn is_accepted(&self, surface: &Surface) -> bool {
        self.best.as_ref().is_some_and(|(_, _, c)| {
            c.score.geometry_accepted && is_plausible_anchor_candidate(c, surface)
        })
    }

    fn has_strong_lock(&self, surface: &Surface, min_inl: usize, min_hold: f64) -> bool {
        self.is_accepted(surface)
            && self.best.as_ref().is_some_and(|(_, _, c)| {
                c.score.inliers >= min_inl && c.score.holdout_inlier_fraction >= min_hold
            })
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "two-stage focal sweep borrows terrain, tile store, GPU renderer, models, query features, and intrinsics"
)]
pub(super) fn run_uncalibrated_focal_sweep(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, lighterglue): (&LoadedXFeat, Option<&LighterGlue>),
    (case, opts): (&PhotoCase, &CorridorTrackOptions),
    (qry_feat, prepared): (&SparseFeatures, &PreparedKashinQuery),
    (k_full, k_render, render_size): (&Matrix3<f64>, &Matrix3<f64>, [u32; 2]),
    aux: &AnchorQueryAux<'_>,
    attempts_log: &mut Vec<Value>,
) -> Result<AnchorAcquisition, PipelineError> {
    let mut estimator = FocalLengthEstimator::default();
    let baseline_full_px = k_full[(0, 0)];
    let multipliers = estimator.sweep_multipliers;
    let stage_opts = stage_match_options(opts);
    let max_lm = opts.query_keypoints.clamp(256, 2048);

    let priority: &[usize] = if opts.pace_realtime {
        &[0, 1, 2]
    } else {
        &[0, 2, 1, 3]
    };
    let mut coarse: Vec<Option<CoarseMultiplierState>> =
        (0..multipliers.len()).map(|_| None).collect();
    let mut total_lg = 0_usize;
    let mut total_rays = 0_usize;

    for &idx in priority.iter().filter(|&&i| i < multipliers.len()) {
        if opts.pace_realtime
            && idx == 2
            && coarse[0].as_ref().is_some_and(|s| s.screened.is_some())
        {
            continue;
        }
        let mult = multipliers[idx];
        let k_full_m = scale_focal_matrix(k_full, mult);
        let k_render_m = scale_focal_matrix(k_render, mult);
        let scaled_live = scale_horizon_attitude(aux.live_attitude, mult);
        let attitude = load_horizon_attitude(&opts.dataset_dir, case, &k_full_m).or(scaled_live);
        let intrinsics = [
            k_render_m[(0, 0)],
            k_render_m[(1, 1)],
            k_render_m[(0, 2)],
            k_render_m[(1, 2)],
        ];
        let cap = if idx != 0 && opts.pace_realtime {
            8
        } else {
            COARSE_FOCAL_SEED_CAP
        };
        let screen_limit = if opts.pace_realtime {
            cap
        } else {
            opts.anchor_attempts
        };
        let screened = if aux.pred_pose.is_none() && attitude.is_none() {
            maybe_screen_low_agl_nadir_seeds(
                (surface, store, projector),
                case,
                prepared,
                intrinsics,
                render_size,
                screen_limit,
            )?
        } else {
            None
        };
        let closest_nadir = screened
            .as_ref()
            .and_then(|s| closest_prior_nadir_seed(surface, case, s.first().copied()));
        let (coarse_seeds, remaining_seeds) = if let Some(ref scr) = screened {
            let c_take = scr.len().min(cap);
            (scr[..c_take].to_vec(), scr[c_take..].to_vec())
        } else {
            build_stratified_coarse_seeds(surface, case, &stage_opts, attitude, aux.pred_pose, cap)
        };
        let mut base_idx = 0_usize;
        let mut best: Option<(usize, usize, PoseCandidate)> = None;
        let mut best_view: Option<VirtualCorridorView> = None;
        let mut seed_pool: Vec<(usize, PoseCandidate)> = Vec::new();
        let mut lg_layers = 0_usize;
        let max_pairs = eval_seed_chunks(
            (surface, store, projector),
            gpu_renderer,
            (xfeat, lighterglue),
            (qry_feat, prepared),
            &coarse_seeds,
            (intrinsics, &k_render_m, render_size, max_lm),
            (aux.pred_pose.is_some(), true, idx == 0 || idx == 2, mult),
            (&mut base_idx, &mut lg_layers),
            (&mut seed_pool, &mut best, &mut best_view),
            attempts_log,
        )?;
        total_lg += lg_layers;

        let mut state = CoarseMultiplierState {
            mult,
            k_render_m,
            attitude,
            screened,
            closest_nadir,
            remaining_seeds,
            base_idx,
            best,
            best_view,
            seed_pool,
            max_pairs,
            polished_in_coarse: false,
        };
        let slot0_inl = coarse[0]
            .as_ref()
            .map_or(0, CoarseMultiplierState::max_inliers);
        let low_agl_slot0 = idx == 0 && state.screened.is_some();
        let can_early_polish =
            idx == 0 || (idx == 2 && state.max_inliers() >= slot0_inl.saturating_mul(2));
        let mut coarse_early_done = if idx == 0 {
            state.has_strong_lock(surface, 48, 0.40)
                || (low_agl_slot0 && state.has_strong_lock(surface, 26, 0.40))
        } else {
            idx == 2
                && state.max_inliers() >= slot0_inl.saturating_mul(2)
                && state.has_strong_lock(surface, 48, 0.45)
        };
        if coarse_early_done {
            state.polished_in_coarse = true;
        } else if can_early_polish
            && opts.polish_anchors
            && (state.max_inliers() >= 20
                || (low_agl_slot0 && (state.max_inliers() >= 6 || opts.pace_realtime)))
        {
            let (rays, p_layers) = polish_anchor_seed_pool(
                surface,
                gpu_renderer,
                xfeat,
                lighterglue,
                qry_feat,
                &state.k_render_m,
                render_size,
                state.mult,
                std::mem::take(&mut state.seed_pool),
                (state.screened.as_deref(), state.closest_nadir),
                &mut state.best,
                attempts_log,
            )?;
            total_rays += rays;
            total_lg += p_layers;
            state.polished_in_coarse = true;
            coarse_early_done = if idx == 0 {
                state.has_strong_lock(surface, 48, 0.40)
                    || (low_agl_slot0 && state.has_strong_lock(surface, 26, 0.40))
            } else {
                state.has_strong_lock(surface, 48, 0.45)
            };
        }
        coarse[idx] = Some(state);
        if coarse_early_done {
            break;
        }
    }

    let coarse_inliers: Vec<usize> = coarse
        .iter()
        .map(|s| s.as_ref().map_or(0, CoarseMultiplierState::max_inliers))
        .collect();
    let mut order: Vec<usize> = (0..coarse.len()).filter(|&i| coarse[i].is_some()).collect();
    order.sort_by_key(|&idx| {
        let rank = coarse[idx].as_ref().map_or((false, 0, 0), |st| {
            (st.is_accepted(surface), st.max_inliers(), st.max_pairs)
        });
        Reverse((rank, idx == 0, Reverse(idx)))
    });
    let mut winners: Vec<usize> = order.into_iter().take(2).collect();
    if winners.len() == 2
        && winners[1] == 0
        && let (Some(w0), Some(s0)) = (coarse[winners[0]].as_ref(), coarse[0].as_ref())
        && !w0.is_accepted(surface)
        && s0.max_inliers() + 4 >= w0.max_inliers()
        && s0.max_pairs * 4 >= w0.max_pairs * 3
    {
        winners.swap(0, 1);
    }
    if opts.pace_realtime
        && winners.len() == 2
        && !coarse[winners[1]]
            .as_ref()
            .is_some_and(|s| s.max_inliers() >= 14)
    {
        winners.truncate(1);
    }
    let second_coarse_inl = winners.get(1).map_or(0, |&idx2| coarse_inliers[idx2]);
    let mut scores = [f64::INFINITY; 4];
    let mut outcomes: Vec<Option<AnchorAcquisition>> =
        (0..multipliers.len()).map(|_| None).collect();

    for (win_pos, &w_idx) in winners.iter().enumerate() {
        let Some(st) = coarse[w_idx].take() else {
            continue;
        };
        let acq = finalize_coarse_multiplier(
            (surface, store, projector),
            gpu_renderer,
            (xfeat, lighterglue),
            (case, opts),
            (qry_feat, prepared),
            render_size,
            max_lm,
            aux,
            st,
            attempts_log,
        )?;
        total_lg += acq.lg_layers;
        total_rays += acq.polish_rays;
        let mut early_done = false;
        if let Some((_, _, ref cand)) = acq.best
            && cand.score.geometry_accepted
            && is_plausible_anchor_candidate(cand, surface)
        {
            let mult = multipliers[w_idx];
            let ang_rmse = compute_holdout_rmse_px(cand) / mult.max(0.25);
            scores[w_idx] = ang_rmse / (cand.score.inliers.max(1) as f64).sqrt();
            let dominant = win_pos == 0
                && cand.score.inliers >= 48
                && cand.score.holdout_inlier_fraction >= 0.45
                && coarse_inliers[w_idx] >= second_coarse_inl.saturating_mul(2).max(16);
            let paced_lock = opts.pace_realtime
                && win_pos == 0
                && cand.score.inliers >= 26
                && cand.score.holdout_inlier_fraction >= 0.40;
            early_done = (w_idx == 0 && is_early_exit_focal_zero(cand, surface))
                || dominant
                || paced_lock
                || (cand.score.inliers >= 80 && cand.score.holdout_inlier_fraction >= 0.45);
        }
        outcomes[w_idx] = Some(acq);
        if early_done {
            break;
        }
    }

    for (idx, rem) in coarse.into_iter().enumerate() {
        if outcomes[idx].is_none()
            && let Some(mut st) = rem
        {
            if let Some((_, _, ref mut c)) = st.best
                && !is_plausible_anchor_candidate(c, surface)
            {
                c.score.geometry_accepted = false;
            }
            outcomes[idx] = Some(AnchorAcquisition {
                best: st.best,
                lg_layers: 0,
                polish_rays: 0,
                best_view: st.best_view,
                locked_focal_px: None,
            });
        }
    }

    if scores.iter().any(|s| s.is_finite()) {
        let locked_px = estimator.select_multiplier(&scores, baseline_full_px);
        let best_idx = multipliers
            .iter()
            .position(|&m| (baseline_full_px * m - locked_px).abs() < 1e-6)
            .unwrap_or(0);
        if let Some(mut chosen) = outcomes.get_mut(best_idx).and_then(Option::take) {
            chosen.lg_layers = total_lg;
            chosen.polish_rays = total_rays;
            chosen.locked_focal_px = Some(locked_px);
            return Ok(chosen);
        }
    }

    let best_fb = outcomes
        .into_iter()
        .flatten()
        .filter(|a| a.best.is_some())
        .max_by_key(|a| {
            a.best
                .as_ref()
                .map(|(_, _, c)| candidate_priority(&c.score))
        });
    Ok(AnchorAcquisition {
        best: best_fb.as_ref().and_then(|a| a.best.clone()),
        lg_layers: total_lg,
        polish_rays: total_rays,
        best_view: best_fb.and_then(|a| a.best_view),
        locked_focal_px: None,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "winning-multiplier finalization runs GPU polish and optional remaining-seed fallback"
)]
fn finalize_coarse_multiplier(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, lighterglue): (&LoadedXFeat, Option<&LighterGlue>),
    (case, opts): (&PhotoCase, &CorridorTrackOptions),
    (qry_feat, prepared): (&SparseFeatures, &PreparedKashinQuery),
    render_size: [u32; 2],
    max_lm: usize,
    aux: &AnchorQueryAux<'_>,
    mut st: CoarseMultiplierState,
    attempts_log: &mut Vec<Value>,
) -> Result<AnchorAcquisition, PipelineError> {
    let mut lg_layers = 0_usize;
    let mut polish_rays = 0_usize;
    if opts.polish_anchors && !st.polished_in_coarse {
        let (rays, p_layers) = polish_anchor_seed_pool(
            surface,
            gpu_renderer,
            xfeat,
            lighterglue,
            qry_feat,
            &st.k_render_m,
            render_size,
            st.mult,
            std::mem::take(&mut st.seed_pool),
            (st.screened.as_deref(), st.closest_nadir),
            &mut st.best,
            attempts_log,
        )?;
        polish_rays += rays;
        lg_layers += p_layers;
    }

    let accepted_after_polish = st.best.as_ref().is_some_and(|(_, _, c)| {
        c.score.geometry_accepted && is_plausible_anchor_candidate(c, surface)
    });
    if !accepted_after_polish {
        if opts.pace_realtime {
            if st.screened.is_some() {
                st.remaining_seeds.clear();
            } else {
                st.remaining_seeds.truncate(8);
            }
        }
        let intrinsics = st.intrinsics();
        let mut extra_pool: Vec<(usize, PoseCandidate)> = Vec::new();
        if !st.remaining_seeds.is_empty() {
            let mut add_lg = 0_usize;
            eval_seed_chunks(
                (surface, store, projector),
                gpu_renderer,
                (xfeat, lighterglue),
                (qry_feat, prepared),
                &st.remaining_seeds,
                (intrinsics, &st.k_render_m, render_size, max_lm),
                (false, false, true, st.mult),
                (&mut st.base_idx, &mut add_lg),
                (&mut extra_pool, &mut st.best, &mut st.best_view),
                attempts_log,
            )?;
            lg_layers += add_lg;
        }
        eval_global_fallback_if_needed(
            (surface, store, projector),
            gpu_renderer,
            (xfeat, lighterglue),
            (case, opts),
            (qry_feat, prepared),
            (intrinsics, &st.k_render_m, render_size, max_lm),
            aux,
            st.attitude,
            (&mut st.base_idx, &mut lg_layers),
            (&mut extra_pool, &mut st.best, &mut st.best_view),
        )?;
        if opts.polish_anchors && !extra_pool.is_empty() {
            let (rays, p_layers) = polish_anchor_seed_pool(
                surface,
                gpu_renderer,
                xfeat,
                lighterglue,
                qry_feat,
                &st.k_render_m,
                render_size,
                st.mult,
                extra_pool,
                (None, None),
                &mut st.best,
                attempts_log,
            )?;
            polish_rays += rays;
            lg_layers += p_layers;
        }
    }

    if let Some((_, _, ref mut cand)) = st.best
        && !is_plausible_anchor_candidate(cand, surface)
    {
        cand.score.geometry_accepted = false;
    }
    Ok(AnchorAcquisition {
        best: st.best,
        lg_layers,
        polish_rays,
        best_view: st.best_view,
        locked_focal_px: None,
    })
}
