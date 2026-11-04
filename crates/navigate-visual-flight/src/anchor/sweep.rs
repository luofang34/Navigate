//! Two-stage coarse-to-fine focal-multiplier sweep for uncalibrated cold anchors.

use super::{
    AnchorAcquisition, AnchorQuery, AnchorQueryAux,
    basin::{classify_anchor_confidence, is_unambiguously_confirmed},
    chunks::eval_seed_chunks,
    eval::{SeedChunkOpts, lead_seed_count},
    is_plausible_anchor_candidate,
    nadir_screen::{
        NadirScreenRequest, closest_prior_nadir_seed, maybe_screen_low_agl_nadir_seeds,
    },
    polish::polish_anchor_seed_pool,
    scale_focal_matrix, scale_horizon_attitude,
    seeds::{
        AnchorSearchOpts, FocalSweepEstimator, SeedPose, build_stratified_coarse_anchor_seeds,
        cardinal_oblique_fallbacks, is_heading_known, resolve_prior_origin,
    },
    sweep_final::{finalize_coarse_multiplier, select_coarse_winners, select_sweep_outcome},
    virt_mask::{rank_oblique_fallbacks, subwindow_oblique_down_deg},
};
use crate::{
    diagnostics::AttemptRecord,
    error::FlightVisualError,
    runtime::LoadedXFeat,
    surface::{Surface, SurfaceGpuRenderer},
    types::AnchorConfidence,
};
use nalgebra::Matrix3;
use navigate_locate::corridor::plausibility::compute_holdout_rmse_px;
use navigate_match::{
    CorridorTileStore, HadamardProjector, KashinScratch, PreparedKashinQuery, SparseFeatures,
    SparseMatcher, VirtualCorridorView,
};
use navigate_visual::PoseCandidate;
use std::sync::Mutex;

const COARSE_FOCAL_SEED_CAP: usize = 12;

fn is_early_exit_focal_zero(cand: &PoseCandidate, surface: &Surface) -> bool {
    cand.score.geometry_accepted
        && is_plausible_anchor_candidate(cand, surface)
        && is_unambiguously_confirmed(&cand.score)
}

pub(super) struct CoarseMultiplierState {
    pub mult: f64,
    pub k_render_m: Matrix3<f64>,
    pub attitude: Option<(f64, f64)>,
    pub screened: Option<Vec<SeedPose>>,
    pub closest_nadir: Option<SeedPose>,
    pub oblique_fallbacks: Vec<SeedPose>,
    pub coarse_seeds: Vec<SeedPose>,
    pub remaining_seeds: Vec<SeedPose>,
    pub base_idx: usize,
    pub best: Option<(usize, usize, PoseCandidate)>,
    pub best_view: Option<VirtualCorridorView>,
    pub seed_pool: Vec<(usize, PoseCandidate)>,
    pub max_pairs: usize,
    pub polished_in_coarse: bool,
    pub was_polished: bool,
    pub confidence: Option<AnchorConfidence>,
}

impl CoarseMultiplierState {
    pub(super) fn intrinsics(&self) -> [f64; 4] {
        [
            self.k_render_m[(0, 0)],
            self.k_render_m[(1, 1)],
            self.k_render_m[(0, 2)],
            self.k_render_m[(1, 2)],
        ]
    }

    pub(super) fn max_inliers(&self) -> usize {
        let best_inl = self.best.as_ref().map_or(0, |(_, _, c)| c.score.inliers);
        let pool_inl = self.seed_pool.iter().map(|(_, c)| c.score.inliers).max();
        best_inl.max(pool_inl.unwrap_or(0))
    }

    pub(super) fn is_accepted(&self, surface: &Surface) -> bool {
        self.best.as_ref().is_some_and(|(_, _, c)| {
            c.score.geometry_accepted && is_plausible_anchor_candidate(c, surface)
        })
    }

    fn has_strong_lock(&self, surface: &Surface, min_inl: usize, min_hold: f64) -> bool {
        let mult_c = self.mult.clamp(1.0, 1.5);
        self.is_accepted(surface)
            && self.best.as_ref().is_some_and(|(_, _, c)| {
                let max_med = if c.score.inliers >= 28 {
                    11.5 * mult_c
                } else {
                    4.5 * mult_c
                };
                c.score.inliers >= min_inl
                    && c.score.holdout_inlier_fraction >= min_hold
                    && c.score.holdout_median_px <= max_med
            })
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "two-stage focal sweep borrows terrain, tile store, GPU renderer, models, query features, and intrinsics"
)]
pub(super) fn run_uncalibrated_focal_sweep(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    scratch: &mut KashinScratch,
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, matcher): (&LoadedXFeat, Option<&dyn SparseMatcher>),
    (query, opts): (&AnchorQuery, &AnchorSearchOpts),
    (qry_feat, prepared): (&SparseFeatures, &PreparedKashinQuery),
    (k_full, k_render, render_size): (&Matrix3<f64>, &Matrix3<f64>, [u32; 2]),
    aux: &AnchorQueryAux<'_>,
    mut attempts: Option<&mut Vec<AttemptRecord>>,
) -> Result<AnchorAcquisition, FlightVisualError> {
    let (_, _, _, agl) = resolve_prior_origin(surface, query.prior);
    let low_agl_nadir =
        query.attitude_override.is_none() && aux.live_attitude.is_none() && agl <= 250.0;
    let mut estimator = FocalSweepEstimator::for_regime(low_agl_nadir);
    let baseline_full_px = k_full[(0, 0)];
    let multipliers = estimator.sweep_multipliers;
    let max_lm = opts.query_keypoints.clamp(256, 2048);

    let priority: &[usize] = if opts.pace_realtime {
        &[0, 1, 2]
    } else if low_agl_nadir {
        &[0, 1, 2, 3]
    } else {
        &[0, 2, 1, 3]
    };

    let mut coarse: Vec<Option<CoarseMultiplierState>> =
        (0..multipliers.len()).map(|_| None).collect();
    let mut total_lg = 0_usize;
    let mut total_rays = 0_usize;

    for &idx in priority.iter().filter(|&&i| i < multipliers.len()) {
        if low_agl_nadir
            && opts.pace_realtime
            && idx == 2
            && coarse[0].as_ref().is_some_and(|s| s.max_inliers() < 6)
            && coarse[1].as_ref().is_some_and(|s| s.max_inliers() >= 8)
        {
            continue;
        }
        if !low_agl_nadir
            && opts.pace_realtime
            && idx == 2
            && coarse[0].as_ref().is_some_and(|s| s.screened.is_some())
        {
            continue;
        }
        let is_barren_slot = |s: &CoarseMultiplierState| {
            s.polished_in_coarse && !s.is_accepted(surface) && s.max_inliers() < 16
        };
        let barren_oblique_sweep = !low_agl_nadir
            && !opts.pace_realtime
            && coarse[0].as_ref().is_some_and(is_barren_slot)
            && coarse
                .get(2)
                .and_then(Option::as_ref)
                .is_some_and(is_barren_slot);
        let prior_oblique_accepted = !low_agl_nadir
            && !opts.pace_realtime
            && (idx == 1 || idx == 3)
            && (coarse[0].as_ref().is_some_and(|s| s.is_accepted(surface))
                || coarse
                    .get(2)
                    .and_then(Option::as_ref)
                    .is_some_and(|s| s.is_accepted(surface)));
        if (barren_oblique_sweep && idx == 3) || prior_oblique_accepted {
            continue;
        }
        let mult = multipliers[idx];
        let k_render_m = scale_focal_matrix(k_render, mult);
        let narrow_subwindow = aux
            .mask_small
            .is_some_and(|m| super::virt_mask::mask_valid_fraction(m) < 0.35);
        let sanitized_live = aux.live_attitude.map(|(d, r)| {
            if narrow_subwindow {
                (d.clamp(13.0, 17.0), r.clamp(-12.0, 12.0))
            } else {
                (d, r)
            }
        });
        let attitude = query
            .attitude_override
            .and_then(|a| scale_horizon_attitude(Some(a), mult))
            .or_else(|| scale_horizon_attitude(sanitized_live, mult));
        let intrinsics = [
            k_render_m[(0, 0)],
            k_render_m[(1, 1)],
            k_render_m[(0, 2)],
            k_render_m[(1, 2)],
        ];
        let cap = if idx != 0 && opts.pace_realtime {
            if low_agl_nadir { 12 } else { 8 }
        } else if barren_oblique_sweep && idx == 1 {
            6
        } else {
            COARSE_FOCAL_SEED_CAP
        };
        let screened = if aux.pred_pose.is_none() && attitude.is_none() {
            maybe_screen_low_agl_nadir_seeds(
                (surface, store),
                scratch,
                NadirScreenRequest {
                    prior: query.prior,
                    cold_prior: aux.cold_prior,
                    confirmatory_center_enu: aux.confirmatory_center_enu,
                    spatial_ring_enabled: true,
                    pace_realtime: opts.pace_realtime,
                    coarse_sweep: true,
                    post_screen_cap: opts.attempts.max(24),
                },
                prepared,
                intrinsics,
                render_size,
            )?
        } else {
            None
        };
        let closest_nadir = screened.as_ref().and_then(|s| {
            closest_prior_nadir_seed(surface, query.prior, aux.cold_prior, s.first().copied())
        });
        let (coarse_seeds, remaining_seeds) = if let Some(ref scr) = screened {
            let c_take = scr.len().min(cap);
            (scr[..c_take].to_vec(), scr[c_take..].to_vec())
        } else {
            build_stratified_coarse_anchor_seeds(
                surface,
                query.prior,
                opts,
                attitude,
                aux.pred_pose,
                cap,
            )
        };
        let oblique_fallbacks: Vec<SeedPose> = if screened.is_none() {
            if !is_heading_known(query.prior, opts) && agl > 250.0 {
                let down_deg = attitude.map_or_else(
                    || subwindow_oblique_down_deg(aux.mask_small, 22.0),
                    |(d, _)| d.clamp(8.0, 24.0),
                );
                let cardinals = cardinal_oblique_fallbacks(surface, query.prior, down_deg);
                if opts.pace_realtime {
                    rank_oblique_fallbacks(
                        store,
                        scratch,
                        prepared,
                        intrinsics,
                        render_size,
                        aux.mask_small,
                        cardinals,
                    )?
                } else {
                    let cap_ob = if idx == 2
                        && coarse[0]
                            .as_ref()
                            .is_some_and(|s| s.polished_in_coarse && s.max_inliers() == 0)
                    {
                        2
                    } else {
                        4
                    };
                    cardinals.into_iter().take(cap_ob).collect()
                }
            } else {
                coarse_seeds.iter().copied().take(3).collect()
            }
        } else {
            Vec::new()
        };
        let mut base_idx = 0_usize;
        let mut best: Option<(usize, usize, PoseCandidate)> = None;
        let mut best_view: Option<VirtualCorridorView> = None;
        let mut seed_pool: Vec<(usize, PoseCandidate)> = Vec::new();
        let mut lg_layers = 0_usize;
        let lead = lead_seed_count(query, opts, aux.pred_pose.is_some());
        let max_pairs = eval_seed_chunks(
            (surface, store, projector),
            gpu_renderer,
            (xfeat, matcher),
            (qry_feat, prepared),
            &coarse_seeds,
            (intrinsics, &k_render_m, render_size, max_lm),
            SeedChunkOpts::new(
                lead,
                true,
                idx == 0 || idx == 2,
                opts.pace_realtime,
                mult,
                aux.mask_small,
            ),
            (&mut base_idx, &mut lg_layers),
            (&mut seed_pool, &mut best, &mut best_view),
            attempts.as_deref_mut(),
        )?;
        total_lg += lg_layers;

        let mut state = CoarseMultiplierState {
            mult,
            k_render_m,
            attitude,
            screened,
            closest_nadir,
            oblique_fallbacks,
            coarse_seeds,
            remaining_seeds,
            base_idx,
            best,
            best_view,
            seed_pool,
            max_pairs,
            polished_in_coarse: false,
            was_polished: false,
            confidence: None,
        };
        let min_lock_inl = if opts.pace_realtime { 26 } else { 30 };
        let slot0_inl = coarse[0]
            .as_ref()
            .map_or(0, CoarseMultiplierState::max_inliers);
        let coarse0_accepted = coarse[0].as_ref().is_some_and(|s| s.is_accepted(surface));
        let coarse0_strong = coarse[0]
            .as_ref()
            .is_some_and(|s| s.has_strong_lock(surface, min_lock_inl, 0.35));
        let low_agl_slot = state.screened.is_some();
        let high_standoff_oblique =
            !low_agl_slot && !opts.pace_realtime && !state.oblique_fallbacks.is_empty();
        let can_early_polish = idx == 0
            || (low_agl_slot
                && (state.max_inliers() >= 8
                    || (idx == 1 && !coarse0_accepted && opts.pace_realtime)))
            || (idx == 2
                && (state.max_inliers() >= slot0_inl.saturating_mul(2)
                    || (!coarse0_strong && (state.max_inliers() >= 20 || high_standoff_oblique))));
        let mut coarse_early_done = if idx == 0 {
            state.has_strong_lock(surface, 48, 0.40)
                || state.has_strong_lock(surface, 36, 0.70)
                || (low_agl_slot && state.has_strong_lock(surface, 40, 0.45))
        } else {
            state.max_inliers() >= slot0_inl.saturating_mul(2)
                && state.has_strong_lock(surface, 36, 0.45)
        };
        if coarse_early_done {
            state.polished_in_coarse = true;
            state.confidence = state
                .best
                .as_ref()
                .map(|(_, _, c)| classify_anchor_confidence(c, &[]));
        } else if can_early_polish
            && opts.polish_anchors
            && (state.max_inliers() >= 20
                || high_standoff_oblique
                || (low_agl_slot
                    && (state.max_inliers() >= 6
                        || state.max_pairs >= 16
                        || (idx == 1 && !coarse0_accepted && opts.pace_realtime))))
        {
            let pol = polish_anchor_seed_pool(
                surface,
                gpu_renderer,
                xfeat,
                matcher,
                qry_feat,
                (
                    &state.k_render_m,
                    render_size,
                    state.mult,
                    opts.pace_realtime,
                    aux.mask_small,
                ),
                std::mem::take(&mut state.seed_pool),
                (
                    state.screened.as_deref(),
                    state.closest_nadir,
                    &state.oblique_fallbacks,
                    &state.coarse_seeds,
                ),
                &mut state.best,
                attempts.as_deref_mut(),
            )?;
            total_rays += pol.polish_rays;
            total_lg += pol.lg_layers;
            state.polished_in_coarse = true;
            state.was_polished |= pol.was_polished;
            state.confidence = pol.confidence;
            coarse_early_done = (state.was_polished
                && (state.has_strong_lock(surface, min_lock_inl.max(28), 0.35)
                    || state.has_strong_lock(surface, 28, 0.85)))
                || state.has_strong_lock(surface, 36, 0.45)
                || state.has_strong_lock(surface, min_lock_inl, 0.65);
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
    let winners = select_coarse_winners(
        surface,
        &coarse,
        priority,
        opts.pace_realtime,
        low_agl_nadir,
    );
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
            (xfeat, matcher),
            opts,
            (qry_feat, prepared),
            render_size,
            max_lm,
            aux,
            st,
            attempts.as_deref_mut(),
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
            let min_dom_inl = if opts.pace_realtime { 26 } else { 30 };
            let dominant_strong = (cand.score.inliers >= 48
                && cand.score.holdout_inlier_fraction >= 0.45)
                || (cand.score.inliers >= min_dom_inl
                    && cand.score.holdout_inlier_fraction >= 0.65
                    && cand.score.holdout_median_px <= 4.2);
            let dominant = win_pos == 0
                && ((dominant_strong
                    && coarse_inliers[w_idx] >= second_coarse_inl.saturating_mul(2).max(16))
                    || (acq.was_polished
                        && cand.score.inliers >= min_dom_inl.max(28)
                        && cand.score.holdout_inlier_fraction >= 0.35
                        && second_coarse_inl < 20));
            let paced_lock = opts.pace_realtime
                && win_pos == 0
                && cand.score.inliers >= 26
                && cand.score.holdout_inlier_fraction >= 0.40
                && (cand.score.holdout_median_px <= 4.5
                    || (cand.score.inliers >= 40 && cand.score.holdout_median_px <= 11.5));
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

    Ok(select_sweep_outcome(
        surface,
        &mut estimator,
        baseline_full_px,
        &scores,
        coarse,
        outcomes,
        (total_lg, total_rays),
    ))
}
