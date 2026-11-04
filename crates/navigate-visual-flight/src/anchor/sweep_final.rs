//! Stage-two finalization for winning focal-multiplier hypotheses.

use super::{
    AnchorAcquisition, AnchorQueryAux,
    basin::classify_anchor_confidence,
    chunks::eval_seed_chunks,
    eval::{SeedChunkOpts, eval_global_fallback_if_needed},
    is_plausible_anchor_candidate,
    polish::polish_anchor_seed_pool,
    seeds::{AnchorSearchOpts, FocalSweepEstimator},
    sweep::CoarseMultiplierState,
};
use crate::{
    diagnostics::AttemptRecord,
    error::FlightVisualError,
    runtime::LoadedXFeat,
    surface::{Surface, SurfaceGpuRenderer},
};
use navigate_match::{
    CorridorTileStore, HadamardProjector, PreparedKashinQuery, SparseFeatures, SparseMatcher,
};
use navigate_visual::{PoseCandidate, holdout_pnp::candidate_priority};
use std::{cmp::Reverse, sync::Mutex};

pub(super) fn select_coarse_winners(
    surface: &Surface,
    coarse: &[Option<CoarseMultiplierState>],
    priority: &[usize],
    pace_realtime: bool,
    low_agl_nadir: bool,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..coarse.len()).filter(|&i| coarse[i].is_some()).collect();
    order.sort_by_key(|&idx| {
        let rank = coarse[idx].as_ref().map_or((false, 0, 0), |st| {
            (st.is_accepted(surface), st.max_inliers(), st.max_pairs)
        });
        let prio_pos = priority.iter().position(|&p| p == idx).unwrap_or(99);
        Reverse((rank, Reverse(prio_pos)))
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
    let min_second_inl = if pace_realtime { 14 } else { 10 };
    let first_30 = !pace_realtime
        && winners
            .first()
            .and_then(|&i| coarse[i].as_ref())
            .is_some_and(|s| s.is_accepted(surface) && s.max_inliers() >= 30);
    let barren_first = !pace_realtime
        && !low_agl_nadir
        && winners
            .first()
            .and_then(|&i| coarse[i].as_ref())
            .is_some_and(|s| {
                s.polished_in_coarse && !s.is_accepted(surface) && s.max_inliers() < 16
            });
    if winners.len() == 2
        && (barren_first
            || !coarse[winners[1]].as_ref().is_some_and(|s| {
                s.max_inliers() >= min_second_inl
                    && (!first_30 || (s.is_accepted(surface) && s.max_inliers() >= 30))
            }))
    {
        winners.truncate(1);
    }
    winners
}

#[expect(
    clippy::too_many_arguments,
    reason = "winning-multiplier finalization runs GPU polish and optional remaining-seed fallback"
)]
pub(super) fn finalize_coarse_multiplier(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, matcher): (&LoadedXFeat, Option<&dyn SparseMatcher>),
    opts: &AnchorSearchOpts,
    (qry_feat, prepared): (&SparseFeatures, &PreparedKashinQuery),
    render_size: [u32; 2],
    max_lm: usize,
    aux: &AnchorQueryAux<'_>,
    mut st: CoarseMultiplierState,
    mut attempts: Option<&mut Vec<AttemptRecord>>,
) -> Result<AnchorAcquisition, FlightVisualError> {
    let mut lg_layers = 0_usize;
    let mut polish_rays = 0_usize;
    if opts.polish_anchors && !st.polished_in_coarse {
        let pol = polish_anchor_seed_pool(
            surface,
            gpu_renderer,
            xfeat,
            matcher,
            qry_feat,
            (
                &st.k_render_m,
                render_size,
                st.mult,
                opts.pace_realtime,
                aux.mask_small,
            ),
            std::mem::take(&mut st.seed_pool),
            (
                st.screened.as_deref(),
                st.closest_nadir,
                &st.oblique_fallbacks,
                &st.coarse_seeds,
            ),
            &mut st.best,
            attempts.as_deref_mut(),
        )?;
        polish_rays += pol.polish_rays;
        lg_layers += pol.lg_layers;
        st.was_polished |= pol.was_polished;
        st.confidence = pol.confidence;
    }

    let accepted_after_polish = st.best.as_ref().is_some_and(|(_, _, c)| {
        c.score.geometry_accepted && is_plausible_anchor_candidate(c, surface)
    });
    if !accepted_after_polish {
        if opts.pace_realtime {
            if st.screened.is_some() {
                st.remaining_seeds.truncate(6);
            } else {
                st.remaining_seeds.truncate(8);
            }
        } else if st.polished_in_coarse
            && (st.max_inliers() < 8 || (st.screened.is_none() && st.max_inliers() < 16))
        {
            st.remaining_seeds.clear();
        } else if st.polished_in_coarse && st.max_inliers() < 12 {
            st.remaining_seeds.truncate(8);
        }
        let intrinsics = st.intrinsics();
        let mut extra_pool: Vec<(usize, PoseCandidate)> = Vec::new();
        if !st.remaining_seeds.is_empty() {
            let mut add_lg = 0_usize;
            eval_seed_chunks(
                (surface, store, projector),
                gpu_renderer,
                (xfeat, matcher),
                (qry_feat, prepared),
                &st.remaining_seeds,
                (intrinsics, &st.k_render_m, render_size, max_lm),
                SeedChunkOpts::new(0, false, true, opts.pace_realtime, st.mult, aux.mask_small),
                (&mut st.base_idx, &mut add_lg),
                (&mut extra_pool, &mut st.best, &mut st.best_view),
                attempts.as_deref_mut(),
            )?;
            lg_layers += add_lg;
        }
        eval_global_fallback_if_needed(
            (surface, store, projector),
            gpu_renderer,
            (xfeat, matcher),
            opts,
            (qry_feat, prepared),
            (intrinsics, &st.k_render_m, render_size, max_lm),
            aux,
            st.attitude,
            (&mut st.base_idx, &mut lg_layers),
            (&mut extra_pool, &mut st.best, &mut st.best_view),
        )?;
        if opts.polish_anchors && !extra_pool.is_empty() {
            let pol = polish_anchor_seed_pool(
                surface,
                gpu_renderer,
                xfeat,
                matcher,
                qry_feat,
                (
                    &st.k_render_m,
                    render_size,
                    st.mult,
                    opts.pace_realtime,
                    aux.mask_small,
                ),
                extra_pool,
                (None, None, &[], &[]),
                &mut st.best,
                attempts,
            )?;
            polish_rays += pol.polish_rays;
            lg_layers += pol.lg_layers;
            st.was_polished |= pol.was_polished;
            if pol.confidence.is_some() {
                st.confidence = pol.confidence;
            }
        }
    }

    if let Some((_, _, ref mut cand)) = st.best {
        if !is_plausible_anchor_candidate(cand, surface) {
            cand.score.geometry_accepted = false;
            st.confidence = None;
        } else if cand.score.geometry_accepted && st.confidence.is_none() {
            st.confidence = Some(classify_anchor_confidence(cand, &[]));
        }
    }
    Ok(AnchorAcquisition {
        best: st.best,
        lg_layers,
        polish_rays,
        best_view: st.best_view,
        locked_focal_px: None,
        confidence: st.confidence,
        was_polished: st.was_polished,
    })
}

pub(super) fn select_sweep_outcome(
    surface: &Surface,
    estimator: &mut FocalSweepEstimator,
    baseline_full_px: f64,
    scores: &[f64; 4],
    coarse: Vec<Option<CoarseMultiplierState>>,
    mut outcomes: Vec<Option<AnchorAcquisition>>,
    (total_lg, total_rays): (usize, usize),
) -> AnchorAcquisition {
    let multipliers = estimator.sweep_multipliers;
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
                confidence: st.confidence,
                was_polished: st.was_polished,
            });
        }
    }

    if scores.iter().any(|s| s.is_finite()) {
        let locked_px = estimator.select_multiplier(scores, baseline_full_px);
        let best_idx = multipliers
            .iter()
            .position(|&m| (baseline_full_px * m - locked_px).abs() < 1e-6)
            .unwrap_or(0);
        if let Some(mut chosen) = outcomes.get_mut(best_idx).and_then(Option::take) {
            chosen.lg_layers = total_lg;
            chosen.polish_rays = total_rays;
            chosen.locked_focal_px = Some(locked_px);
            return chosen;
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
    AnchorAcquisition {
        best: best_fb.as_ref().and_then(|a| a.best.clone()),
        lg_layers: total_lg,
        polish_rays: total_rays,
        best_view: best_fb.as_ref().and_then(|a| a.best_view.clone()),
        locked_focal_px: None,
        confidence: best_fb.as_ref().and_then(|a| a.confidence),
        was_polished: best_fb.is_some_and(|a| a.was_polished),
    }
}
