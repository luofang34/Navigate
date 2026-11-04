//! Virtual-view map-anchor acquisition, multi-hypothesis basin gating, and focal sweeps.

pub mod basin;
mod chunks;
mod eval;
pub(crate) mod nadir_screen;
pub mod polish;
pub mod seeds;
pub mod sparse_polish;
mod sweep;
mod sweep_final;
mod virt_mask;

pub use basin::{
    basin_priority_key, basin_quality_score, classify_anchor_confidence,
    is_unambiguously_confirmed, should_override_provisional_lock,
};
pub use seeds::{
    AnchorPrior, AnchorSearchOpts, DEFAULT_FOCAL_MM, FocalSweepEstimator,
    LOW_AGL_FOCAL_MULTIPLIERS, OBLIQUE_FOCAL_MULTIPLIERS, SeedPose, build_anchor_seeds,
    build_stratified_coarse_anchor_seeds, generate_anchor_seeds, is_heading_known,
};
pub use sparse_polish::polish_pose_sparse;

use crate::{
    diagnostics::AttemptRecord,
    error::FlightVisualError,
    runtime::LoadedXFeat,
    surface::{Surface, SurfaceGpuRenderer},
    types::AnchorConfidence,
};
use chunks::eval_seed_chunks;
use eval::{SeedChunkOpts, eval_global_fallback_if_needed, lead_seed_count};
use image::{GrayImage, RgbImage};
use nadir_screen::{
    NadirScreenRequest, closest_prior_nadir_seed, maybe_screen_low_agl_nadir_seeds,
};
use nalgebra::Matrix3;
use navigate_locate::corridor::{
    ColdDeadReckonPrior,
    multitier::SearchTierMatcher,
    plausibility::{PhysicalPlausibilityConfig, is_plausible_candidate, is_sane_seed_candidate},
};
use navigate_match::{
    CorridorTileStore, HadamardProjector, KashinScratch, PreparedKashinQuery, SparseFeatures,
    SparseMatcher, VirtualCorridorView,
};
use navigate_visual::PoseCandidate;
use polish::polish_anchor_seed_pool;
use std::sync::Mutex;

pub(crate) const ANCHOR_BATCH_SIZE: usize = 4;

/// Sans-IO cold or confirmatory map-anchor query specification.
#[derive(Clone, Copy, Debug)]
pub struct AnchorQuery {
    /// Geodetic position, altitude, and compass heading prior.
    pub prior: AnchorPrior,
    /// True when the camera focal length is calibrated.
    pub calibrated_focal: bool,
    /// True when an earlier cold sweep already locked the focal length.
    pub focal_locked: bool,
    /// Optional pitch/roll attitude `(down_deg, roll_deg)` from IMU or external horizon detector.
    pub attitude_override: Option<(f64, f64)>,
}

/// Optional borrowed in-memory context for [`acquire_corridor_anchor`].
pub struct AnchorQueryAux<'a> {
    /// Predicted pose `(eye_enu, world_to_cam)` from foreground tracking or VO.
    pub pred_pose: Option<SeedPose>,
    /// Live horizon attitude `(down_deg, roll_deg)` from [`crate::AmortizedMasker`].
    pub live_attitude: Option<(f64, f64)>,
    /// Dead-reckoned search center propagated across cold frames (`prev_eye.is_none()`).
    pub cold_prior: Option<&'a ColdDeadReckonPrior>,
    /// Extra confirmatory center `ENU` when re-verifying a [`AnchorConfidence::Provisional`] lock.
    pub confirmatory_center_enu: Option<[f64; 3]>,
    /// Optional downsampled in-memory `RGB` buffer for `SearchTierMatcher` global retrieval.
    pub small_rgb: Option<&'a RgbImage>,
    /// Optional binary valid-terrain mask (`255` = keep).
    pub mask_small: Option<&'a GrayImage>,
    /// Optional frame median luminance (`0..=255`) used to reject dark sensor noise.
    pub luma_p50: Option<u8>,
    /// Optional `DINOv2` multi-tier place matcher.
    pub search_tier_matcher: Option<&'a SearchTierMatcher>,
}

/// Result of a corridor map-anchor search.
pub struct AnchorAcquisition {
    /// Winning `(seed_idx, step_idx, PoseCandidate)` if any candidate was fitted.
    pub best: Option<(usize, usize, PoseCandidate)>,
    /// Total `LighterGlue` cross-attention layers executed.
    pub lg_layers: usize,
    /// Total CPU raycast operations executed during sparse polish.
    pub polish_rays: usize,
    /// Best projected `VirtualCorridorView` for landmark track seeding.
    pub best_view: Option<VirtualCorridorView>,
    /// Locked full-frame pixel focal length when estimated by an uncalibrated sweep.
    pub locked_focal_px: Option<f64>,
    /// Multi-hypothesis basin verification confidence when accepted.
    pub confidence: Option<AnchorConfidence>,
    /// True when the winning candidate was verified on rendered orthophoto/terrain polish.
    pub was_polished: bool,
}

/// Scale `(fx, fy)` of `k` by `mult` while preserving `(cx, cy)`.
#[must_use]
pub fn scale_focal_matrix(k: &Matrix3<f64>, mult: f64) -> Matrix3<f64> {
    Matrix3::new(
        k[(0, 0)] * mult,
        0.0,
        k[(0, 2)],
        0.0,
        k[(1, 1)] * mult,
        k[(1, 2)],
        0.0,
        0.0,
        1.0,
    )
}

pub(crate) fn scale_horizon_attitude(att: Option<(f64, f64)>, mult: f64) -> Option<(f64, f64)> {
    att.map(|(down_deg, roll_deg)| {
        let scaled_down = (down_deg.to_radians().tan() / mult.max(0.1))
            .atan()
            .to_degrees()
            .clamp(-45.0, 89.0);
        (scaled_down, roll_deg)
    })
}

/// Acquire a corridor map anchor without disk I/O using multi-hypothesis basin verification.
///
/// # Errors
///
/// Returns [`FlightVisualError`] when tile projection, feature extraction, or matching fails.
#[expect(
    clippy::too_many_arguments,
    reason = "anchor acquisition borrows terrain, tile store, scratch buffers, GPU renderer, models, query features, and intrinsics"
)]
pub fn acquire_corridor_anchor(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    scratch: &mut KashinScratch,
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, matcher): (&LoadedXFeat, Option<&dyn SparseMatcher>),
    (query, opts): (&AnchorQuery, &AnchorSearchOpts),
    (qry_feat, prepared): (&SparseFeatures, &PreparedKashinQuery),
    (k_full, k_render, render_size): (&Matrix3<f64>, &Matrix3<f64>, [u32; 2]),
    aux: AnchorQueryAux<'_>,
    attempts: Option<&mut Vec<AttemptRecord>>,
) -> Result<AnchorAcquisition, FlightVisualError> {
    let extreme_dark = aux.luma_p50.is_some_and(|l| l <= 5);
    let run_focal_sweep = !query.calibrated_focal && aux.pred_pose.is_none() && !query.focal_locked;
    let mut acq = if !run_focal_sweep {
        acquire_corridor_anchor_single(
            (surface, store, projector),
            scratch,
            gpu_renderer,
            (xfeat, matcher),
            (query, opts),
            (qry_feat, prepared),
            (k_full, k_render, render_size),
            &aux,
            1.0,
            attempts,
        )?
    } else {
        sweep::run_uncalibrated_focal_sweep(
            (surface, store, projector),
            scratch,
            gpu_renderer,
            (xfeat, matcher),
            (query, opts),
            (qry_feat, prepared),
            (k_full, k_render, render_size),
            &aux,
            attempts,
        )?
    };
    if extreme_dark
        && let Some((_, _, ref mut cand)) = acq.best
        && (cand.score.inliers < 48
            || cand.score.image_support_fraction < 0.12
            || cand.score.holdout_count < 12)
    {
        cand.score.geometry_accepted = false;
        acq.confidence = None;
        acq.locked_focal_px = None;
    }
    Ok(acq)
}

#[expect(
    clippy::too_many_arguments,
    reason = "single-focal anchor pass shares terrain, tile store, scratch buffers, GPU renderer, models, and query state"
)]
fn acquire_corridor_anchor_single(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    scratch: &mut KashinScratch,
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    (xfeat, matcher): (&LoadedXFeat, Option<&dyn SparseMatcher>),
    (query, opts): (&AnchorQuery, &AnchorSearchOpts),
    (qry_feat, prepared): (&SparseFeatures, &PreparedKashinQuery),
    (_k_full, k_render, render_size): (&Matrix3<f64>, &Matrix3<f64>, [u32; 2]),
    aux: &AnchorQueryAux<'_>,
    focal_mult: f64,
    mut attempts: Option<&mut Vec<AttemptRecord>>,
) -> Result<AnchorAcquisition, FlightVisualError> {
    let narrow_subwindow = aux
        .mask_small
        .is_some_and(|m| virt_mask::mask_valid_fraction(m) < 0.35);
    let sanitized_live = aux.live_attitude.map(|(d, r)| {
        if narrow_subwindow {
            (d.clamp(13.0, 17.0), r.clamp(-12.0, 12.0))
        } else {
            (d, r)
        }
    });
    let attitude = query
        .attitude_override
        .or_else(|| scale_horizon_attitude(sanitized_live, focal_mult));
    let intrinsics = [
        k_render[(0, 0)],
        k_render[(1, 1)],
        k_render[(0, 2)],
        k_render[(1, 2)],
    ];
    let want_nadir_screen =
        (aux.pred_pose.is_none() || aux.confirmatory_center_enu.is_some()) && attitude.is_none();
    let screened = if want_nadir_screen {
        maybe_screen_low_agl_nadir_seeds(
            (surface, store),
            scratch,
            NadirScreenRequest {
                prior: query.prior,
                cold_prior: aux.cold_prior,
                confirmatory_center_enu: aux.confirmatory_center_enu,
                spatial_ring_enabled: true,
                pace_realtime: opts.pace_realtime,
                coarse_sweep: false,
                post_screen_cap: opts.attempts.max(36),
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
    let mut seeds = if let Some(ref scr) = screened {
        let mut v = Vec::with_capacity(scr.len() + 1);
        if let Some(pp) = aux.pred_pose {
            v.push(pp);
        }
        v.extend_from_slice(scr);
        v
    } else {
        build_anchor_seeds(surface, query.prior, opts, attitude, aux.pred_pose)
    };
    if seeds.is_empty()
        && let Some(pp) = aux.pred_pose
    {
        seeds.push(pp);
    }
    if opts.pace_realtime && aux.pred_pose.is_some() && aux.confirmatory_center_enu.is_none() {
        seeds.truncate(1);
    } else if opts.pace_realtime && (screened.is_some() || attitude.is_some()) {
        seeds.truncate(12);
    }
    let oblique_fallbacks: Vec<SeedPose> = if screened.is_none() {
        if let Some(pp) = aux
            .pred_pose
            .filter(|_| aux.confirmatory_center_enu.is_none())
        {
            vec![pp]
        } else {
            let (_, _, _, agl) = seeds::resolve_prior_origin(surface, query.prior);
            if !is_heading_known(query.prior, opts) && agl > 250.0 {
                let down_deg = attitude.map_or_else(
                    || virt_mask::subwindow_oblique_down_deg(aux.mask_small, 22.0),
                    |(d, _)| d.clamp(8.0, 24.0),
                );
                virt_mask::rank_oblique_fallbacks(
                    store,
                    scratch,
                    prepared,
                    intrinsics,
                    render_size,
                    aux.mask_small,
                    seeds::cardinal_oblique_fallbacks(surface, query.prior, down_deg),
                )?
            } else {
                seeds.iter().copied().take(3).collect()
            }
        }
    } else {
        Vec::new()
    };

    let max_lm = opts.query_keypoints.clamp(256, 2048);
    let mut best: Option<(usize, usize, PoseCandidate)> = None;
    let mut best_view: Option<VirtualCorridorView> = None;
    let mut seed_pool: Vec<(usize, PoseCandidate)> = Vec::new();
    let mut lg_layers = 0_usize;
    let mut base_idx = 0_usize;

    eval_seed_chunks(
        (surface, store, projector),
        gpu_renderer,
        (xfeat, matcher),
        (qry_feat, prepared),
        &seeds,
        (intrinsics, k_render, render_size, max_lm),
        SeedChunkOpts::new(
            lead_seed_count(query, opts, aux.pred_pose.is_some()),
            false,
            true,
            opts.pace_realtime,
            focal_mult,
            aux.mask_small,
        ),
        (&mut base_idx, &mut lg_layers),
        (&mut seed_pool, &mut best, &mut best_view),
        attempts.as_deref_mut(),
    )?;

    eval_global_fallback_if_needed(
        (surface, store, projector),
        gpu_renderer,
        (xfeat, matcher),
        opts,
        (qry_feat, prepared),
        (intrinsics, k_render, render_size, max_lm),
        aux,
        attitude,
        (&mut base_idx, &mut lg_layers),
        (&mut seed_pool, &mut best, &mut best_view),
    )?;

    let mut polish_rays = 0_usize;
    let mut was_polished = false;
    let mut confidence = None;
    if opts.polish_anchors {
        let pol = polish_anchor_seed_pool(
            surface,
            gpu_renderer,
            xfeat,
            matcher,
            qry_feat,
            (
                k_render,
                render_size,
                focal_mult,
                opts.pace_realtime,
                aux.mask_small,
            ),
            seed_pool,
            (
                screened.as_deref(),
                closest_nadir,
                &oblique_fallbacks,
                &seeds,
            ),
            &mut best,
            attempts,
        )?;
        polish_rays += pol.polish_rays;
        lg_layers += pol.lg_layers;
        was_polished = pol.was_polished;
        confidence = pol.confidence;
    }

    if let Some((_, _, ref mut cand)) = best {
        if !is_plausible_anchor_candidate(cand, surface) {
            cand.score.geometry_accepted = false;
            confidence = None;
        } else if cand.score.geometry_accepted && confidence.is_none() {
            confidence = Some(classify_anchor_confidence(cand, &[]));
        }
    }
    Ok(AnchorAcquisition {
        best,
        lg_layers,
        polish_rays,
        best_view,
        locked_focal_px: None,
        confidence,
        was_polished,
    })
}

pub(crate) type SeedStateMut<'a> = (
    &'a mut Vec<(usize, PoseCandidate)>,
    &'a mut Option<(usize, usize, PoseCandidate)>,
    &'a mut Option<VirtualCorridorView>,
);

/// Return true when `cand` is finite and above terrain clearance (`5.0 m`).
#[must_use]
pub fn is_sane_anchor_seed(cand: &PoseCandidate, surface: &Surface) -> bool {
    is_sane_seed_candidate(cand, surface, 5.0)
}

/// Return true when `cand` passes physical terrain, pitch, and conditioning plausibility checks.
#[must_use]
pub fn is_plausible_anchor_candidate(cand: &PoseCandidate, surface: &Surface) -> bool {
    let config = PhysicalPlausibilityConfig::default();
    is_plausible_candidate(cand, surface, None, false, &config)
}

#[expect(
    clippy::too_many_arguments,
    reason = "typed attempt helper captures solver label, seed/step indices, match count, candidate score, and virtual-view metadata"
)]
pub(crate) fn record_attempt(
    log: &mut Vec<AttemptRecord>,
    backend: &'static str,
    seed: usize,
    step: usize,
    sketch_score: f32,
    pairs: usize,
    cand: Option<&PoseCandidate>,
    virt_meta: Option<(usize, usize, f64)>,
) {
    let (virtual_landmarks, visible_tiles, focal_multiplier) = match virt_meta {
        Some((lm, vt, fm)) => (Some(lm), Some(vt), Some(fm)),
        None => (None, None, None),
    };
    log.push(AttemptRecord {
        backend,
        seed,
        step,
        sketch_score,
        pairs: cand.map_or(pairs, |c| c.score.pairs),
        score: cand.map(|c| c.score.clone()),
        virtual_landmarks,
        visible_tiles,
        focal_multiplier,
    });
}
