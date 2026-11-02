//! Fallback camera focal length estimation and zero-bank focal sweeps.

use super::{
    MatchStageOptions, estimate_intrinsics,
    gpu_match::{StageTerrainRenderer, TerrainMatchQuery},
    has_calibrated_focal_prior, load_horizon_attitude, scale_k, scaled_size,
    seeds::{build_stratified_coarse_seeds, generate_seeds},
};
use crate::{
    PipelineError,
    matcher_client::PipelineMatcher,
    pose::{PoseCandidate, candidate_priority, fit_pose_ransac},
    surface::ContextSurface,
    testset::PhotoCase,
};
use image::{DynamicImage, GrayImage, RgbImage, imageops::FilterType};
use nalgebra::Matrix3;
use serde_json::{Value, json};
use std::cmp::Reverse;

/// Default 35mm-equivalent focal length in millimetres (`26.0` mm).
pub const DEFAULT_FOCAL_MM: f64 = 26.0;

/// Focal multiplier sweep values for unverified cold anchors.
pub const FOCAL_SWEEP_MULTIPLIERS: [f64; 4] = [1.0, 0.55, 1.45, 2.0];

/// Estimator for camera focal length with sweep multipliers and lock state.
#[derive(Debug, Clone, PartialEq)]
pub struct FocalLengthEstimator {
    /// Nominal 35mm-equivalent focal length in millimetres.
    pub default_focal_mm: f64,
    /// Multipliers evaluated during cold anchor acquisition.
    pub sweep_multipliers: [f64; 4],
    /// Locked refined focal length in pixels.
    pub locked_focal_px: Option<f64>,
}

impl Default for FocalLengthEstimator {
    fn default() -> Self {
        Self {
            default_focal_mm: DEFAULT_FOCAL_MM,
            sweep_multipliers: FOCAL_SWEEP_MULTIPLIERS,
            locked_focal_px: None,
        }
    }
}

impl FocalLengthEstimator {
    /// Computes baseline pixel focal length using standard 36mm sensor width.
    #[must_use]
    pub fn baseline_focal_px(&self, image_width_px: u32) -> f64 {
        (self.default_focal_mm / 36.0) * f64::from(image_width_px)
    }

    /// Selects the best multiplier based on candidate reprojection RMSE scores.
    pub fn select_multiplier(&mut self, scores: &[f64; 4], baseline_px: f64) -> f64 {
        let mut best_idx = 0;
        let mut min_score = f64::INFINITY;
        for (i, &score) in scores.iter().enumerate() {
            if score < min_score {
                min_score = score;
                best_idx = i;
            }
        }
        let chosen_focal = baseline_px * self.sweep_multipliers[best_idx];
        self.locked_focal_px = Some(chosen_focal);
        chosen_focal
    }
}

fn scale_focal_only(k: &Matrix3<f64>, mult: f64) -> Matrix3<f64> {
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

#[expect(
    clippy::too_many_arguments,
    reason = "seed evaluation shares renderer, terrain surface, query image buffers, matcher, and telemetry log"
)]
fn eval_seed_steps(
    terrain_renderer: &mut StageTerrainRenderer,
    surface: &ContextSurface,
    matcher: &mut PipelineMatcher,
    (qry_small, qry_small_gray, valid_mask): (&RgbImage, &GrayImage, &GrayImage),
    (full_size, render_size, q_scale): ([u32; 2], [u32; 2], [f64; 2]),
    k_render: &Matrix3<f64>,
    focal_mult: f64,
    (seed_idx, mut eye, mut rot): (usize, [f64; 3], Matrix3<f64>),
    (start_step, max_steps): (usize, usize),
    best: &mut Option<(usize, usize, PoseCandidate)>,
    attempts_log: &mut Vec<Value>,
) -> Result<(usize, usize), PipelineError> {
    let mut best_inl = 0_usize;
    let mut best_pairs = 0_usize;
    for step_idx in start_step..max_steps {
        let (raw_count, world_pts, img_px) = terrain_renderer.match_view(
            surface,
            &TerrainMatchQuery {
                eye,
                rot: &rot,
                k_render,
                render_size,
                qry_small_rgb: qry_small,
                qry_small_gray,
                valid_mask,
                full_size,
                q_scale,
            },
            matcher,
        )?;
        best_pairs = best_pairs.max(world_pts.len());
        let Some(cand) =
            fit_pose_ransac(&world_pts, &img_px, k_render, render_size, Some((eye, rot)))
        else {
            attempts_log.push(json!({
                "backend": matcher.identity(),
                "seed": seed_idx,
                "step": step_idx,
                "focal_multiplier": focal_mult,
                "raw_pairs": raw_count,
                "pairs": world_pts.len(),
                "inliers": 0,
                "geometry_accepted": false,
                "prior_gate": false,
            }));
            break;
        };
        best_inl = best_inl.max(cand.score.inliers);
        best_pairs = best_pairs.max(cand.score.pairs);
        attempts_log.push(json!({
            "backend": matcher.identity(),
            "seed": seed_idx,
            "step": step_idx,
            "focal_multiplier": focal_mult,
            "raw_pairs": raw_count,
            "pairs": cand.score.pairs,
            "inliers": cand.score.inliers,
            "train_inliers": cand.score.train_inliers,
            "train_rms_px": cand.score.train_rms_px,
            "holdout_count": cand.score.holdout_count,
            "holdout_inlier_fraction": cand.score.holdout_inlier_fraction,
            "holdout_median_px": cand.score.holdout_median_px,
            "image_support_fraction": cand.score.image_support_fraction,
            "positive_depth_fraction": cand.score.positive_depth_fraction,
            "focal_px": cand.score.focal_px,
            "geometry_accepted": cand.score.geometry_accepted,
            "prior_gate": true,
        }));

        eye = cand.eye;
        rot = cand.rotation;
        let replace = best.as_ref().is_none_or(|(_, _, prev)| {
            candidate_priority(&cand.score) > candidate_priority(&prev.score)
        });
        if replace {
            *best = Some((seed_idx, step_idx, cand.clone()));
        }
        if cand.score.train_inliers < 12 {
            break;
        }
    }
    Ok((best_inl, best_pairs))
}

/// Indexed candidate tuple `(seed_index, step_index, candidate)` for stage pose search.
pub(super) type StageBestCandidate = (usize, usize, PoseCandidate);

struct StageFocalSlot {
    mult: f64,
    k_render_m: Matrix3<f64>,
    remaining_seeds: Vec<([f64; 3], Matrix3<f64>)>,
    base_idx: usize,
    best: Option<StageBestCandidate>,
    max_inliers: usize,
    max_pairs: usize,
}

/// Evaluate one photo case in `run_match_stage`, using coarse-to-fine focal sweep when uncalibrated.
pub(super) fn match_case_with_focal_policy(
    opts: &MatchStageOptions,
    surface: &ContextSurface,
    terrain_renderer: &mut StageTerrainRenderer,
    matcher: &mut PipelineMatcher,
    case: &PhotoCase,
    query_rgb: &RgbImage,
    valid_mask: &GrayImage,
) -> Result<(Vec<Value>, Option<StageBestCandidate>), PipelineError> {
    let k = estimate_intrinsics(case);
    let render_size = scaled_size(case.size, 800);
    let scale_x = f64::from(render_size[0]) / f64::from(case.size[0].max(1));
    let scale_y = f64::from(render_size[1]) / f64::from(case.size[1].max(1));
    let k_render = scale_k(&k, scale_x, scale_y);
    let qry_small = image::imageops::resize(
        query_rgb,
        render_size[0],
        render_size[1],
        FilterType::Triangle,
    );
    let qry_small_gray = DynamicImage::ImageRgb8(qry_small.clone()).into_luma8();
    let max_steps = opts.steps.max(1);
    let mut attempts_log = Vec::new();

    if has_calibrated_focal_prior(case) {
        let attitude = load_horizon_attitude(&opts.dataset_dir, case, &k);
        let seeds = generate_seeds(&surface.primary, case, opts, attitude);
        let mut best: Option<(usize, usize, PoseCandidate)> = None;
        for (seed_idx, (eye, rot)) in seeds.into_iter().enumerate() {
            let _ = eval_seed_steps(
                terrain_renderer,
                surface,
                matcher,
                (&qry_small, &qry_small_gray, valid_mask),
                (case.size, render_size, [scale_x, scale_y]),
                &k_render,
                1.0,
                (seed_idx, eye, rot),
                (0, max_steps),
                &mut best,
                &mut attempts_log,
            )?;
        }
        return Ok((attempts_log, best));
    }

    let mut estimator = FocalLengthEstimator::default();
    let baseline_full_px = k[(0, 0)];
    let multipliers = estimator.sweep_multipliers;
    let coarse_cap = opts.attempts.clamp(4, 12);
    const PRIORITY: [usize; 4] = [0, 2, 1, 3];
    let mut slots: Vec<Option<StageFocalSlot>> = (0..multipliers.len()).map(|_| None).collect();

    for &idx in &PRIORITY {
        let mult = multipliers[idx];
        let k_m = scale_focal_only(&k, mult);
        let k_render_m = scale_focal_only(&k_render, mult);
        let attitude = load_horizon_attitude(&opts.dataset_dir, case, &k_m);
        let (coarse_seeds, remaining_seeds) =
            build_stratified_coarse_seeds(&surface.primary, case, opts, attitude, None, coarse_cap);
        let mut best: Option<(usize, usize, PoseCandidate)> = None;
        let mut max_inliers = 0_usize;
        let mut max_pairs = 0_usize;
        for (seed_idx, (eye, rot)) in coarse_seeds.iter().copied().enumerate() {
            let (inl, prs) = eval_seed_steps(
                terrain_renderer,
                surface,
                matcher,
                (&qry_small, &qry_small_gray, valid_mask),
                (case.size, render_size, [scale_x, scale_y]),
                &k_render_m,
                mult,
                (seed_idx, eye, rot),
                (0, 1),
                &mut best,
                &mut attempts_log,
            )?;
            max_inliers = max_inliers.max(inl);
            max_pairs = max_pairs.max(prs);
            if best.as_ref().is_some_and(|(_, _, c)| {
                c.score.geometry_accepted
                    && c.score.inliers >= 48
                    && c.score.holdout_inlier_fraction >= 0.45
            }) {
                break;
            }
        }
        let early_zero = idx == 0
            && best.as_ref().is_some_and(|(_, _, c)| {
                c.score.geometry_accepted
                    && c.score.inliers >= 48
                    && c.score.holdout_inlier_fraction >= 0.40
            });
        slots[idx] = Some(StageFocalSlot {
            mult,
            k_render_m,
            remaining_seeds,
            base_idx: coarse_seeds.len(),
            best,
            max_inliers,
            max_pairs,
        });
        if early_zero {
            break;
        }
    }

    let mut order: Vec<usize> = slots
        .iter()
        .enumerate()
        .filter_map(|(i, s)| s.is_some().then_some(i))
        .collect();
    order.sort_by_key(|&i| {
        let Some(s) = slots[i].as_ref() else {
            return Reverse((false, 0, 0, false, Reverse(i)));
        };
        let acc = s
            .best
            .as_ref()
            .is_some_and(|(_, _, c)| c.score.geometry_accepted);
        Reverse((acc, s.max_inliers, s.max_pairs, i == 0, Reverse(i)))
    });
    let winners: Vec<usize> = order.into_iter().take(2).collect();
    let mut scores = [f64::INFINITY; 4];
    let mut finals: Vec<Option<(usize, usize, PoseCandidate)>> =
        (0..multipliers.len()).map(|_| None).collect();

    for (win_pos, &w_idx) in winners.iter().enumerate() {
        let Some(mut slot) = slots[w_idx].take() else {
            continue;
        };
        if max_steps > 1
            && let Some((b_seed, _, ref b_cand)) = slot.best.clone()
            && b_cand.score.train_inliers >= 12
        {
            let _ = eval_seed_steps(
                terrain_renderer,
                surface,
                matcher,
                (&qry_small, &qry_small_gray, valid_mask),
                (case.size, render_size, [scale_x, scale_y]),
                &slot.k_render_m,
                slot.mult,
                (b_seed, b_cand.eye, b_cand.rotation),
                (1, max_steps),
                &mut slot.best,
                &mut attempts_log,
            )?;
        }
        let accepted = slot
            .best
            .as_ref()
            .is_some_and(|(_, _, c)| c.score.geometry_accepted);
        if !accepted {
            for (rem_i, (eye, rot)) in slot.remaining_seeds.into_iter().enumerate() {
                let seed_idx = slot.base_idx + rem_i;
                let _ = eval_seed_steps(
                    terrain_renderer,
                    surface,
                    matcher,
                    (&qry_small, &qry_small_gray, valid_mask),
                    (case.size, render_size, [scale_x, scale_y]),
                    &slot.k_render_m,
                    slot.mult,
                    (seed_idx, eye, rot),
                    (0, max_steps),
                    &mut slot.best,
                    &mut attempts_log,
                )?;
                if slot
                    .best
                    .as_ref()
                    .is_some_and(|(_, _, c)| c.score.geometry_accepted && c.score.inliers >= 42)
                {
                    break;
                }
            }
        }
        let mut early_done = false;
        if let Some((_, _, ref cand)) = slot.best
            && cand.score.geometry_accepted
        {
            let ang_err = cand.score.holdout_median_px / slot.mult.max(0.25);
            scores[w_idx] = ang_err / (cand.score.inliers.max(1) as f64).sqrt();
            early_done = win_pos == 0
                && cand.score.inliers >= 48
                && cand.score.holdout_inlier_fraction >= 0.45;
        }
        finals[w_idx] = slot.best;
        if early_done {
            break;
        }
    }

    for (idx, rem) in slots.into_iter().enumerate() {
        if finals[idx].is_none()
            && let Some(slot) = rem
        {
            finals[idx] = slot.best;
        }
    }

    if scores.iter().any(|s| s.is_finite()) {
        let locked_px = estimator.select_multiplier(&scores, baseline_full_px);
        let best_idx = multipliers
            .iter()
            .position(|&m| (baseline_full_px * m - locked_px).abs() < 1e-6)
            .unwrap_or(0);
        if let Some(chosen) = finals.get_mut(best_idx).and_then(Option::take) {
            return Ok((attempts_log, Some(chosen)));
        }
    }

    let mut fallback: Option<(usize, usize, PoseCandidate)> = None;
    for cand_opt in finals.into_iter().flatten() {
        let take = fallback.as_ref().is_none_or(|(_, _, prev)| {
            candidate_priority(&cand_opt.2.score) > candidate_priority(&prev.score)
        });
        if take {
            fallback = Some(cand_opt);
        }
    }
    Ok((attempts_log, fallback))
}
