//! Multi-hypothesis basin scoring and [`AnchorConfidence`] classification.

use crate::types::AnchorConfidence;
use nalgebra::Matrix3;
use navigate_locate::corridor::plausibility::compute_fisher_min_eigenvalue;
use navigate_visual::{PoseCandidate, PoseScore, holdout_pnp::candidate_priority};

/// Minimum horizontal separation in meters for diverse top-K seed polish selection.
pub const POLISH_DIVERSITY_RADIUS_M: f64 = 15.0;

/// Minimum horizontal separation in meters for competing spatial basin checks.
pub const COMPETING_BASIN_SEPARATION_M: f64 = 20.0;

/// Required quality ratio over any competing spatial basin to classify an anchor as [`AnchorConfidence::Confirmed`].
pub const CONFIRMED_BASIN_MARGIN_RATIO: f64 = 1.35;

/// Return true when `score` has high support and tight residuals that need no further multi-seed polish.
#[must_use]
pub fn is_unambiguously_confirmed(score: &PoseScore) -> bool {
    score.geometry_accepted
        && ((score.inliers >= 48
            && score.holdout_inlier_fraction >= 0.45
            && score.holdout_median_px <= 5.5
            && score.train_rms_px <= 5.8)
            || (score.inliers >= 88
                && score.holdout_inlier_fraction >= 0.34
                && score.holdout_median_px <= 6.2
                && score.train_rms_px <= 6.2))
}

/// Compute a continuous basin quality score combining inliers, holdout residuals, and Fisher conditioning.
///
/// This score penalizes elevated holdout median (`> 3.8 px`) and training RMS (`> 4.0 px`) residuals
/// so a tight true pose (`~2.4 px`) outranks a loose spatial-ring alias (`~9.0 px`) even when the
/// false alias has more raw correspondences.
#[must_use]
pub fn basin_quality_score(cand: &PoseCandidate) -> f64 {
    let s = &cand.score;
    if s.inliers == 0
        || !s.holdout_median_px.is_finite()
        || !s.train_rms_px.is_finite()
        || !s.holdout_inlier_fraction.is_finite()
    {
        return 0.0;
    }
    let support = (s.inliers as f64) * (0.40 + 0.60 * s.holdout_inlier_fraction.clamp(0.0, 1.0));
    let h_excess = (s.holdout_median_px - 3.8).max(0.0);
    let r_excess = (s.train_rms_px - 4.0).max(0.0);
    let residual_weight =
        1.0 / ((1.0 + 0.45 * h_excess * h_excess) * (1.0 + 0.35 * r_excess * r_excess));
    let min_eig = compute_fisher_min_eigenvalue(cand).max(0.0);
    let conditioning = if min_eig <= 0.0 {
        0.50
    } else {
        (min_eig / (min_eig + 1.0e-4)).sqrt().clamp(0.50, 1.15)
    };
    support * residual_weight * conditioning
}

/// Return a total-order comparison key ranking accepted poses by [`basin_quality_score`] first.
#[must_use]
pub fn basin_priority_key(
    accepted: bool,
    cand: &PoseCandidate,
) -> (bool, i64, (bool, usize, i64, i64)) {
    let q_milli = (basin_quality_score(cand) * 1_000.0).round() as i64;
    (accepted, q_milli, candidate_priority(&cand.score))
}

/// Return true when `(a_eye, a_rot)` and `(b_eye, b_rot)` lie in the same spatial-angular basin.
#[must_use]
pub fn within_pose_neighborhood(
    a_eye: [f64; 3],
    a_rot: &Matrix3<f64>,
    b_eye: [f64; 3],
    b_rot: &Matrix3<f64>,
    horiz_radius_m: f64,
) -> bool {
    let horiz = (a_eye[0] - b_eye[0]).hypot(a_eye[1] - b_eye[1]);
    let cos_ang = (((a_rot * b_rot.transpose()).trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
    horiz <= horiz_radius_m && cos_ang >= 0.94
}

/// Classify an accepted winning pose as [`AnchorConfidence::Confirmed`] or [`AnchorConfidence::Provisional`].
///
/// This function marks the winner [`AnchorConfidence::Provisional`] when its residuals exceed
/// confirmation bounds without dense inlier support, or when a competing spatial basin more than
/// `20 m` horizontally away reaches at least `1 / 1.35` of the winner quality.
#[must_use]
pub fn classify_anchor_confidence(
    winner: &PoseCandidate,
    evaluated_basins: &[(bool, PoseCandidate)],
) -> AnchorConfidence {
    let dense_low_rms = winner.score.inliers >= 64
        && winner.score.holdout_inlier_fraction >= 0.34
        && winner.score.holdout_median_px <= 12.6
        && winner.score.train_rms_px <= 6.8;
    if !is_unambiguously_confirmed(&winner.score)
        && !dense_low_rms
        && (winner.score.holdout_median_px > 8.3 || winner.score.train_rms_px > 6.6)
    {
        return AnchorConfidence::Provisional;
    }
    let win_q = basin_quality_score(winner);
    for (is_acc, other) in evaluated_basins {
        let viable_competing =
            *is_acc || (other.score.inliers >= 20 && other.score.holdout_median_px <= 6.0);
        if !viable_competing {
            continue;
        }
        let dist_xy = (winner.eye[0] - other.eye[0]).hypot(winner.eye[1] - other.eye[1]);
        if dist_xy > COMPETING_BASIN_SEPARATION_M {
            let other_q = basin_quality_score(other);
            if win_q < CONFIRMED_BASIN_MARGIN_RATIO * other_q {
                return AnchorConfidence::Provisional;
            }
        }
    }
    AnchorConfidence::Confirmed
}

/// Return true when a newly arrived anchor should override an active [`AnchorConfidence::Provisional`] lock.
#[must_use]
pub fn should_override_provisional_lock(
    prev_holdout_median_px: f64,
    new_cand: &PoseCandidate,
    new_confidence: AnchorConfidence,
) -> bool {
    let h_new = new_cand.score.holdout_median_px;
    if !h_new.is_finite() {
        return false;
    }
    if new_confidence == AnchorConfidence::Confirmed
        && h_new <= 4.0
        && h_new <= 0.70 * prev_holdout_median_px.max(4.5)
    {
        return true;
    }
    h_new <= 3.5 && h_new + 1.5 < prev_holdout_median_px
}

pub(super) fn seed_only_candidate(
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
            holdout_median_px: 1.0e6,
            image_support_fraction: 0.0,
            inliers: 0,
            pairs: 0,
            positive_depth_fraction: 0.0,
            train_inliers: 0,
            train_rms_px: 1.0e6,
        },
    }
}

pub(super) fn clamp_screened_step1_seed(
    cand: &mut PoseCandidate,
    orig_eye: [f64; 3],
    orig_rot: Matrix3<f64>,
    pace_realtime: bool,
) {
    let dz = if pace_realtime {
        (60.0, 40.0)
    } else {
        (45.0, 35.0)
    };
    cand.eye[0] = (cand.eye[0] - orig_eye[0]).clamp(-75.0, 75.0) + orig_eye[0];
    cand.eye[1] = (cand.eye[1] - orig_eye[1]).clamp(-75.0, 75.0) + orig_eye[1];
    cand.eye[2] = cand.eye[2].clamp(orig_eye[2] - dz.0, orig_eye[2] + dz.1);
    let cos_d = ((cand.rotation * orig_rot.transpose()).trace() - 1.0) * 0.5;
    if (-orig_rot[(2, 2)] > 0.90 && -cand.rotation[(2, 2)] < 0.85)
        || (!pace_realtime && cos_d < 0.90)
    {
        cand.rotation = orig_rot;
    }
}

pub(crate) fn is_fragile_virtual_anchor(score: &PoseScore) -> bool {
    score.pairs > 0 && (score.inliers < 40 || score.holdout_count < 10)
}

pub(crate) type SeedPriorityKey = (
    bool,
    bool,
    bool,
    bool,
    usize,
    usize,
    (bool, usize, i64, i64),
);

/// Rank intermediate seed candidates for reproject and GPU polish by holdout coherence first.
pub(crate) fn seed_refinement_priority(score: &PoseScore) -> SeedPriorityKey {
    let hold_inl = ((score.holdout_count as f64) * score.holdout_inlier_fraction).round() as usize;
    let coherent_holdout = (score.inliers >= 7
        && ((hold_inl >= 1 && score.holdout_median_px <= 50.0)
            || score.holdout_median_px <= 45.0
            || score.inliers >= 14))
        || (score.inliers == 6 && score.pairs >= 20 && score.holdout_median_px <= 40.0);
    let has_holdout = coherent_holdout && (score.inliers >= 10 || hold_inl > 0);
    let tight_train =
        (score.inliers >= 7 || (score.inliers == 6 && hold_inl >= 1)) && score.train_rms_px <= 7.8;
    (
        score.geometry_accepted,
        coherent_holdout,
        has_holdout,
        tight_train,
        score.inliers,
        hold_inl,
        candidate_priority(score),
    )
}

pub(super) fn veto_unverified_virtual_best(
    best: &mut Option<(usize, usize, PoseCandidate)>,
    seed_idx: usize,
    orig_eye: [f64; 3],
    orig_rot: &Matrix3<f64>,
) {
    if let Some((b_seed, b_step, ref mut b_cand)) = *best
        && b_step <= 1
        && b_cand.score.pairs > 0
        && (b_cand.score.inliers < 40 || b_cand.score.holdout_count < 10)
        && (b_seed == seed_idx
            || within_pose_neighborhood(b_cand.eye, &b_cand.rotation, orig_eye, orig_rot, 25.0))
    {
        b_cand.score.geometry_accepted = false;
    }
}
