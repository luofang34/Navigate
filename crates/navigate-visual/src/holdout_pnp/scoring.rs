//! Geometric candidate scoring with dual tight and low-AGL relief holdout gates.

use super::{GRID_DIM, INLIER_PX, PoseCandidate, PoseScore, reproj_error};
use nalgebra::{Matrix3, Vector3};
use std::collections::BTreeSet;

pub(crate) fn scaled_inlier_px(size: [u32; 2]) -> f64 {
    INLIER_PX * (f64::from(size[0].max(size[1])) / 640.0).max(1.0)
}

pub(crate) fn low_agl_relief_inlier_px(
    inlier_px: f64,
    world_points: &[[f64; 3]],
    eye: [f64; 3],
    k: &Matrix3<f64>,
) -> f64 {
    if world_points.is_empty() || !eye.iter().all(|v| v.is_finite()) {
        return inlier_px;
    }
    let n = world_points.len() as f64;
    let (mut sum_z, mut sum_dist) = (0.0, 0.0);
    let eye_v = Vector3::from(eye);
    for &pt in world_points {
        sum_z += pt[2];
        sum_dist += (Vector3::from(pt) - eye_v).norm();
    }
    let agl_m = eye[2] - (sum_z / n);
    let mean_dist = sum_dist / n;
    if agl_m > 0.0 && agl_m < 200.0 && mean_dist < 600.0 {
        let focal_px = 0.5 * (k[(0, 0)] + k[(1, 1)]);
        let relief_px = (0.55 * (10.0 / agl_m.max(25.0)) * focal_px).min(12.0);
        inlier_px.max(relief_px)
    } else {
        inlier_px
    }
}

pub(crate) fn score_pose_candidate(
    world_points: &[[f64; 3]],
    image_pixels: &[[f64; 2]],
    held: &[bool],
    (eye, rot): ([f64; 3], Matrix3<f64>),
    k: &Matrix3<f64>,
    size: [u32; 2],
    inlier_px: f64,
) -> PoseCandidate {
    let base_inlier_px = scaled_inlier_px(size);
    let eff_inlier_px = low_agl_relief_inlier_px(inlier_px, world_points, eye, k);
    let relief_ratio = (eff_inlier_px / base_inlier_px.max(1.0)).max(1.0);
    let hold_count = held.iter().filter(|&&h| h).count();
    let mut inliers = vec![false; world_points.len()];
    let (mut pos_depth, mut train_sq_sum, mut train_inliers, mut hold_inliers) = (0, 0.0, 0, 0);
    let (mut tight_train_inliers, mut tight_hold_inliers) = (0_usize, 0_usize);
    let mut hold_errors = Vec::with_capacity(hold_count);
    let mut cells = BTreeSet::new();
    let mut tight_cells = BTreeSet::new();
    let (w, h) = (f64::from(size[0].max(1)), f64::from(size[1].max(1)));

    for i in 0..world_points.len() {
        let p_cam = rot * (Vector3::from(world_points[i]) - Vector3::from(eye));
        pos_depth += usize::from(p_cam.z > 0.1);
        let err = reproj_error(world_points[i], image_pixels[i], eye, &rot, k).unwrap_or(1e6);
        let ok = err <= eff_inlier_px;
        let tight_ok = err <= base_inlier_px;
        inliers[i] = ok;
        if ok {
            let gx = ((image_pixels[i][0] / w) * f64::from(GRID_DIM))
                .floor()
                .clamp(0.0, 11.0) as u32;
            let gy = ((image_pixels[i][1] / h) * f64::from(GRID_DIM))
                .floor()
                .clamp(0.0, 11.0) as u32;
            cells.insert((gx, gy));
            if tight_ok {
                tight_cells.insert((gx, gy));
            }
        }
        if tight_ok {
            if held[i] {
                tight_hold_inliers += 1;
            } else {
                tight_train_inliers += 1;
            }
        }
        if held[i] {
            hold_errors.push(err);
            hold_inliers += usize::from(ok);
        } else if ok {
            train_inliers += 1;
            train_sq_sum += err * err;
        }
    }

    hold_errors.sort_by(f64::total_cmp);
    let hold_median = hold_errors
        .get(hold_errors.len() / 2)
        .copied()
        .unwrap_or(1e6);
    let train_rms = if train_inliers > 0 {
        (train_sq_sum / (train_inliers as f64)).sqrt()
    } else {
        1e6
    };
    let hold_frac = (hold_inliers as f64) / (hold_count.max(1) as f64);
    let pos_frac = (pos_depth as f64) / (world_points.len() as f64);
    let support_frac = (cells.len() as f64) / f64::from(GRID_DIM * GRID_DIM);
    let total_inliers = train_inliers + hold_inliers;

    let base_min_hold = 8.min((hold_count * 2 / 5).max(2)).min(hold_count);
    let tight_total = tight_train_inliers + tight_hold_inliers;
    let tight_hold_frac = (tight_hold_inliers as f64) / (hold_count.max(1) as f64);
    let tight_min_train = 24.min(
        tight_total
            .saturating_sub(tight_hold_inliers.max(base_min_hold))
            .max(12),
    );
    let tight_min_hold_frac = if tight_total >= 64 { 0.32 } else { 0.40 };
    let tight_accepted = tight_total >= 26
        && tight_train_inliers >= tight_min_train
        && tight_hold_inliers >= base_min_hold
        && tight_hold_frac >= tight_min_hold_frac
        && hold_median <= 1.5 * base_inlier_px
        && pos_frac >= 0.98
        && tight_cells.len() >= 6;

    let used_px = base_inlier_px
        .max(train_rms * 1.12)
        .max(hold_median.min(train_rms * 1.35))
        .min(eff_inlier_px);
    let used_ratio = (used_px / base_inlier_px.max(1.0)).max(1.0);
    let scaled_min_inliers = (26.0 * used_ratio.sqrt()).ceil() as usize;
    let strong_sparse_relief = (train_rms <= base_inlier_px * 1.12 && hold_frac >= 0.55)
        || (hold_frac >= 0.75 && hold_inliers >= 10);
    let min_total_inliers = if relief_ratio > 1.5 {
        if strong_sparse_relief {
            26
        } else if hold_frac >= 0.60 {
            32
        } else {
            scaled_min_inliers.max(33)
        }
    } else {
        scaled_min_inliers.max(26)
    };
    let min_hold_floor = if hold_count <= 6 { 3 } else { 5 };
    let min_hold = if relief_ratio > 1.5 && !strong_sparse_relief {
        12.min((hold_count * 3 / 5).max(min_hold_floor))
            .min(hold_count)
    } else {
        base_min_hold
    };
    let min_cells = if relief_ratio > 1.5 && strong_sparse_relief {
        6
    } else if used_ratio > 1.5 && total_inliers >= 40 {
        10
    } else if relief_ratio > 1.5 && hold_frac >= 0.60 {
        7
    } else if relief_ratio > 1.5 {
        8
    } else {
        6
    };
    let min_train = 24.min(
        total_inliers
            .saturating_sub(hold_inliers.max(min_hold))
            .max(12),
    );
    let min_hold_frac = if total_inliers >= 64 { 0.32 } else { 0.40 };
    let relief_accepted = total_inliers >= min_total_inliers
        && train_inliers >= min_train
        && hold_inliers >= min_hold
        && hold_frac >= min_hold_frac
        && hold_median <= 1.5 * eff_inlier_px
        && pos_frac >= 0.98
        && cells.len() >= min_cells;

    let geometry_accepted = tight_accepted || relief_accepted;

    let score = PoseScore {
        focal_px: k[(0, 0)],
        geometry_accepted,
        holdout_count: hold_count,
        holdout_inlier_fraction: hold_frac,
        holdout_median_px: hold_median,
        image_support_fraction: support_frac,
        inliers: total_inliers,
        pairs: world_points.len(),
        positive_depth_fraction: pos_frac,
        train_inliers,
        train_rms_px: train_rms,
    };

    PoseCandidate {
        eye,
        rotation: rot,
        k: *k,
        size,
        points: world_points.to_vec(),
        pixels: image_pixels.to_vec(),
        inliers,
        score,
    }
}

/// Return a sortable priority tuple for [`PoseScore`] where larger values rank higher.
#[must_use]
pub fn candidate_priority(score: &PoseScore) -> (bool, usize, i64, i64) {
    (
        score.geometry_accepted,
        score.inliers,
        (score.holdout_inlier_fraction * 1_000_000.0).round() as i64,
        -(score.holdout_median_px * 1_000.0).round() as i64,
    )
}
