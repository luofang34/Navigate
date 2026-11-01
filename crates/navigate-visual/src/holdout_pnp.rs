//! Spatial-holdout Perspective-n-Point (PnP) solvers and geometric scoring.

pub mod holdout;
mod l1_gnc;
#[cfg(test)]
mod tests;
mod two_point;

pub use holdout::{MaskAwareSpatialHoldout, QueryRayDomain};
pub use l1_gnc::{
    fit_pose_hybrid, fit_pose_l1_gnc, point_jacobian, refine_seed_eye_2pt, reproj_error,
};
use two_point::{
    dlt_pnp, select_refined_ransac_candidate, solve_2pt_gravity_pose, solve_2pt_rot_at_eye,
};

use nalgebra::{Matrix3, Matrix6, Rotation3, Vector3, Vector6};
use std::collections::BTreeSet;

const INLIER_PX: f64 = 5.0;
const GRID_DIM: u32 = 12;

/// Geometric verification metrics for a solved camera pose candidate.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PoseScore {
    /// Camera focal length in pixels.
    pub focal_px: f64,
    /// `true` when all holdout and coverage thresholds pass.
    pub geometry_accepted: bool,
    /// Number of held-out correspondences.
    pub holdout_count: usize,
    /// Fraction of held-out correspondences within the scaled inlier threshold.
    pub holdout_inlier_fraction: f64,
    /// Median reprojection error in pixels on held-out correspondences.
    pub holdout_median_px: f64,
    /// Fraction of `12 x 12` image grid cells covered by inliers.
    pub image_support_fraction: f64,
    /// Total inliers across training and holdout splits.
    pub inliers: usize,
    /// Correspondence count.
    pub pairs: usize,
    /// Fraction of correspondences in front of the camera.
    pub positive_depth_fraction: f64,
    /// Number of inliers in the training split.
    pub train_inliers: usize,
    /// Root-mean-square reprojection error on training inliers in pixels.
    pub train_rms_px: f64,
}

/// Verified six-degree-of-freedom (6-DoF) camera pose candidate and correspondences.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseCandidate {
    /// Optical centre in local East-North-Up (ENU) coordinates `[east_m, north_m, up_m]`.
    pub eye: [f64; 3],
    /// World-to-camera `3 x 3` rotation matrix.
    pub rotation: Matrix3<f64>,
    /// Pinhole intrinsic matrix `K`.
    pub k: Matrix3<f64>,
    /// Image dimensions `[width, height]`.
    pub size: [u32; 2],
    /// Three-dimensional world points in local ENU coordinates.
    pub points: Vec<[f64; 3]>,
    /// Two-dimensional image pixel coordinates `[u, v]`.
    pub pixels: Vec<[f64; 2]>,
    /// Boolean mask of correspondences within the scaled inlier threshold.
    pub inliers: Vec<bool>,
    /// Geometric score summary.
    pub score: PoseScore,
}

/// Compute a deterministic spatial holdout mask over `12 x 12` cells where `(gx + gy) % 4 == 0`.
#[must_use]
pub fn spatial_holdout(pixels: &[[f64; 2]], size: [u32; 2]) -> Vec<bool> {
    let w = f64::from(size[0].max(1));
    let h = f64::from(size[1].max(1));
    pixels
        .iter()
        .map(|uv| {
            let gx = ((uv[0] / w) * f64::from(GRID_DIM)).floor().clamp(0.0, 11.0) as u32;
            let gy = ((uv[1] / h) * f64::from(GRID_DIM)).floor().clamp(0.0, 11.0) as u32;
            (gx + gy).is_multiple_of(4)
        })
        .collect()
}

/// Fit and score a 6-DoF camera pose from 3D-2D correspondences with RANSAC and spatial holdout.
///
/// This function refines `[eye, rvec]` on training inliers only.
/// It then evaluates reprojection error on held-out correspondences.
#[must_use]
pub fn fit_pose_ransac(
    world_points: &[[f64; 3]],
    image_pixels: &[[f64; 2]],
    k: &Matrix3<f64>,
    size: [u32; 2],
    initial_pose: Option<([f64; 3], Matrix3<f64>)>,
) -> Option<PoseCandidate> {
    if world_points.len() != image_pixels.len() || world_points.len() < 16 {
        return None;
    }
    let held = spatial_holdout(image_pixels, size);
    let train_idx: Vec<usize> = (0..world_points.len()).filter(|&i| !held[i]).collect();
    let hold_idx: Vec<usize> = (0..world_points.len()).filter(|&i| held[i]).collect();
    if train_idx.len() < 12 || hold_idx.is_empty() {
        return None;
    }

    let inlier_px = scaled_inlier_px(size);
    let ransac_px = inlier_px * 2.0;
    let mut best_pose = initial_pose;
    let mut best_inliers = best_pose.map_or(0, |(eye0, rot0)| {
        count_inliers(
            world_points,
            image_pixels,
            &train_idx,
            eye0,
            &rot0,
            k,
            ransac_px,
        )
    });

    let trials = 96usize.min(train_idx.len() * 3);
    for t in 0..trials {
        let subset = sample_six(&train_idx, t as u64 + 1);
        if let Some((eye_c, rot_c)) = dlt_pnp(&subset, world_points, image_pixels, k)
            && initial_pose
                .is_none_or(|(e0, _)| (Vector3::from(eye_c) - Vector3::from(e0)).norm() <= 5_000.0)
        {
            let count = count_inliers(
                world_points,
                image_pixels,
                &train_idx,
                eye_c,
                &rot_c,
                k,
                ransac_px,
            );
            if count > best_inliers {
                best_inliers = count;
                best_pose = Some((eye_c, rot_c));
            }
        }
        if let Some((eye0, rot0)) = initial_pose {
            for (ia, ib) in [(0, 1), (2, 3), (4, 5)] {
                if let Some((eye_g, rot_g)) = solve_2pt_gravity_pose(
                    world_points[subset[ia]],
                    world_points[subset[ib]],
                    image_pixels[subset[ia]],
                    image_pixels[subset[ib]],
                    (eye0, rot0),
                    k,
                ) {
                    let count = count_inliers(
                        world_points,
                        image_pixels,
                        &train_idx,
                        eye_g,
                        &rot_g,
                        k,
                        ransac_px,
                    );
                    if count > best_inliers {
                        best_inliers = count;
                        best_pose = Some((eye_g, rot_g));
                    }
                }
                if let Some((eye_r, rot_r)) = solve_2pt_rot_at_eye(
                    world_points[subset[ia]],
                    world_points[subset[ib]],
                    image_pixels[subset[ia]],
                    image_pixels[subset[ib]],
                    eye0,
                    k,
                ) && (((rot_r * rot0.transpose()).trace() - 1.0) * 0.5) >= 0.65
                {
                    let count = count_inliers(
                        world_points,
                        image_pixels,
                        &train_idx,
                        eye_r,
                        &rot_r,
                        k,
                        ransac_px,
                    );
                    if count > best_inliers {
                        best_inliers = count;
                        best_pose = Some((eye_r, rot_r));
                    }
                }
            }
            let (eye_lm, rot_lm) =
                refine_pose_lm(&subset[..4], world_points, image_pixels, eye0, rot0, k);
            if (Vector3::from(eye_lm) - Vector3::from(eye0)).norm() <= 5_000.0 {
                let count = count_inliers(
                    world_points,
                    image_pixels,
                    &train_idx,
                    eye_lm,
                    &rot_lm,
                    k,
                    ransac_px,
                );
                if count > best_inliers {
                    best_inliers = count;
                    best_pose = Some((eye_lm, rot_lm));
                }
            }
        }
    }

    let (init_best_eye, init_best_rot) = best_pose?;
    let (mut eye, mut rot) = (init_best_eye, init_best_rot);
    for thresh in [inlier_px * 3.0, inlier_px * 1.5, inlier_px] {
        let inlier_train: Vec<usize> = train_idx
            .iter()
            .copied()
            .filter(|&i| {
                reproj_error(world_points[i], image_pixels[i], eye, &rot, k)
                    .is_some_and(|e| e <= thresh)
            })
            .collect();
        if inlier_train.len() >= 4 {
            let (re, rr) = refine_pose_lm(&inlier_train, world_points, image_pixels, eye, rot, k);
            eye = re;
            rot = rr;
        }
    }

    let cand_6dof = score_pose_candidate(
        world_points,
        image_pixels,
        &held,
        (eye, rot),
        k,
        size,
        inlier_px,
    );
    Some(select_refined_ransac_candidate(
        world_points,
        image_pixels,
        &train_idx,
        &held,
        (init_best_eye, init_best_rot),
        cand_6dof,
        initial_pose,
        k,
        size,
        inlier_px,
    ))
}

pub(crate) fn scaled_inlier_px(size: [u32; 2]) -> f64 {
    INLIER_PX * (f64::from(size[0].max(size[1])) / 640.0).max(1.0)
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
    let hold_count = held.iter().filter(|&&h| h).count();
    let mut inliers = vec![false; world_points.len()];
    let (mut pos_depth, mut train_sq_sum, mut train_inliers, mut hold_inliers) = (0, 0.0, 0, 0);
    let mut hold_errors = Vec::with_capacity(hold_count);
    let mut cells = BTreeSet::new();
    let (w, h) = (f64::from(size[0].max(1)), f64::from(size[1].max(1)));

    for i in 0..world_points.len() {
        let p_cam = rot * (Vector3::from(world_points[i]) - Vector3::from(eye));
        pos_depth += usize::from(p_cam.z > 0.1);
        let err = reproj_error(world_points[i], image_pixels[i], eye, &rot, k).unwrap_or(1e6);
        let ok = err <= inlier_px;
        inliers[i] = ok;
        if ok {
            let gx = ((image_pixels[i][0] / w) * f64::from(GRID_DIM))
                .floor()
                .clamp(0.0, 11.0) as u32;
            let gy = ((image_pixels[i][1] / h) * f64::from(GRID_DIM))
                .floor()
                .clamp(0.0, 11.0) as u32;
            cells.insert((gx, gy));
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

    let geometry_accepted = train_inliers >= 24
        && hold_inliers >= 8
        && hold_frac >= 0.40
        && hold_median <= 1.5 * inlier_px
        && pos_frac >= 0.98
        && cells.len() >= 6;

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

fn sample_six(indices: &[usize], seed: u64) -> [usize; 6] {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).max(1);
    let mut picked = [indices[0]; 6];
    let mut count = 0usize;
    let mut guard = 0usize;
    while count < 6 && guard < 128 {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        let idx = indices[(state.wrapping_mul(0x2545_F491_4F6C_DD1D) as usize) % indices.len()];
        if !picked[..count].contains(&idx) || indices.len() < 6 {
            picked[count] = idx;
            count += 1;
        }
        guard += 1;
    }
    picked
}

pub(crate) fn refine_pose_lm(
    indices: &[usize],
    world: &[[f64; 3]],
    pixels: &[[f64; 2]],
    mut eye: [f64; 3],
    mut rot: Matrix3<f64>,
    k: &Matrix3<f64>,
) -> ([f64; 3], Matrix3<f64>) {
    let mut lambda = 1e-2f64;
    let mut cur_cost = total_sq_err(indices, world, pixels, eye, &rot, k);
    for _ in 0..15 {
        let mut h = Matrix6::<f64>::zeros();
        let mut g = Vector6::<f64>::zeros();
        let eye_v = Vector3::from(eye);
        for &i in indices {
            if let Some((ru, rv, ju, jv)) = point_jacobian(world[i], pixels[i], eye_v, &rot, k) {
                h += ju * ju.transpose() + jv * jv.transpose();
                g += ju * ru + jv * rv;
            }
        }
        for d in 0..6 {
            h[(d, d)] += lambda;
        }
        let Some(step) = h.lu().solve(&(-g)) else {
            break;
        };
        if step.norm() < 1e-7 {
            break;
        }
        let cand_rot = Rotation3::new(Vector3::new(step[0], step[1], step[2])).into_inner() * rot;
        let cand_eye = [eye[0] + step[3], eye[1] + step[4], eye[2] + step[5]];
        let cand_cost = total_sq_err(indices, world, pixels, cand_eye, &cand_rot, k);
        if cand_cost < cur_cost {
            eye = cand_eye;
            rot = cand_rot;
            cur_cost = cand_cost;
            lambda = (lambda * 0.5).max(1e-6);
        } else {
            lambda = (lambda * 4.0).min(1e4);
        }
    }
    (eye, rot)
}

fn count_inliers(
    world: &[[f64; 3]],
    pixels: &[[f64; 2]],
    indices: &[usize],
    eye: [f64; 3],
    rot: &Matrix3<f64>,
    k: &Matrix3<f64>,
    inlier_px: f64,
) -> usize {
    indices
        .iter()
        .filter(|&&i| {
            reproj_error(world[i], pixels[i], eye, rot, k).is_some_and(|e| e <= inlier_px)
        })
        .count()
}

fn total_sq_err(
    indices: &[usize],
    world: &[[f64; 3]],
    pixels: &[[f64; 2]],
    eye: [f64; 3],
    rot: &Matrix3<f64>,
    k: &Matrix3<f64>,
) -> f64 {
    indices
        .iter()
        .map(|&i| {
            reproj_error(world[i], pixels[i], eye, rot, k)
                .unwrap_or(100.0)
                .powi(2)
        })
        .sum()
}
