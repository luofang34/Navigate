//! Spatial-holdout Perspective-n-Point (PnP) solvers and geometric scoring.

pub mod holdout;
mod l1_gnc;
mod scoring;
#[cfg(test)]
mod tests;
mod two_point;

pub use holdout::{MaskAwareSpatialHoldout, QueryRayDomain};
pub use l1_gnc::{
    fit_pose_hybrid, fit_pose_l1_gnc, point_jacobian, refine_seed_eye_2pt, reproj_error,
};
pub use scoring::candidate_priority;
pub(crate) use scoring::{low_agl_relief_inlier_px, scaled_inlier_px, score_pose_candidate};
use two_point::{
    dlt_pnp, sample_six, select_refined_ransac_candidate, solve_2pt_gravity_pose,
    solve_2pt_rot_at_eye,
};

use nalgebra::{Matrix3, Matrix6, Rotation3, Vector3, Vector6};

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

    let base_inlier_px = scaled_inlier_px(size);
    let inlier_px = initial_pose.map_or(base_inlier_px, |(eye0, _)| {
        low_agl_relief_inlier_px(base_inlier_px, world_points, eye0, k)
    });
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
