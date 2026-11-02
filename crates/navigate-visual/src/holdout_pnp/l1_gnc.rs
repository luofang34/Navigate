//! Deterministic L1 IRLS and graduated non-convexity (GNC) pose solver with spatial holdout.

use super::{
    PoseCandidate, fit_pose_ransac, low_agl_relief_inlier_px, refine_pose_lm, scaled_inlier_px,
    score_pose_candidate, spatial_holdout,
    two_point::{solve_2pt_gravity_pose, solve_2pt_rot_at_eye, triad_rotation},
};
use nalgebra::{Matrix3, Matrix6, Rotation3, Vector3, Vector6};

const L1_STEPS: usize = 6;
const GNC_STAGES: usize = 5;
const MAX_DISPLACEMENT_M: f64 = 5_000.0;

#[derive(Clone, Copy)]
enum TrainKernel {
    L1 { floor_px: f64, behind_px: f64 },
    Gnc { scale_sq: f64 },
}

impl TrainKernel {
    fn weight(self, r: f64) -> f64 {
        match self {
            Self::L1 { floor_px, .. } => 1.0 / r.max(floor_px),
            Self::Gnc { scale_sq } => (scale_sq / (r * r + scale_sq)).powi(2),
        }
    }

    fn loss(self, r: Option<f64>) -> f64 {
        match (self, r) {
            (Self::L1 { behind_px, .. }, None) => behind_px,
            (Self::L1 { floor_px, .. }, Some(e)) if e < floor_px => {
                0.5 * e * e / floor_px + 0.5 * floor_px
            }
            (Self::L1 { .. }, Some(e)) => e,
            (Self::Gnc { scale_sq }, None) => 0.5 * scale_sq,
            (Self::Gnc { scale_sq }, Some(e)) => 0.5 * scale_sq * e * e / (e * e + scale_sq),
        }
    }
}

struct TrainOptimizer<'a> {
    train_idx: &'a [usize],
    world: &'a [[f64; 3]],
    pixels: &'a [[f64; 2]],
    k: &'a Matrix3<f64>,
    eye: [f64; 3],
    rot: Matrix3<f64>,
    damping: f64,
}

impl TrainOptimizer<'_> {
    fn step(&mut self, kernel: TrainKernel) -> Option<f64> {
        let mut h = Matrix6::<f64>::zeros();
        let mut g = Vector6::<f64>::zeros();
        let mut active = 0usize;
        let eye_v = Vector3::from(self.eye);
        for &i in self.train_idx {
            let Some((ru, rv, ju, jv)) =
                point_jacobian(self.world[i], self.pixels[i], eye_v, &self.rot, self.k)
            else {
                continue;
            };
            let w = kernel.weight(ru.hypot(rv));
            if w > 0.0 {
                active += 1;
                h += (ju * ju.transpose() + jv * jv.transpose()) * w;
                g += (ju * ru + jv * rv) * w;
            }
        }
        if active < 6 {
            return None;
        }
        if g.norm() < 1e-9 {
            return Some(0.0);
        }
        for d in 0..6 {
            h[(d, d)] += self.damping * h[(d, d)].max(1e-9);
        }
        let delta = h.cholesky()?.solve(&(-g));
        if !delta.iter().all(|v| v.is_finite()) {
            return None;
        }
        let step_norm = delta.norm();
        if step_norm < 1e-7 {
            return Some(step_norm);
        }
        let cur_cost = self.cost(self.eye, &self.rot, kernel);
        let cand_rot =
            Rotation3::new(Vector3::new(delta[0], delta[1], delta[2])).into_inner() * self.rot;
        let cand_eye = [
            self.eye[0] + delta[3],
            self.eye[1] + delta[4],
            self.eye[2] + delta[5],
        ];
        if self.cost(cand_eye, &cand_rot, kernel) < cur_cost {
            self.eye = cand_eye;
            self.rot = cand_rot;
            self.damping = (self.damping * 0.5).max(1e-6);
        } else {
            self.damping = (self.damping * 8.0).min(1e6);
        }
        Some(step_norm)
    }

    fn cost(&self, eye: [f64; 3], rot: &Matrix3<f64>, kernel: TrainKernel) -> f64 {
        self.train_idx
            .iter()
            .map(|&i| {
                kernel.loss(reproj_error(
                    self.world[i],
                    self.pixels[i],
                    eye,
                    rot,
                    self.k,
                ))
            })
            .sum()
    }

    fn max_residual(&self, cap_px: f64) -> f64 {
        self.train_idx
            .iter()
            .filter_map(|&i| {
                reproj_error(self.world[i], self.pixels[i], self.eye, &self.rot, self.k)
            })
            .fold(0.0_f64, |acc, r| acc.max(r.min(cap_px)))
    }
}

/// Fit and score a 6-DoF camera pose deterministically with L1 IRLS and GNC.
///
/// This solver fits the pose on training correspondences outside [`spatial_holdout`].
/// It rejects translations larger than `5,000` metres from `initial_pose`.
/// It then scores the pose on held-out grid cells.
#[must_use]
pub fn fit_pose_l1_gnc(
    world_points: &[[f64; 3]],
    image_pixels: &[[f64; 2]],
    k: &Matrix3<f64>,
    size: [u32; 2],
    initial_pose: ([f64; 3], Matrix3<f64>),
) -> Option<PoseCandidate> {
    if world_points.len() != image_pixels.len()
        || world_points.len() < 16
        || !initial_pose.0.iter().all(|v| v.is_finite())
        || !initial_pose.1.iter().all(|v| v.is_finite())
        || !k.iter().all(|v| v.is_finite())
    {
        return None;
    }
    let held = spatial_holdout(image_pixels, size);
    let train_idx: Vec<usize> = (0..world_points.len()).filter(|&i| !held[i]).collect();
    let hold_idx: Vec<usize> = (0..world_points.len()).filter(|&i| held[i]).collect();
    if train_idx.len() < 12 || hold_idx.is_empty() {
        return None;
    }

    let base_inlier_px = scaled_inlier_px(size);
    let inlier_px = low_agl_relief_inlier_px(base_inlier_px, world_points, initial_pose.0, k);
    let floor_px = 0.5 * (base_inlier_px / 5.0).max(1.0);
    let cap_px = 2.0 * f64::from(size[0].max(size[1]).max(640));

    let mut opt = TrainOptimizer {
        train_idx: &train_idx,
        world: world_points,
        pixels: image_pixels,
        k,
        eye: initial_pose.0,
        rot: initial_pose.1,
        damping: 1e-3,
    };

    let l1_kernel = TrainKernel::L1 {
        floor_px,
        behind_px: 2.0 * cap_px,
    };
    for _ in 0..L1_STEPS {
        if opt.step(l1_kernel)? < 1e-6 {
            break;
        }
    }

    let mu_start = (opt.max_residual(cap_px) / inlier_px).powi(2).max(4.0);
    for stage in 0..GNC_STAGES {
        let mu = mu_start.powf(1.0 - (stage as f64) / ((GNC_STAGES - 1) as f64));
        let gnc_kernel = TrainKernel::Gnc {
            scale_sq: mu * inlier_px * inlier_px,
        };
        opt.damping = 1e-3;
        if opt.step(gnc_kernel)? >= 1e-4 {
            let _second = opt.step(gnc_kernel)?;
        }
    }

    let (mut eye, mut rot) = (opt.eye, opt.rot);
    for thresh in [inlier_px * 1.5, inlier_px] {
        let inlier_train: Vec<usize> = train_idx
            .iter()
            .copied()
            .filter(|&i| {
                reproj_error(world_points[i], image_pixels[i], eye, &rot, k)
                    .is_some_and(|e| e <= thresh)
            })
            .collect();
        if inlier_train.len() >= 4 {
            (eye, rot) = refine_pose_lm(&inlier_train, world_points, image_pixels, eye, rot, k);
        }
    }

    if !eye.iter().all(|v| v.is_finite())
        || !rot.iter().all(|v| v.is_finite())
        || (Vector3::from(eye) - Vector3::from(initial_pose.0)).norm() > MAX_DISPLACEMENT_M
    {
        return None;
    }

    Some(score_pose_candidate(
        world_points,
        image_pixels,
        &held,
        (eye, rot),
        k,
        size,
        inlier_px,
    ))
}

/// Fit a 6-DoF camera pose with deterministic L1-GNC first and RANSAC as fallback.
///
/// When `initial_pose` is `Some`, this function runs [`fit_pose_l1_gnc`] first.
/// It returns the candidate immediately if `geometry_accepted` is `true`.
/// Otherwise, it runs [`fit_pose_ransac`] and returns the higher-priority candidate.
#[must_use]
pub fn fit_pose_hybrid(
    world_points: &[[f64; 3]],
    image_pixels: &[[f64; 2]],
    k: &Matrix3<f64>,
    size: [u32; 2],
    initial_pose: Option<([f64; 3], Matrix3<f64>)>,
) -> Option<PoseCandidate> {
    let steep_nadir_init = initial_pose.is_some_and(|(_, r0)| -r0[(2, 2)] > 0.90);
    let gnc =
        initial_pose.and_then(|init| fit_pose_l1_gnc(world_points, image_pixels, k, size, init));
    if gnc.as_ref().is_some_and(|c| {
        c.score.geometry_accepted && (!steep_nadir_init || -c.rotation[(2, 2)] >= 0.85)
    }) {
        return gnc;
    }
    let ransac = fit_pose_ransac(world_points, image_pixels, k, size, initial_pose);
    match (gnc, ransac) {
        (Some(g), Some(r)) => {
            let g_tilted_planar = steep_nadir_init
                && !g.score.geometry_accepted
                && -g.rotation[(2, 2)] < 0.94
                && -r.rotation[(2, 2)] >= 0.95
                && r.score.inliers + 5 >= g.score.inliers;
            if !g_tilted_planar
                && super::candidate_priority(&g.score) >= super::candidate_priority(&r.score)
            {
                Some(g)
            } else {
                Some(r)
            }
        }
        (Some(g), None) => Some(g),
        (None, r) => r,
    }
}

/// Triangulate a 6-DoF `(eye, rotation)` seed from 2-ray bundles and triad alignment.
#[must_use]
pub fn refine_seed_eye_2pt(
    world_pts: &[[f64; 3]],
    pixels: &[[f64; 2]],
    (hyp_eye, hyp_rot): ([f64; 3], Matrix3<f64>),
    k: &Matrix3<f64>,
) -> Option<([f64; 3], Matrix3<f64>)> {
    let n = world_pts.len().min(pixels.len());
    if n < 12 {
        return None;
    }
    let (fx, fy, cx, cy) = (k[(0, 0)], k[(1, 1)], k[(0, 2)], k[(1, 2)]);
    let rt = hyp_rot.transpose();
    let cam_rays: Vec<Vector3<f64>> = pixels[..n]
        .iter()
        .map(|uv| Vector3::new((uv[0] - cx) / fx, (uv[1] - cy) / fy, 1.0).normalize())
        .collect();
    let world_rays: Vec<Vector3<f64>> = cam_rays.iter().map(|c| rt * c).collect();
    let hyp_v = Vector3::from(hyp_eye);
    let max_pt_z = world_pts[..n]
        .iter()
        .map(|w| w[2])
        .fold(f64::NEG_INFINITY, f64::max);
    let min_att_cos = 28.0_f64.to_radians().cos();
    let mut best_pose = (hyp_eye, hyp_rot);
    let mut best_cnt = 0_usize;
    let total_pairs = n * (n - 1) / 2;
    let stride = (total_pairs / 360).max(1);
    let mut pair_idx = 0_usize;

    for i in 0..n {
        for j in (i + 1)..n {
            pair_idx += 1;
            if !pair_idx.is_multiple_of(stride) {
                continue;
            }
            let dx_px = pixels[i][0] - pixels[j][0];
            let dy_px = pixels[i][1] - pixels[j][1];
            if dx_px * dx_px + dy_px * dy_px < 1_600.0
                || cam_rays[i].dot(&cam_rays[j]).abs() > 0.998
            {
                continue;
            }
            if let Some(cand_g) = solve_2pt_gravity_pose(
                world_pts[i],
                world_pts[j],
                pixels[i],
                pixels[j],
                (hyp_eye, hyp_rot),
                k,
            ) {
                score_2pt_seed(
                    &world_pts[..n],
                    &pixels[..n],
                    k,
                    cand_g,
                    &mut best_cnt,
                    &mut best_pose,
                );
            }
            if let Some(cand_r) =
                solve_2pt_rot_at_eye(world_pts[i], world_pts[j], pixels[i], pixels[j], hyp_eye, k)
                && (((cand_r.1 * rt).trace() - 1.0) * 0.5) >= 0.65
            {
                score_2pt_seed(
                    &world_pts[..n],
                    &pixels[..n],
                    k,
                    cand_r,
                    &mut best_cnt,
                    &mut best_pose,
                );
            }
            let p_i = Matrix3::identity() - world_rays[i] * world_rays[i].transpose();
            let p_j = Matrix3::identity() - world_rays[j] * world_rays[j].transpose();
            let wi = Vector3::from(world_pts[i]);
            let wj = Vector3::from(world_pts[j]);
            let Some(chol) = (p_i + p_j).cholesky() else {
                continue;
            };
            let eye_v = chol.solve(&(p_i * wi + p_j * wj));
            let di_raw = wi - eye_v;
            let dj_raw = wj - eye_v;
            let horiz_shift = (eye_v.x - hyp_v.x).hypot(eye_v.y - hyp_v.y);
            if horiz_shift > 1_800.0
                || (eye_v.z - hyp_v.z).abs() > 650.0
                || eye_v.z < max_pt_z + 4.5
                || di_raw.dot(&world_rays[i]) <= 4.5
                || dj_raw.dot(&world_rays[j]) <= 4.5
            {
                continue;
            }
            let Some(rot_ij) = triad_rotation(cam_rays[i], cam_rays[j], di_raw, dj_raw) else {
                continue;
            };
            let att_cos = ((rot_ij * rt).trace() - 1.0) * 0.5;
            if att_cos < min_att_cos {
                continue;
            }
            score_2pt_seed(
                &world_pts[..n],
                &pixels[..n],
                k,
                ([eye_v.x, eye_v.y, eye_v.z], rot_ij),
                &mut best_cnt,
                &mut best_pose,
            );
        }
    }
    (best_cnt >= 6).then_some(best_pose)
}

fn score_2pt_seed(
    world_pts: &[[f64; 3]],
    pixels: &[[f64; 2]],
    k: &Matrix3<f64>,
    (eye, rot): ([f64; 3], Matrix3<f64>),
    best_cnt: &mut usize,
    best_pose: &mut ([f64; 3], Matrix3<f64>),
) {
    let (fx, fy, cx, cy) = (k[(0, 0)], k[(1, 1)], k[(0, 2)], k[(1, 2)]);
    let eye_v = Vector3::from(eye);
    let mut cells = 0_u64;
    let mut cnt = 0_usize;
    for (w, px) in world_pts.iter().zip(pixels) {
        let pc = rot * (Vector3::from(*w) - eye_v);
        if pc.z > 4.5 {
            let du = fx * (pc.x / pc.z) + cx - px[0];
            let dv = fy * (pc.y / pc.z) + cy - px[1];
            if du * du + dv * dv <= 784.0 {
                cnt += 1;
                let bx = ((px[0] / (cx * 2.0).max(1.0)) * 8.0).clamp(0.0, 7.0) as u32;
                let by = ((px[1] / (cy * 2.0).max(1.0)) * 8.0).clamp(0.0, 7.0) as u32;
                cells |= 1_u64 << (by * 8 + bx);
            }
        }
    }
    if cells.count_ones() >= 3 && cnt > *best_cnt {
        *best_cnt = cnt;
        *best_pose = (eye, rot);
    }
}

/// Compute pixel reprojection residuals and `6`-DoF pose Jacobians for one 3D-2D pair.
#[must_use]
pub fn point_jacobian(
    world: [f64; 3],
    pixel: [f64; 2],
    eye: Vector3<f64>,
    rot: &Matrix3<f64>,
    k: &Matrix3<f64>,
) -> Option<(f64, f64, Vector6<f64>, Vector6<f64>)> {
    let pc = rot * (Vector3::from(world) - eye);
    if pc.z <= 0.1 || !pc.z.is_finite() {
        return None;
    }
    let (fx, fy, cx, cy) = (k[(0, 0)], k[(1, 1)], k[(0, 2)], k[(1, 2)]);
    let inv_z = 1.0 / pc.z;
    let inv_z2 = inv_z * inv_z;
    let ru = fx * pc.x * inv_z + cx - pixel[0];
    let rv = fy * pc.y * inv_z + cy - pixel[1];
    let du = Vector3::new(fx * inv_z, 0.0, -fx * pc.x * inv_z2);
    let dv = Vector3::new(0.0, fy * inv_z, -fy * pc.y * inv_z2);
    let j_rot_u = pc.cross(&du);
    let j_rot_v = pc.cross(&dv);
    let j_eye_u = -(rot.transpose() * du);
    let j_eye_v = -(rot.transpose() * dv);
    let ju = Vector6::new(
        j_rot_u.x, j_rot_u.y, j_rot_u.z, j_eye_u.x, j_eye_u.y, j_eye_u.z,
    );
    let jv = Vector6::new(
        j_rot_v.x, j_rot_v.y, j_rot_v.z, j_eye_v.x, j_eye_v.y, j_eye_v.z,
    );
    Some((ru, rv, ju, jv))
}

/// Compute Euclidean pixel reprojection error for a point in front of the camera.
#[must_use]
pub fn reproj_error(
    world: [f64; 3],
    pixel: [f64; 2],
    eye: [f64; 3],
    rot: &Matrix3<f64>,
    k: &Matrix3<f64>,
) -> Option<f64> {
    let pc = rot * (Vector3::from(world) - Vector3::from(eye));
    if pc.z <= 0.1 {
        return None;
    }
    let u = k[(0, 0)] * (pc.x / pc.z) + k[(0, 2)];
    let v = k[(1, 1)] * (pc.y / pc.z) + k[(1, 2)];
    Some((u - pixel[0]).hypot(v - pixel[1]))
}
