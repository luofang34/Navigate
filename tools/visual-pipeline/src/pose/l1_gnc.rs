//! Deterministic L1 IRLS and graduated non-convexity pose solver with spatial holdout.

use super::{
    PoseCandidate, fit_pose_ransac, point_jacobian, refine_pose_lm, reproj_error, scaled_inlier_px,
    score_pose_candidate, spatial_holdout,
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

    let inlier_px = scaled_inlier_px(size);
    let floor_px = 0.5 * (inlier_px / 5.0).max(1.0);
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
/// Otherwise, it calls [`fit_pose_ransac`].
pub fn fit_pose_hybrid(
    world_points: &[[f64; 3]],
    image_pixels: &[[f64; 2]],
    k: &Matrix3<f64>,
    size: [u32; 2],
    initial_pose: Option<([f64; 3], Matrix3<f64>)>,
) -> Option<PoseCandidate> {
    let gnc =
        initial_pose.and_then(|init| fit_pose_l1_gnc(world_points, image_pixels, k, size, init));
    if gnc.as_ref().is_some_and(|c| c.score.geometry_accepted) {
        return gnc;
    }
    let ransac = fit_pose_ransac(world_points, image_pixels, k, size, initial_pose);
    match (gnc, ransac) {
        (Some(g), Some(r)) => {
            if super::candidate_priority(&g.score) >= super::candidate_priority(&r.score) {
                Some(g)
            } else {
                Some(r)
            }
        }
        (Some(g), None) => Some(g),
        (None, r) => r,
    }
}

/// Triangulate a 6-DoF `(eye, rotation)` seed from 2-ray bundles and symmetric triad alignment.
pub(crate) fn refine_seed_eye_2pt(
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
            if dx_px * dx_px + dy_px * dy_px < 3_600.0
                || cam_rays[i].dot(&cam_rays[j]).abs() > 0.995
            {
                continue;
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
                || eye_v.z < max_pt_z + 80.0
                || di_raw.dot(&world_rays[i]) <= 80.0
                || dj_raw.dot(&world_rays[j]) <= 80.0
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
            let mut cells = 0_u64;
            let mut cnt = 0_usize;
            for m in 0..n {
                let pc = rot_ij * (Vector3::from(world_pts[m]) - eye_v);
                if pc.z > 80.0 {
                    let du = fx * (pc.x / pc.z) + cx - pixels[m][0];
                    let dv = fy * (pc.y / pc.z) + cy - pixels[m][1];
                    if du * du + dv * dv <= 784.0 {
                        cnt += 1;
                        let bx =
                            ((pixels[m][0] / (cx * 2.0).max(1.0)) * 8.0).clamp(0.0, 7.0) as u32;
                        let by =
                            ((pixels[m][1] / (cy * 2.0).max(1.0)) * 8.0).clamp(0.0, 7.0) as u32;
                        cells |= 1_u64 << (by * 8 + bx);
                    }
                }
            }
            if cells.count_ones() >= 4 && cnt > best_cnt {
                best_cnt = cnt;
                best_pose = ([eye_v.x, eye_v.y, eye_v.z], rot_ij);
            }
        }
    }
    (best_cnt >= 8).then_some(best_pose)
}

fn triad_rotation(
    ci: Vector3<f64>,
    cj: Vector3<f64>,
    di_raw: Vector3<f64>,
    dj_raw: Vector3<f64>,
) -> Option<Matrix3<f64>> {
    let di = di_raw.try_normalize(1e-6)?;
    let dj = dj_raw.try_normalize(1e-6)?;
    let bc1 = (ci + cj).try_normalize(1e-6)?;
    let bc2 = ci.cross(&cj).try_normalize(1e-6)?;
    let bc3 = bc1.cross(&bc2);
    let bw1 = (di + dj).try_normalize(1e-6)?;
    let bw2 = di.cross(&dj).try_normalize(1e-6)?;
    let bw3 = bw1.cross(&bw2);
    let c_mat = Matrix3::from_columns(&[bc1, bc2, bc3]);
    let w_mat = Matrix3::from_columns(&[bw1, bw2, bw3]);
    Some(c_mat * w_mat.transpose())
}

#[cfg(test)]
mod tests {
    use super::*;

    struct HoldoutScene {
        world: Vec<[f64; 3]>,
        pixels: Vec<[f64; 2]>,
        k: Matrix3<f64>,
        size: [u32; 2],
        true_eye: [f64; 3],
        true_rot: Matrix3<f64>,
    }

    fn synthetic_holdout_scene(scale_m: f64, outlier_every_five: usize) -> HoldoutScene {
        let size = [960, 720];
        let k = Matrix3::new(780.0, 0.0, 479.5, 0.0, 780.0, 359.5, 0.0, 0.0, 1.0);
        let true_eye = [0.21 * scale_m, -0.14 * scale_m, 0.92 * scale_m];
        let look_down = Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0);
        let tilt = Rotation3::from_euler_angles(0.16, -0.11, 0.35).into_inner();
        let true_rot = tilt * look_down;

        let mut world = Vec::with_capacity(80);
        let mut pixels = Vec::with_capacity(80);
        for gy in 0..8 {
            for gx in 0..10 {
                let u = 48.0 + f64::from(gx) * 96.0;
                let v = 45.0 + f64::from(gy) * 90.0;
                let depth =
                    scale_m * (0.82 + 0.14 * (f64::from(gx) * 0.7 + f64::from(gy) * 0.9).sin());
                let ray_cam = Vector3::new(
                    (u - k[(0, 2)]) * depth / k[(0, 0)],
                    (v - k[(1, 2)]) * depth / k[(1, 1)],
                    depth,
                );
                let pt = Vector3::from(true_eye) + true_rot.transpose() * ray_cam;
                world.push([pt.x, pt.y, pt.z]);
                pixels.push([u, v]);
            }
        }

        let mut outliers = 0usize;
        for (idx, px) in pixels.iter_mut().enumerate() {
            if (idx * 3 + 1) % 5 < outlier_every_five {
                let mag = 120.0 + (outliers as f64) * 6.0;
                let ang = 0.4 + (outliers as f64) * 0.23;
                px[0] += mag * ang.cos();
                px[1] += mag * ang.sin();
                outliers += 1;
            }
        }
        HoldoutScene {
            world,
            pixels,
            k,
            size,
            true_eye,
            true_rot,
        }
    }

    #[test]
    fn l1_gnc_and_hybrid_accept_accurate_pose_under_40_percent_outliers() {
        let s = synthetic_holdout_scene(1_000.0, 2);
        let init_eye = [
            s.true_eye[0] + 65.0,
            s.true_eye[1] - 45.0,
            s.true_eye[2] + 30.0,
        ];
        let init_rot = Rotation3::from_euler_angles(0.06, -0.05, 0.07).into_inner() * s.true_rot;

        let cand = fit_pose_l1_gnc(&s.world, &s.pixels, &s.k, s.size, (init_eye, init_rot))
            .expect("L1-GNC must produce a pose candidate");
        let repeat = fit_pose_l1_gnc(&s.world, &s.pixels, &s.k, s.size, (init_eye, init_rot))
            .expect("repeated L1-GNC call must match");
        assert_eq!(cand, repeat);
        assert!(cand.score.geometry_accepted, "score: {:?}", cand.score);

        let eye_err = (Vector3::from(cand.eye) - Vector3::from(s.true_eye)).norm();
        let rel_rot = cand.rotation * s.true_rot.transpose();
        let trace_cos = ((rel_rot.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
        let rot_err_deg = trace_cos.acos().to_degrees();
        assert!(eye_err < 0.05, "eye error {eye_err} m >= 0.05 m");
        assert!(
            rot_err_deg < 0.01,
            "rotation error {rot_err_deg} deg >= 0.01 deg"
        );
        assert_eq!(cand.score.inliers, 48);

        let hybrid = fit_pose_hybrid(
            &s.world,
            &s.pixels,
            &s.k,
            s.size,
            Some((init_eye, init_rot)),
        )
        .expect("hybrid solver must accept deterministic L1-GNC candidate");
        assert_eq!(hybrid, cand);

        let near = synthetic_holdout_scene(10.0, 1);
        let fallback = fit_pose_hybrid(&near.world, &near.pixels, &near.k, near.size, None)
            .expect("hybrid solver without prior must fall back to RANSAC");
        assert!(fallback.score.geometry_accepted);
    }

    #[test]
    fn enforces_displacement_guard_and_minimum_pair_count() {
        let s = synthetic_holdout_scene(1_000.0, 2);
        assert!(
            fit_pose_l1_gnc(
                &s.world[..15],
                &s.pixels[..15],
                &s.k,
                s.size,
                (s.true_eye, s.true_rot)
            )
            .is_none()
        );

        let far_eye = [s.true_eye[0] + 5_500.0, s.true_eye[1], s.true_eye[2]];
        assert!(
            fit_pose_l1_gnc(&s.world, &s.pixels, &s.k, s.size, (far_eye, s.true_rot)).is_none()
        );
    }
}
