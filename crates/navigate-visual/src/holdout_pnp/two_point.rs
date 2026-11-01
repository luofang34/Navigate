//! Two-point gravity-aligned and fixed-position pose hypotheses for planar and grazing views.

use super::{
    PoseCandidate, candidate_priority, point_jacobian, reproj_error, score_pose_candidate,
};
use nalgebra::{Matrix3, Matrix4, Rotation3, Vector3, Vector4};

const MIN_DEPTH_M: f64 = 4.5;
const MIN_PIXEL_SEP_PX: f64 = 24.0;

/// Compute a world-to-camera rotation matrix aligning two world rays to two unit camera rays.
#[must_use]
pub(super) fn triad_rotation(
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

/// Solve a 3-DoF world-to-camera rotation at a fixed `eye0` from two 3D-2D correspondences.
#[must_use]
pub(super) fn solve_2pt_rot_at_eye(
    wi: [f64; 3],
    wj: [f64; 3],
    px_i: [f64; 2],
    px_j: [f64; 2],
    eye0: [f64; 3],
    k: &Matrix3<f64>,
) -> Option<([f64; 3], Matrix3<f64>)> {
    if (px_i[0] - px_j[0]).hypot(px_i[1] - px_j[1]) < MIN_PIXEL_SEP_PX {
        return None;
    }
    let (fx, fy, cx, cy) = (
        k[(0, 0)].max(1e-6),
        k[(1, 1)].max(1e-6),
        k[(0, 2)],
        k[(1, 2)],
    );
    let ci = Vector3::new((px_i[0] - cx) / fx, (px_i[1] - cy) / fy, 1.0).normalize();
    let cj = Vector3::new((px_j[0] - cx) / fx, (px_j[1] - cy) / fy, 1.0).normalize();
    if ci.dot(&cj).abs() > 0.9995 {
        return None;
    }
    let eye_v = Vector3::from(eye0);
    let di_raw = Vector3::from(wi) - eye_v;
    let dj_raw = Vector3::from(wj) - eye_v;
    if di_raw.norm() <= MIN_DEPTH_M || dj_raw.norm() <= MIN_DEPTH_M {
        return None;
    }
    let rot = triad_rotation(ci, cj, di_raw, dj_raw)?;
    if (rot * di_raw).z <= MIN_DEPTH_M || (rot * dj_raw).z <= MIN_DEPTH_M {
        return None;
    }
    Some((eye0, rot))
}

/// Solve a 4-DoF `(eye, yaw)` pose keeping tilt from `hyp_rot` using two 3D-2D correspondences.
#[must_use]
pub(super) fn solve_2pt_gravity_pose(
    wi_in: [f64; 3],
    wj_in: [f64; 3],
    px_i_in: [f64; 2],
    px_j_in: [f64; 2],
    (hyp_eye, hyp_rot): ([f64; 3], Matrix3<f64>),
    k: &Matrix3<f64>,
) -> Option<([f64; 3], Matrix3<f64>)> {
    if (px_i_in[0] - px_j_in[0]).hypot(px_i_in[1] - px_j_in[1]) < MIN_PIXEL_SEP_PX {
        return None;
    }
    let (fx, fy, cx, cy) = (
        k[(0, 0)].max(1e-6),
        k[(1, 1)].max(1e-6),
        k[(0, 2)],
        k[(1, 2)],
    );
    let ci0 = Vector3::new((px_i_in[0] - cx) / fx, (px_i_in[1] - cy) / fy, 1.0).normalize();
    let cj0 = Vector3::new((px_j_in[0] - cx) / fx, (px_j_in[1] - cy) / fy, 1.0).normalize();
    let rt = hyp_rot.transpose();
    let qi0 = rt * ci0;
    let qj0 = rt * cj0;

    let (wi, wj, qi, qj) = if qi0.z.abs() > qj0.z.abs() {
        (wj_in, wi_in, qj0, qi0)
    } else {
        (wi_in, wj_in, qi0, qj0)
    };
    if qj.z.abs() <= 0.05 {
        return None;
    }

    let dw = Vector3::from(wj) - Vector3::from(wi);
    let dw_xy_sq = dw.x * dw.x + dw.y * dw.y;
    if dw_xy_sq < 0.25 {
        return None;
    }

    let alpha = dw.z / qj.z;
    let beta = qi.z / qj.z;
    let ax = alpha * qj.x;
    let ay = alpha * qj.y;
    let bx = beta * qj.x - qi.x;
    let by = beta * qj.y - qi.y;
    let qa = bx * bx + by * by;
    let qb = 2.0 * (ax * bx + ay * by);
    let qc = ax * ax + ay * ay - dw_xy_sq;
    let disc = qb * qb - 4.0 * qa * qc;
    if qa <= 1e-10 || disc < 0.0 || !disc.is_finite() {
        return None;
    }

    let sqrt_disc = disc.sqrt();
    let hyp_v = Vector3::from(hyp_eye);
    let is_nadir = -hyp_rot[(2, 2)] > 0.90;
    let max_ground_z = wi[2].max(wj[2]);
    type ScoredPose = (f64, ([f64; 3], Matrix3<f64>));
    let mut best: Option<ScoredPose> = None;

    for sign in [1.0_f64, -1.0_f64] {
        let li = (-qb + sign * sqrt_disc) / (2.0 * qa);
        let lj = alpha + beta * li;
        if li <= MIN_DEPTH_M || lj <= MIN_DEPTH_M {
            continue;
        }
        let vx = ax + bx * li;
        let vy = ay + by * li;
        let v_norm_sq = vx * vx + vy * vy;
        if v_norm_sq <= 1e-8 {
            continue;
        }
        let raw_cos = (vx * dw.x + vy * dw.y) / v_norm_sq;
        let raw_sin = (vx * dw.y - vy * dw.x) / v_norm_sq;
        let sc = raw_cos.hypot(raw_sin);
        if sc <= 0.5 {
            continue;
        }
        let cos_psi = raw_cos / sc;
        let sin_psi = raw_sin / sc;
        if !is_nadir && cos_psi < 0.70 {
            continue;
        }
        let g_z = Matrix3::new(cos_psi, -sin_psi, 0.0, sin_psi, cos_psi, 0.0, 0.0, 0.0, 1.0);
        let rot = hyp_rot * g_z.transpose();
        let eye_i = Vector3::from(wi) - li * (g_z * qi);
        let eye_j = Vector3::from(wj) - lj * (g_z * qj);
        let eye_v = (eye_i + eye_j) * 0.5;
        let shift = (eye_v - hyp_v).norm();
        if !eye_v.iter().all(|v| v.is_finite())
            || eye_v.z < max_ground_z + MIN_DEPTH_M
            || shift > 2_500.0
        {
            continue;
        }
        if best.is_none_or(|(prev_shift, _)| shift < prev_shift) {
            best = Some((shift, ([eye_v.x, eye_v.y, eye_v.z], rot)));
        }
    }
    best.map(|(_, pose)| pose)
}

/// Refine a 4-DoF gravity-aligned `(eye, world_yaw)` pose with Levenberg-Marquardt steps.
#[must_use]
pub(super) fn refine_gravity_pose_lm(
    indices: &[usize],
    world: &[[f64; 3]],
    pixels: &[[f64; 2]],
    mut eye: [f64; 3],
    mut rot: Matrix3<f64>,
    k: &Matrix3<f64>,
) -> ([f64; 3], Matrix3<f64>) {
    let mut lambda = 1e-2_f64;
    let mut cur_cost = rot_sq_err(indices, world, pixels, eye, &rot, k);
    for _ in 0..12 {
        let mut h = Matrix4::<f64>::zeros();
        let mut g = Vector4::<f64>::zeros();
        let eye_v = Vector3::from(eye);
        let rz = rot.column(2);
        let mut used = 0_usize;
        for &i in indices {
            if let Some((ru, rv, ju, jv)) = point_jacobian(world[i], pixels[i], eye_v, &rot, k) {
                let j4u = Vector4::new(
                    ju[0] * rz[0] + ju[1] * rz[1] + ju[2] * rz[2],
                    ju[3],
                    ju[4],
                    ju[5],
                );
                let j4v = Vector4::new(
                    jv[0] * rz[0] + jv[1] * rz[1] + jv[2] * rz[2],
                    jv[3],
                    jv[4],
                    jv[5],
                );
                h += j4u * j4u.transpose() + j4v * j4v.transpose();
                g += j4u * ru + j4v * rv;
                used += 1;
            }
        }
        if used < 2 {
            break;
        }
        for d in 0..4 {
            h[(d, d)] += lambda * h[(d, d)].max(1e-6);
        }
        let Some(step) = h.lu().solve(&(-g)) else {
            break;
        };
        if !step.iter().all(|v| v.is_finite()) || step.norm() < 1e-7 {
            break;
        }
        let cand_rot = rot * Rotation3::new(Vector3::new(0.0, 0.0, step[0])).into_inner();
        let cand_eye = [eye[0] + step[1], eye[1] + step[2], eye[2] + step[3]];
        let cand_cost = rot_sq_err(indices, world, pixels, cand_eye, &cand_rot, k);
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

/// Refine a 3-DoF world-to-camera rotation at fixed `eye` with Levenberg-Marquardt steps.
#[must_use]
pub(super) fn refine_rotation_lm(
    indices: &[usize],
    world: &[[f64; 3]],
    pixels: &[[f64; 2]],
    eye: [f64; 3],
    mut rot: Matrix3<f64>,
    k: &Matrix3<f64>,
) -> Matrix3<f64> {
    let mut lambda = 1e-2_f64;
    let eye_v = Vector3::from(eye);
    let mut cur_cost = rot_sq_err(indices, world, pixels, eye, &rot, k);
    for _ in 0..12 {
        let mut h = Matrix3::<f64>::zeros();
        let mut g = Vector3::<f64>::zeros();
        for &i in indices {
            if let Some((ru, rv, ju, jv)) = point_jacobian(world[i], pixels[i], eye_v, &rot, k) {
                let jru = Vector3::new(ju[0], ju[1], ju[2]);
                let jrv = Vector3::new(jv[0], jv[1], jv[2]);
                h += jru * jru.transpose() + jrv * jrv.transpose();
                g += jru * ru + jrv * rv;
            }
        }
        for d in 0..3 {
            h[(d, d)] += lambda;
        }
        let Some(step) = h.lu().solve(&(-g)) else {
            break;
        };
        if step.norm() < 1e-7 {
            break;
        }
        let cand_rot = Rotation3::new(step).into_inner() * rot;
        let cand_cost = rot_sq_err(indices, world, pixels, eye, &cand_rot, k);
        if cand_cost < cur_cost {
            rot = cand_rot;
            cur_cost = cand_cost;
            lambda = (lambda * 0.5).max(1e-6);
        } else {
            lambda = (lambda * 4.0).min(1e4);
        }
    }
    rot
}

fn rot_sq_err(
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

/// Solve a linear 6-point Direct Linear Transform (DLT) camera pose for non-coplanar points.
#[must_use]
pub(super) fn dlt_pnp(
    subset: &[usize; 6],
    world: &[[f64; 3]],
    pixels: &[[f64; 2]],
    k: &Matrix3<f64>,
) -> Option<([f64; 3], Matrix3<f64>)> {
    let fx = k[(0, 0)].max(1e-6);
    let fy = k[(1, 1)].max(1e-6);
    let cx = k[(0, 2)];
    let cy = k[(1, 2)];
    let mut a = nalgebra::DMatrix::<f64>::zeros(12, 12);
    for (row_pair, &idx) in subset.iter().enumerate() {
        let p = world[idx];
        let xn = (pixels[idx][0] - cx) / fx;
        let yn = (pixels[idx][1] - cy) / fy;
        let r0 = row_pair * 2;
        let r1 = r0 + 1;
        for c in 0..3 {
            a[(r0, c)] = p[c];
            a[(r0, 8 + c)] = -xn * p[c];
            a[(r1, 4 + c)] = p[c];
            a[(r1, 8 + c)] = -yn * p[c];
        }
        a[(r0, 3)] = 1.0;
        a[(r0, 11)] = -xn;
        a[(r1, 7)] = 1.0;
        a[(r1, 11)] = -yn;
    }
    let svd = a.svd(false, true);
    let v_t = svd.v_t?;
    let last = v_t.row(v_t.nrows() - 1);
    let m = Matrix3::new(
        last[0], last[1], last[2], last[4], last[5], last[6], last[8], last[9], last[10],
    );
    let mut t = Vector3::new(last[3], last[7], last[11]);
    let m_svd = m.svd(true, true);
    let (u, vt) = (m_svd.u?, m_svd.v_t?);
    let scale = m_svd.singular_values.sum() / 3.0;
    if scale <= 1e-12 {
        return None;
    }
    let mut r = u * vt;
    if r.determinant() < 0.0 {
        r = -r;
        t = -t;
    }
    t /= scale;
    let p0 = r * Vector3::from(world[subset[0]]) + t;
    if p0.z < 0.0 {
        r = -r;
        t = -t;
        if r.determinant() < 0.0 {
            return None;
        }
    }
    let eye = -(r.transpose() * t);
    Some(([eye.x, eye.y, eye.z], r))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn select_refined_ransac_candidate(
    world_points: &[[f64; 3]],
    image_pixels: &[[f64; 2]],
    train_idx: &[usize],
    held: &[bool],
    (init_eye, init_rot): ([f64; 3], Matrix3<f64>),
    cand_6dof: PoseCandidate,
    initial_pose: Option<([f64; 3], Matrix3<f64>)>,
    k: &Matrix3<f64>,
    size: [u32; 2],
    inlier_px: f64,
) -> PoseCandidate {
    let mut best = cand_6dof;
    if initial_pose.is_some() {
        let (mut eye_g, mut rot_g) = (init_eye, init_rot);
        for thresh in [inlier_px * 3.0, inlier_px * 1.5, inlier_px] {
            let inl: Vec<usize> = train_idx
                .iter()
                .copied()
                .filter(|&i| {
                    reproj_error(world_points[i], image_pixels[i], eye_g, &rot_g, k)
                        .is_some_and(|e| e <= thresh)
                })
                .collect();
            if inl.len() >= 2 {
                (eye_g, rot_g) =
                    refine_gravity_pose_lm(&inl, world_points, image_pixels, eye_g, rot_g, k);
            }
        }
        let cand_4dof = score_pose_candidate(
            world_points,
            image_pixels,
            held,
            (eye_g, rot_g),
            k,
            size,
            inlier_px,
        );
        let axis_cos = best.rotation.row(2).dot(&rot_g.row(2));
        let planar_overfit = !best.score.geometry_accepted
            && best.score.inliers < 22
            && axis_cos < 0.965
            && cand_4dof.score.inliers + 2 >= best.score.inliers;
        if planar_overfit || candidate_priority(&cand_4dof.score) > candidate_priority(&best.score)
        {
            best = cand_4dof;
        }
    }
    if let Some((eye0, _)) = initial_pose
        && (Vector3::from(best.eye) - Vector3::from(eye0)).norm() > 600.0
    {
        let mut rot_fixed = init_rot;
        for thresh in [inlier_px * 3.0, inlier_px * 1.5, inlier_px] {
            let inl: Vec<usize> = train_idx
                .iter()
                .copied()
                .filter(|&i| {
                    reproj_error(world_points[i], image_pixels[i], eye0, &rot_fixed, k)
                        .is_some_and(|e| e <= thresh)
                })
                .collect();
            if inl.len() >= 3 {
                rot_fixed =
                    refine_rotation_lm(&inl, world_points, image_pixels, eye0, rot_fixed, k);
            }
        }
        let cand_fixed = score_pose_candidate(
            world_points,
            image_pixels,
            held,
            (eye0, rot_fixed),
            k,
            size,
            inlier_px,
        );
        if cand_fixed.score.geometry_accepted || cand_fixed.score.inliers + 3 >= best.score.inliers
        {
            return cand_fixed;
        }
    }
    best
}
