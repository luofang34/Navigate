//! Inter-frame rotation RANSAC and ground visual-odometry translation helpers.

use crate::corridor::kinematics::VehicleDynamicsProfile;
use nalgebra::{Matrix3, Vector3};
use navigate_match::PreparedKashinQuery;

/// Solve the orthogonal Procrustes rotation `R` (`b_i ≈ R * a_i`) with `3 x 3` SVD.
#[must_use]
pub fn kabsch_3d(rays_a: &[[f64; 3]], rays_b: &[[f64; 3]]) -> Option<Matrix3<f64>> {
    if rays_a.len() != rays_b.len() || rays_a.len() < 2 {
        return None;
    }
    let mut h = Matrix3::<f64>::zeros();
    for (a, b) in rays_a.iter().zip(rays_b) {
        let va = Vector3::from(*a);
        let vb = Vector3::from(*b);
        let na = va.norm();
        let nb = vb.norm();
        if na > 1e-9 && nb > 1e-9 {
            h += (vb / nb) * (va / na).transpose();
        }
    }
    let svd = h.svd(true, true);
    let (u, vt) = (svd.u?, svd.v_t?);
    let mut r = u * vt;
    if r.determinant() < 0.0 {
        let mut u_fix = u;
        u_fix.set_column(2, &(-u.column(2)));
        r = u_fix * vt;
    }
    Some(r)
}

/// Estimate relative camera rotation `R_ba` between calibrated pixel pairs with RANSAC.
#[must_use]
pub fn relative_rotation_ransac(
    px_a: &[[f64; 2]],
    px_b: &[[f64; 2]],
    k: &Matrix3<f64>,
    threshold_px: f64,
) -> Option<(Matrix3<f64>, Vec<bool>)> {
    if px_a.len() != px_b.len() || px_a.len() < 4 {
        return None;
    }
    let fx = k[(0, 0)].max(1.0);
    let fy = k[(1, 1)].max(1.0);
    let cx = k[(0, 2)];
    let cy = k[(1, 2)];
    let rays_a: Vec<[f64; 3]> = px_a
        .iter()
        .map(|p| {
            let v = Vector3::new((p[0] - cx) / fx, (p[1] - cy) / fy, 1.0).normalize();
            [v.x, v.y, v.z]
        })
        .collect();
    let rays_b: Vec<[f64; 3]> = px_b
        .iter()
        .map(|p| {
            let v = Vector3::new((p[0] - cx) / fx, (p[1] - cy) / fy, 1.0).normalize();
            [v.x, v.y, v.z]
        })
        .collect();

    let max_angle = (threshold_px / fx.min(fy)).atan();
    let cos_thresh = max_angle.cos();
    let mut best_r = Matrix3::identity();
    let mut best_count = 0usize;

    let trials = 48usize.min(rays_a.len() * 2);
    for t in 0..trials {
        let i0 = (t * 7) % rays_a.len();
        let i1 = (t * 13 + 1) % rays_a.len();
        let i2 = (t * 29 + 2) % rays_a.len();
        if i0 == i1 || i1 == i2 || i0 == i2 {
            continue;
        }
        let sub_a = [rays_a[i0], rays_a[i1], rays_a[i2]];
        let sub_b = [rays_b[i0], rays_b[i1], rays_b[i2]];
        let Some(r) = kabsch_3d(&sub_a, &sub_b) else {
            continue;
        };
        let count = rays_a
            .iter()
            .zip(&rays_b)
            .filter(|(a, b)| (r * Vector3::from(**a)).dot(&Vector3::from(**b)) >= cos_thresh)
            .count();
        if count > best_count {
            best_count = count;
            best_r = r;
        }
    }
    if best_count < 3 {
        return None;
    }
    let inliers: Vec<bool> = rays_a
        .iter()
        .zip(&rays_b)
        .map(|(a, b)| (best_r * Vector3::from(*a)).dot(&Vector3::from(*b)) >= cos_thresh)
        .collect();
    let in_a: Vec<[f64; 3]> = rays_a
        .iter()
        .zip(&inliers)
        .filter_map(|(a, &ok)| ok.then_some(*a))
        .collect();
    let in_b: Vec<[f64; 3]> = rays_b
        .iter()
        .zip(&inliers)
        .filter_map(|(b, &ok)| ok.then_some(*b))
        .collect();
    let refined = kabsch_3d(&in_a, &in_b).unwrap_or(best_r);
    Some((refined, inliers))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn estimate_ground_vo_delta(
    px_a: &[[f64; 2]],
    px_b: &[[f64; 2]],
    rot_step: &Matrix3<f64>,
    world_to_cam: &Matrix3<f64>,
    k: &Matrix3<f64>,
    agl_m: f64,
    dt_s: f64,
    profile: &VehicleDynamicsProfile,
) -> [f64; 3] {
    if px_a.len() < 8 || !agl_m.is_finite() || agl_m < 15.0 {
        return [0.0; 3];
    }
    let fx = k[(0, 0)].max(1e-6);
    let fy = k[(1, 1)].max(1e-6);
    let cx = k[(0, 2)];
    let cy = k[(1, 2)];
    let cam_to_world = world_to_cam.transpose();
    let mut east_samples = Vec::with_capacity(px_a.len());
    let mut north_samples = Vec::with_capacity(px_a.len());
    let (filter_gate, clamp_gate) = profile.ground_vo_bounds(dt_s);

    for (&[ax, ay], &[bx, by]) in px_a.iter().zip(px_b.iter()) {
        let ray_a = Vector3::new((ax - cx) / fx, (ay - cy) / fy, 1.0);
        let dir_world = cam_to_world * (ray_a / ray_a.norm().max(1e-9));
        if dir_world.z >= -0.15 {
            continue;
        }
        let rot_ray = rot_step * ray_a;
        if rot_ray.z <= 1e-3 {
            continue;
        }
        let ax_derot = fx * (rot_ray.x / rot_ray.z) + cx;
        let ay_derot = fy * (rot_ray.y / rot_ray.z) + cy;
        let slant_m = (agl_m / (-dir_world.z)).clamp(15.0, 8_000.0);
        let z_cam = slant_m / ray_a.norm().max(1.0);
        let d_cam = Vector3::new(
            -z_cam * (bx - ax_derot) / fx,
            -z_cam * (by - ay_derot) / fy,
            0.0,
        );
        let d_enu = cam_to_world * d_cam;
        if d_enu.x.hypot(d_enu.y) <= filter_gate {
            east_samples.push(d_enu.x);
            north_samples.push(d_enu.y);
        }
    }
    if east_samples.len() < 6 {
        return [0.0; 3];
    }
    east_samples.sort_by(f64::total_cmp);
    north_samples.sort_by(f64::total_cmp);
    let mid = east_samples.len() / 2;
    [
        east_samples[mid].clamp(-clamp_gate, clamp_gate),
        north_samples[mid].clamp(-clamp_gate, clamp_gate),
        0.0,
    ]
}

pub(super) fn match_interframe_top(
    prev_q: &PreparedKashinQuery,
    cur_q: &PreparedKashinQuery,
    cap: usize,
    max_pixel_disp: f64,
) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let na = prev_q.valid_indices().len().min(cap);
    let nb = cur_q.valid_indices().len().min(cap);
    let (sa, sb) = (prev_q.sign_bits(), cur_q.sign_bits());
    let (pa, pb) = (prev_q.pixels(), cur_q.pixels());
    let (ia, ib) = (prev_q.valid_indices(), cur_q.valid_indices());
    let mut out_a = Vec::new();
    let mut out_b = Vec::new();
    for a in 0..na {
        let [ax, ay] = pa[ia[a]];
        let mut best = (usize::MAX, 65_u32, 65_u32);
        for b in 0..nb {
            let [bx, by] = pb[ib[b]];
            if (ax - bx).abs() > max_pixel_disp || (ay - by).abs() > max_pixel_disp {
                continue;
            }
            let hd = (sa[a] ^ sb[b]).count_ones();
            if hd < best.1 {
                best = (b, hd, best.1);
            } else if hd < best.2 {
                best.2 = hd;
            }
        }
        if best.0 != usize::MAX && best.1 <= 14 && best.1 + 3 <= best.2 {
            out_a.push([ax, ay]);
            out_b.push(pb[ib[best.0]]);
        }
    }
    (out_a, out_b)
}
