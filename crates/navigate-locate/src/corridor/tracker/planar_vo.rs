//! Planar `SE(2)` optical-axis yaw RANSAC and warm 3D landmark tracking helpers.

use crate::corridor::kinematics::{TrackingSearchStrategy, match_tracks_kashin};
use nalgebra::Matrix3;
use navigate_match::{ActiveTrackSet, MatchError, PreparedKashinQuery, TrackMatchConfig};
use navigate_visual::holdout_pnp::{PoseCandidate, fit_pose_hybrid, fit_pose_l1_gnc};

/// Diagnostic summary of one warm 3D landmark solve attempt.
#[derive(Clone, Debug)]
pub struct WarmTrackAttempt {
    /// Matching and PnP solver label.
    pub method: &'static str,
    /// Number of 3D-2D correspondences fed into PnP.
    pub matched_landmarks: usize,
    /// Unfiltered PnP candidate before inlier threshold gating.
    pub candidate: Option<PoseCandidate>,
}

/// Match `tracks` against `prepared` and solve PnP around `pred`.
///
/// # Errors
/// Returns [`MatchError`] when radius landmark projection or matching fails.
pub fn try_warm_track(
    tracks: &ActiveTrackSet,
    prepared: &PreparedKashinQuery,
    pred: ([f64; 3], Matrix3<f64>),
    k_render: &Matrix3<f64>,
    render_size: [u32; 2],
    min_inliers: usize,
    strategy: TrackingSearchStrategy,
) -> Result<(WarmTrackAttempt, Option<PoseCandidate>), MatchError> {
    let (world, px) = match strategy {
        TrackingSearchStrategy::RadiusSearch { radius_px } => {
            let cfg = TrackMatchConfig {
                search_radius_px: radius_px as f32,
                hamming_gate: 22,
                min_dot: 0.64,
                ratio_limit: 0.90,
            };
            let intrinsics = [
                k_render[(0, 0)],
                k_render[(1, 1)],
                k_render[(0, 2)],
                k_render[(1, 2)],
            ];
            let r = &pred.1;
            let row_major = [
                r[(0, 0)],
                r[(0, 1)],
                r[(0, 2)],
                r[(1, 0)],
                r[(1, 1)],
                r[(1, 2)],
                r[(2, 0)],
                r[(2, 1)],
                r[(2, 2)],
            ];
            let matches =
                tracks.match_radius(prepared, pred.0, &row_major, intrinsics, render_size, &cfg)?;
            let mut w = Vec::with_capacity(matches.len());
            let mut p = Vec::with_capacity(matches.len());
            for m in &matches {
                w.push(m.world_point_enu);
                p.push(m.query_pixel);
            }
            (w, p)
        }
        TrackingSearchStrategy::GlobalFrustumMatch => match_tracks_kashin(tracks, prepared, 384),
    };
    let mut cand = fit_pose_l1_gnc(&world, &px, k_render, render_size, pred);
    if !cand.as_ref().is_some_and(|c| c.score.geometry_accepted) && world.len() >= 16 {
        let fb = fit_pose_hybrid(&world, &px, k_render, render_size, None)
            .or_else(|| fit_pose_hybrid(&world, &px, k_render, render_size, Some(pred)));
        if let Some(fb_cand) = fb
            && cand.as_ref().is_none_or(|prev| {
                (fb_cand.score.geometry_accepted && !prev.score.geometry_accepted)
                    || fb_cand.score.inliers > prev.score.inliers
            })
        {
            cand = Some(fb_cand);
        }
    }
    let method = match strategy {
        TrackingSearchStrategy::RadiusSearch { .. } => "kashin-3d-radius-track+l1-gnc",
        TrackingSearchStrategy::GlobalFrustumMatch => "kashin-3d-global-frustum+l1-gnc",
    };
    let accepted = cand
        .clone()
        .filter(|c| c.score.geometry_accepted && c.score.inliers >= min_inliers);
    let attempt = WarmTrackAttempt {
        method,
        matched_landmarks: world.len(),
        candidate: cand,
    };
    Ok((attempt, accepted))
}

/// Estimate pure optical-axis camera rotation `R_z(dpsi)` decoupled from 2D planar translation.
///
/// On steep-nadir views, full 3D rotation RANSAC conflates planar ground translation with small
/// camera pitch and roll rotations. Fitting a 2D rigid model `(dpsi, tx, ty)` in normalized image
/// coordinates isolates true yaw rotation `R_z(dpsi)` while preserving translational pixel flow.
#[must_use]
pub fn nadir_yaw_translation_ransac(
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
    let f_mean = 0.5 * (fx + fy);
    let thresh_sq = (threshold_px / f_mean).powi(2);
    let scale_tol = 2.0 * (threshold_px / f_mean);

    let norm_a: Vec<[f64; 2]> = px_a
        .iter()
        .map(|p| [(p[0] - cx) / fx, (p[1] - cy) / fy])
        .collect();
    let norm_b: Vec<[f64; 2]> = px_b
        .iter()
        .map(|p| [(p[0] - cx) / fx, (p[1] - cy) / fy])
        .collect();

    let mut best_model = (1.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut best_count = 0_usize;
    let n = norm_a.len();
    let trials = 48_usize.min(n * 2);

    for t in 0..trials {
        let i0 = (t * 7) % n;
        let i1 = (t * 13 + 3) % n;
        if i0 == i1 {
            continue;
        }
        let da = [norm_a[i1][0] - norm_a[i0][0], norm_a[i1][1] - norm_a[i0][1]];
        let db = [norm_b[i1][0] - norm_b[i0][0], norm_b[i1][1] - norm_b[i0][1]];
        let la = da[0].hypot(da[1]);
        let lb = db[0].hypot(db[1]);
        if la < 0.06 || lb < 0.06 || (la - lb).abs() > scale_tol {
            continue;
        }
        let denom = la * lb;
        let cos_t = ((da[0] * db[0] + da[1] * db[1]) / denom).clamp(-1.0, 1.0);
        let sin_t = ((da[0] * db[1] - da[1] * db[0]) / denom).clamp(-1.0, 1.0);
        let tx0 = norm_b[i0][0] - (cos_t * norm_a[i0][0] - sin_t * norm_a[i0][1]);
        let ty0 = norm_b[i0][1] - (sin_t * norm_a[i0][0] + cos_t * norm_a[i0][1]);
        let tx1 = norm_b[i1][0] - (cos_t * norm_a[i1][0] - sin_t * norm_a[i1][1]);
        let ty1 = norm_b[i1][1] - (sin_t * norm_a[i1][0] + cos_t * norm_a[i1][1]);
        let (tx, ty) = (0.5 * (tx0 + tx1), 0.5 * (ty0 + ty1));

        let count = norm_a
            .iter()
            .zip(&norm_b)
            .filter(|(a, b)| {
                let rx = cos_t * a[0] - sin_t * a[1] + tx - b[0];
                let ry = sin_t * a[0] + cos_t * a[1] + ty - b[1];
                rx * rx + ry * ry <= thresh_sq
            })
            .count();
        if count > best_count {
            best_count = count;
            best_model = (cos_t, sin_t, tx, ty);
        }
    }

    if best_count < 4 {
        return None;
    }

    let (c0, s0, tx0, ty0) = best_model;
    let inliers: Vec<bool> = norm_a
        .iter()
        .zip(&norm_b)
        .map(|(a, b)| {
            let rx = c0 * a[0] - s0 * a[1] + tx0 - b[0];
            let ry = s0 * a[0] + c0 * a[1] + ty0 - b[1];
            rx * rx + ry * ry <= thresh_sq
        })
        .collect();

    let (mut ca_x, mut ca_y, mut cb_x, mut cb_y, mut cnt) = (0.0, 0.0, 0.0, 0.0, 0_usize);
    for ((a, b), &ok) in norm_a.iter().zip(&norm_b).zip(&inliers) {
        if ok {
            ca_x += a[0];
            ca_y += a[1];
            cb_x += b[0];
            cb_y += b[1];
            cnt += 1;
        }
    }
    if cnt < 3 {
        return None;
    }
    let inv = 1.0 / (cnt as f64);
    let (ca_x, ca_y, cb_x, cb_y) = (ca_x * inv, ca_y * inv, cb_x * inv, cb_y * inv);

    let (mut s_dot, mut s_cross) = (0.0_f64, 0.0_f64);
    for ((a, b), &ok) in norm_a.iter().zip(&norm_b).zip(&inliers) {
        if ok {
            let (ax, ay) = (a[0] - ca_x, a[1] - ca_y);
            let (bx, by) = (b[0] - cb_x, b[1] - cb_y);
            s_dot += ax * bx + ay * by;
            s_cross += ax * by - ay * bx;
        }
    }
    let mag = s_dot.hypot(s_cross);
    let (cos_f, sin_f) = if mag > 1e-12 {
        (s_dot / mag, s_cross / mag)
    } else {
        (c0, s0)
    };
    let rot_z = Matrix3::new(cos_f, -sin_f, 0.0, sin_f, cos_f, 0.0, 0.0, 0.0, 1.0);
    Some((rot_z, inliers))
}
