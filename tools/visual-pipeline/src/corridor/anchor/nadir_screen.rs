//! Low-AGL nadir seed screening and prior-heading nadir pose selection.

use crate::{
    PipelineError,
    corridor::{
        kinematics::{approx_dot, match_virtual_kashin},
        nadir::DENSE_NADIR_YAW_BANDS_DEG,
    },
    realtime::helpers::rot_to_row_major,
    surface::Surface,
    testset::PhotoCase,
};
use nalgebra::Matrix3;
use navigate_match::{
    CorridorTileStore, HadamardProjector, PreparedKashinQuery, VirtualCorridorView,
    VirtualViewCamera,
};

pub(super) type SeedPose = ([f64; 3], Matrix3<f64>);
type RankedSeedPose = ((bool, i64), SeedPose);

fn heading_diff_deg(a_deg: f64, b_deg: f64) -> f64 {
    ((a_deg - b_deg + 180.0).rem_euclid(360.0) - 180.0).abs()
}

/// Return the `1.0 * agl` nadir seed closest to the reported heading prior when `agl <= 250 m`.
pub(super) fn closest_prior_nadir_seed(
    surface: &Surface,
    case: &PhotoCase,
    screened_first: Option<SeedPose>,
) -> Option<SeedPose> {
    let pos = case.prior.as_ref().and_then(|p| p.position.as_ref())?;
    let alt_m = pos.altitude_m?;
    let dem_h = surface.sample_dem(pos.latitude, pos.longitude)?;
    let agl = (alt_m - dem_h).max(0.0);
    if agl > 250.0 {
        return None;
    }
    let native_eye = surface.local(pos.latitude, pos.longitude, dem_h + agl.max(25.0));
    if let Some(dir_deg) = case
        .prior
        .as_ref()
        .and_then(|p| p.direction_deg)
        .filter(|d| d.is_finite())
    {
        let best_band = DENSE_NADIR_YAW_BANDS_DEG
            .into_iter()
            .min_by(|&a, &b| heading_diff_deg(a, dir_deg).total_cmp(&heading_diff_deg(b, dir_deg)))
            .unwrap_or(dir_deg.rem_euclid(360.0));
        return Some((
            native_eye,
            surface.camera_rotation(pos.latitude, pos.longitude, best_band, 88.0, 0.0),
        ));
    }
    screened_first.or_else(|| {
        Some((
            native_eye,
            surface.camera_rotation(pos.latitude, pos.longitude, 0.0, 88.0, 0.0),
        ))
    })
}

pub(super) fn maybe_screen_low_agl_nadir_seeds(
    (surface, store, projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    case: &PhotoCase,
    prepared: &PreparedKashinQuery,
    intrinsics: [f64; 4],
    render_size: [u32; 2],
    limit: usize,
) -> Result<Option<Vec<SeedPose>>, PipelineError> {
    let Some(pos) = case.prior.as_ref().and_then(|p| p.position.as_ref()) else {
        return Ok(None);
    };
    let Some(alt_m) = pos.altitude_m else {
        return Ok(None);
    };
    let Some(dem_h) = surface.sample_dem(pos.latitude, pos.longitude) else {
        return Ok(None);
    };
    let agl = (alt_m - dem_h).max(0.0);
    if agl > 250.0 {
        return Ok(None);
    }
    let prior_hdg = case
        .prior
        .as_ref()
        .and_then(|p| p.direction_deg)
        .filter(|d| d.is_finite())
        .map(|d| d.rem_euclid(360.0));
    let mut ordered_yaws = DENSE_NADIR_YAW_BANDS_DEG;
    if let Some(ph) = prior_hdg {
        ordered_yaws.sort_by(|&a, &b| heading_diff_deg(a, ph).total_cmp(&heading_diff_deg(b, ph)));
    }

    let center = surface.local(pos.latitude, pos.longitude, 0.0);
    let agl_native = agl.max(25.0);
    let agl_center = (agl * 1.50).clamp(130.0, 175.0);
    let native_eye = surface.local(pos.latitude, pos.longitude, dem_h + agl_native);
    let center_eye = surface.local(pos.latitude, pos.longitude, dem_h + agl_center);
    let cap = limit.max(36);
    let mut out: Vec<SeedPose> = Vec::with_capacity(cap);

    let mut center_candidates: Vec<(RankedSeedPose, bool)> = Vec::with_capacity(24);
    let mut center_eyes = vec![(native_eye, true)];
    if (agl_center - agl_native).abs() > 8.0 {
        center_eyes.push((center_eye, false));
    }
    for (c_eye, is_native) in center_eyes {
        for &hdg in &ordered_yaws {
            let crot = surface.camera_rotation(pos.latitude, pos.longitude, hdg, 88.0, 0.0);
            let virt = store.project_virtual_view(
                &VirtualViewCamera {
                    eye_enu: c_eye,
                    world_to_cam: rot_to_row_major(&crot),
                    intrinsics,
                    image_size: render_size,
                    max_landmarks: 512,
                    min_cos_incidence: 0.45,
                    min_cos_orientation: 0.94,
                },
                projector,
            )?;
            let strict_cnt = match_virtual_kashin(&virt, prepared, 512).0.len();
            let smooth = kashin_mutual_top_sum(&virt, prepared, 512, 32);
            let hdg_bonus = prior_hdg.map_or(0_i64, |ph| {
                if heading_diff_deg(hdg, ph) <= 18.0 {
                    50_000
                } else {
                    0
                }
            });
            let rank = (
                strict_cnt >= 8 || hdg_bonus > 0,
                hdg_bonus + (smooth * 1_000.0).round() as i64 + (strict_cnt as i64) * 150,
            );
            center_candidates.push(((rank, (c_eye, crot)), is_native));
        }
    }
    center_candidates.sort_by_key(|a| std::cmp::Reverse((a.0).0));
    // Keep top-ranked 1.0 * agl and 1.5 * agl center poses (up to 20 total).
    let mut native_kept = 0_usize;
    let mut elevated_kept = 0_usize;
    for ((_, pose), is_native) in center_candidates {
        if is_native && native_kept < 12 {
            out.push(pose);
            native_kept += 1;
        } else if !is_native && elevated_kept < 12 {
            out.push(pose);
            elevated_kept += 1;
        }
    }

    let step_xy = (agl * 0.70).clamp(58.0, 85.0);
    let agl_ring = (agl * 1.32).clamp(115.0, 155.0);
    let ring_offsets = [
        (1_i32, 0_i32),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ];
    let mut ring_scored: Vec<RankedSeedPose> = Vec::with_capacity(16);
    for (gx, gy) in ring_offsets {
        let [glat, glon, _] = surface.geodetic([
            center[0] + f64::from(gx) * step_xy,
            center[1] + f64::from(gy) * step_xy,
            0.0,
        ]);
        let Some(gdem) = surface.sample_dem(glat, glon) else {
            continue;
        };
        let geye = surface.local(glat, glon, gdem + agl_ring);
        let mut cell_yaws: Vec<RankedSeedPose> = Vec::with_capacity(12);
        for &hdg in &ordered_yaws {
            let grot = surface.camera_rotation(glat, glon, hdg, 88.0, 0.0);
            let virt = store.project_virtual_view(
                &VirtualViewCamera {
                    eye_enu: geye,
                    world_to_cam: rot_to_row_major(&grot),
                    intrinsics,
                    image_size: render_size,
                    max_landmarks: 512,
                    min_cos_incidence: 0.45,
                    min_cos_orientation: 0.94,
                },
                projector,
            )?;
            if virt.world_points_enu.len() < 16 {
                continue;
            }
            let strict_cnt = match_virtual_kashin(&virt, prepared, 512).0.len();
            let smooth = kashin_mutual_top_sum(&virt, prepared, 512, 32);
            let rank = (
                strict_cnt >= 8,
                (smooth * 1_000.0).round() as i64 + (strict_cnt as i64) * 150,
            );
            cell_yaws.push((rank, (geye, grot)));
        }
        cell_yaws.sort_by_key(|a| std::cmp::Reverse(a.0));
        ring_scored.extend(cell_yaws.into_iter().take(2));
    }
    ring_scored.sort_by_key(|a| std::cmp::Reverse(a.0));
    out.extend(
        ring_scored
            .into_iter()
            .take(cap.saturating_sub(out.len()))
            .map(|(_, pose)| pose),
    );
    out.truncate(cap);
    Ok(Some(out))
}

fn kashin_mutual_top_sum(
    virt: &VirtualCorridorView,
    query: &PreparedKashinQuery,
    cap: usize,
    top_k: usize,
) -> f64 {
    let n_r = virt.sign_bits.len().min(cap);
    let n_q = query.valid_indices().len().min(cap);
    if n_r == 0 || n_q == 0 {
        return 0.0;
    }
    let q_signs = query.sign_bits();
    let q_rots = query.rotated();
    let mut fwd = vec![(usize::MAX, -2.0_f32); n_r];
    let mut rev = vec![(usize::MAX, -2.0_f32); n_q];
    for (r, slot) in fwd.iter_mut().enumerate() {
        let rs = virt.sign_bits[r];
        let rc = &virt.kashin_i8[r];
        for q in 0..n_q {
            if (rs ^ q_signs[q]).count_ones() > 26 {
                continue;
            }
            let dot = approx_dot(rc, &q_rots[q]);
            if dot > slot.1 {
                *slot = (q, dot);
            }
            if dot > rev[q].1 {
                rev[q] = (r, dot);
            }
        }
    }
    let mut dots: Vec<f32> = fwd
        .into_iter()
        .enumerate()
        .filter_map(|(r, (q, d))| (q < n_q && rev[q].0 == r && d >= 0.42).then_some(d))
        .collect();
    dots.sort_by(|a, b| b.total_cmp(a));
    dots.into_iter().take(top_k).map(f64::from).sum()
}
