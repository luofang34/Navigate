//! Low-AGL nadir seed screening and prior-heading nadir pose selection.

use crate::{
    PipelineError,
    corridor::nadir::DENSE_NADIR_YAW_BANDS_DEG,
    realtime::helpers::rot_to_row_major,
    stages::{MatchStageOptions, generate_seeds},
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
type ScreenedCenterWithHeadings = Option<(Vec<SeedPose>, Vec<f64>)>;

pub(super) fn build_anchor_seeds(
    surface: &Surface,
    case: &PhotoCase,
    stage_opts: &MatchStageOptions,
    attitude: Option<(f64, f64)>,
    pred_pose: Option<SeedPose>,
    anchor_attempts: usize,
) -> Vec<SeedPose> {
    pred_pose
        .into_iter()
        .chain(
            generate_seeds(surface, case, stage_opts, attitude)
                .into_iter()
                .take(anchor_attempts.max(1)),
        )
        .collect()
}

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
    (surface, store, _projector): (&Surface, &CorridorTileStore, &HadamardProjector),
    case: &PhotoCase,
    prepared: &PreparedKashinQuery,
    intrinsics: [f64; 4],
    render_size: [u32; 2],
    limit: usize,
) -> Result<Option<Vec<SeedPose>>, PipelineError> {
    let center_only = limit <= 12;
    let Some((center, top_headings)) = maybe_screen_low_agl_center_with_headings(
        (surface, store),
        case,
        prepared,
        intrinsics,
        render_size,
        center_only,
    )?
    else {
        return Ok(None);
    };
    if center_only {
        let mut out = center;
        out.truncate(limit.max(1));
        return Ok(Some(out));
    }
    let cap = limit.max(36);
    let ring = screen_low_agl_ring_seeds(
        (surface, store),
        case,
        prepared,
        intrinsics,
        render_size,
        &top_headings,
    )?;
    let c1 = center.len().min(12);
    let r1 = ring.len().min(12);
    let mut out = Vec::with_capacity((center.len() + ring.len()).min(cap));
    out.extend_from_slice(&center[..c1]);
    out.extend_from_slice(&ring[..r1]);
    out.extend_from_slice(&center[c1..]);
    out.extend_from_slice(&ring[r1..]);
    out.truncate(cap);
    Ok(Some(out))
}

/// Select up to three spatially and angularly diverse nadir fallback seeds (`center[0]`, `ring[0]`, `center[1]`).
pub(super) fn select_nadir_polish_fallbacks(
    screened: &[SeedPose],
    closest_nadir: Option<SeedPose>,
) -> Vec<(usize, SeedPose)> {
    let mut out = Vec::with_capacity(3);
    let Some(first) = closest_nadir.or_else(|| screened.first().copied()) else {
        return out;
    };
    out.push((0, first));
    if let Some((idx, ring_pose)) = screened
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, (e, _))| (e[0] - first.0[0]).hypot(e[1] - first.0[1]) > 35.0)
    {
        out.push((idx, ring_pose));
    }
    if let Some((idx, pose2)) = screened.iter().copied().enumerate().find(|&(_, (e, r))| {
        let near_center = (e[0] - first.0[0]).hypot(e[1] - first.0[1]) <= 35.0;
        let cos_ang = (((r * first.1.transpose()).trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
        near_center && cos_ang < 0.94
    }) {
        out.push((idx, pose2));
    }
    out
}

fn maybe_screen_low_agl_center_with_headings(
    (surface, store): (&Surface, &CorridorTileStore),
    case: &PhotoCase,
    prepared: &PreparedKashinQuery,
    intrinsics: [f64; 4],
    render_size: [u32; 2],
    native_only: bool,
) -> Result<ScreenedCenterWithHeadings, PipelineError> {
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

    let agl_native = agl.max(25.0);
    let agl_elev = (agl * 1.32).clamp(115.0, 155.0);
    let native_eye = surface.local(pos.latitude, pos.longitude, dem_h + agl_native);
    let elev_eye = surface.local(pos.latitude, pos.longitude, dem_h + agl_elev);
    let has_elevated = !native_only && (agl_elev - agl_native).abs() > 8.0;
    let mut out: Vec<SeedPose> = Vec::with_capacity(20);

    let mut native_yaws: Vec<((bool, i64), f64, SeedPose)> = Vec::with_capacity(ordered_yaws.len());
    let mut elevated_yaws: Vec<((bool, i64), f64, SeedPose)> =
        Vec::with_capacity(ordered_yaws.len());
    for &hdg in &ordered_yaws {
        let crot = surface.camera_rotation(pos.latitude, pos.longitude, hdg, 88.0, 0.0);
        let hdg_bonus = prior_hdg.map_or(0_i64, |ph| {
            if heading_diff_deg(hdg, ph) <= 18.0 {
                50_000
            } else {
                0
            }
        });
        let nat_ranked = score_nadir_yaw(
            (store, prepared),
            (native_eye, crot),
            intrinsics,
            render_size,
            hdg_bonus,
        )?;
        native_yaws.push((nat_ranked.0, hdg, nat_ranked.1));
        if has_elevated {
            let elev_ranked = score_nadir_yaw(
                (store, prepared),
                (elev_eye, crot),
                intrinsics,
                render_size,
                hdg_bonus,
            )?;
            elevated_yaws.push((elev_ranked.0, hdg, elev_ranked.1));
        }
    }
    native_yaws.sort_by_key(|a| std::cmp::Reverse(a.0));
    let top_headings: Vec<f64> = native_yaws.iter().take(6).map(|&(_, h, _)| h).collect();
    out.extend(native_yaws.into_iter().map(|(_, _, p)| p));
    if has_elevated {
        elevated_yaws.sort_by_key(|a| std::cmp::Reverse(a.0));
        out.extend(elevated_yaws.into_iter().take(8).map(|(_, _, p)| p));
    }
    Ok(Some((out, top_headings)))
}

fn screen_low_agl_ring_seeds(
    (surface, store): (&Surface, &CorridorTileStore),
    case: &PhotoCase,
    prepared: &PreparedKashinQuery,
    intrinsics: [f64; 4],
    render_size: [u32; 2],
    candidate_headings: &[f64],
) -> Result<Vec<SeedPose>, PipelineError> {
    let Some(pos) = case.prior.as_ref().and_then(|p| p.position.as_ref()) else {
        return Ok(Vec::new());
    };
    let (Some(alt_m), Some(dem_h)) = (
        pos.altitude_m,
        surface.sample_dem(pos.latitude, pos.longitude),
    ) else {
        return Ok(Vec::new());
    };
    let agl = (alt_m - dem_h).max(0.0);
    if agl > 250.0 {
        return Ok(Vec::new());
    }
    let yaws: &[f64] = if candidate_headings.is_empty() {
        &DENSE_NADIR_YAW_BANDS_DEG
    } else {
        candidate_headings
    };
    let center = surface.local(pos.latitude, pos.longitude, 0.0);
    let step_xy = (agl * 0.58).clamp(45.0, 75.0);
    let agl_ring = (agl * 1.08).clamp(45.0, 140.0);
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
        let mut cell_yaws: Vec<RankedSeedPose> = Vec::with_capacity(yaws.len());
        for &hdg in yaws {
            let grot = surface.camera_rotation(glat, glon, hdg, 88.0, 0.0);
            let virt = store.project_virtual_kashin(&VirtualViewCamera {
                eye_enu: geye,
                world_to_cam: rot_to_row_major(&grot),
                intrinsics,
                image_size: render_size,
                max_landmarks: 384,
                min_cos_incidence: 0.45,
                min_cos_orientation: 0.94,
            })?;
            if virt.world_points_enu.len() < 16 {
                continue;
            }
            let (strict_cnt, smooth) = kashin_screen_stats(&virt, prepared, 384, 32);
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
    Ok(ring_scored.into_iter().map(|(_, pose)| pose).collect())
}

fn score_nadir_yaw(
    (store, prepared): (&CorridorTileStore, &PreparedKashinQuery),
    (eye, rot): SeedPose,
    intrinsics: [f64; 4],
    render_size: [u32; 2],
    hdg_bonus: i64,
) -> Result<RankedSeedPose, PipelineError> {
    let virt = store.project_virtual_kashin(&VirtualViewCamera {
        eye_enu: eye,
        world_to_cam: rot_to_row_major(&rot),
        intrinsics,
        image_size: render_size,
        max_landmarks: 384,
        min_cos_incidence: 0.45,
        min_cos_orientation: 0.94,
    })?;
    let (strict_cnt, smooth) = kashin_screen_stats(&virt, prepared, 384, 32);
    let rank = (
        strict_cnt >= 8 || hdg_bonus > 0,
        hdg_bonus + (smooth * 1_000.0).round() as i64 + (strict_cnt as i64) * 150,
    );
    Ok((rank, (eye, rot)))
}

fn decode_unit_kashin(code: &[u8; 64]) -> [f32; 64] {
    let mut norm_sq = 0_i32;
    for &raw in code {
        let vi = i32::from(raw as i8);
        norm_sq += vi * vi;
    }
    if norm_sq <= 0 {
        return [0.0; 64];
    }
    let inv = 1.0 / (norm_sq as f32).sqrt();
    let mut out = [0.0_f32; 64];
    for (dst, &raw) in out.iter_mut().zip(code) {
        *dst = f32::from(raw as i8) * inv;
    }
    out
}

fn kashin_screen_stats(
    virt: &VirtualCorridorView,
    query: &PreparedKashinQuery,
    cap: usize,
    top_k: usize,
) -> (usize, f64) {
    let n_r = virt.sign_bits.len().min(cap);
    let n_q = query.valid_indices().len().min(cap);
    if n_r == 0 || n_q == 0 {
        return (0, 0.0);
    }
    let q_signs = query.sign_bits();
    let q_rots = query.rotated();
    let mut smooth_fwd = vec![(usize::MAX, -2.0_f32); n_r];
    let mut smooth_rev = vec![(usize::MAX, -2.0_f32); n_q];
    let mut strict_fwd = vec![(usize::MAX, -2.0_f32, -2.0_f32); n_r];
    let mut strict_rev = vec![(usize::MAX, -2.0_f32); n_q];
    for r in 0..n_r {
        let rs = virt.sign_bits[r];
        let ru = decode_unit_kashin(&virt.kashin_i8[r]);
        let s_slot = &mut smooth_fwd[r];
        let st_slot = &mut strict_fwd[r];
        for q in 0..n_q {
            let ham = (rs ^ q_signs[q]).count_ones();
            if ham > 25 {
                continue;
            }
            let dot: f32 = ru.iter().zip(&q_rots[q]).map(|(&a, &b)| a * b).sum();
            if dot > s_slot.1 {
                *s_slot = (q, dot);
            }
            if dot > smooth_rev[q].1 {
                smooth_rev[q] = (r, dot);
            }
            if ham <= 24 {
                if dot > st_slot.1 {
                    *st_slot = (q, dot, st_slot.1);
                } else if dot > st_slot.2 {
                    st_slot.2 = dot;
                }
                if dot > strict_rev[q].1 {
                    strict_rev[q] = (r, dot);
                }
            }
        }
    }
    let strict_cnt = strict_fwd
        .iter()
        .enumerate()
        .filter(|&(r, &(q, b1, b2))| {
            q < n_q
                && strict_rev[q].0 == r
                && b1 >= 0.62
                && (b2 <= -1.5 || (1.0 - b1).max(0.0) <= 0.85 * (1.0 - b2).max(0.0))
        })
        .count();
    let mut dots: Vec<f32> = smooth_fwd
        .into_iter()
        .enumerate()
        .filter_map(|(r, (q, d))| (q < n_q && smooth_rev[q].0 == r && d >= 0.42).then_some(d))
        .collect();
    dots.sort_by(|a, b| b.total_cmp(a));
    (
        strict_cnt,
        dots.into_iter().take(top_k).map(f64::from).sum(),
    )
}
