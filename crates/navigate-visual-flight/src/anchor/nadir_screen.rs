//! Allocation-free low-AGL nadir seed screening with dead-reckoned search centers and spatial rings.

use super::seeds::{AnchorPrior, SeedPose, push_unique_seed};
use crate::{error::FlightVisualError, nadir::DENSE_NADIR_YAW_BANDS_DEG, surface::Surface};
use nalgebra::Matrix3;
use navigate_locate::corridor::ColdDeadReckonPrior;
use navigate_match::{
    CorridorTileStore, KashinScratch, PreparedKashinQuery, VirtualViewCamera, kashin_screen_stats,
};
use std::cmp::Reverse;

type RankedSeedPose = ((bool, i64), SeedPose);

/// Context for low-AGL (`<= 250 m`) nadir seed screening.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NadirScreenRequest<'a> {
    pub prior: AnchorPrior,
    pub cold_prior: Option<&'a ColdDeadReckonPrior>,
    pub confirmatory_center_enu: Option<[f64; 3]>,
    pub spatial_ring_enabled: bool,
    pub pace_realtime: bool,
    pub coarse_sweep: bool,
    pub post_screen_cap: usize,
}

fn heading_diff_deg(a_deg: f64, b_deg: f64) -> f64 {
    ((a_deg - b_deg + 180.0).rem_euclid(360.0) - 180.0).abs()
}

pub(crate) fn rot_to_row_major(r: &Matrix3<f64>) -> [f64; 9] {
    std::array::from_fn(|i| r[(i / 3, i % 3)])
}

/// Return the `1.0 * agl` nadir seed closest to the reported heading prior when `agl <= 250 m`.
#[must_use]
pub(crate) fn closest_prior_nadir_seed(
    surface: &Surface,
    prior: AnchorPrior,
    cold_prior: Option<&ColdDeadReckonPrior>,
    screened_first: Option<SeedPose>,
) -> Option<SeedPose> {
    let (lat, lon) = prior.lat_lon()?;
    let alt_m = prior.alt_msl_m.filter(|a| a.is_finite())?;
    let dem_h = surface.sample_dem(lat, lon)?;
    let agl = (alt_m - dem_h).max(0.0);
    if agl > 250.0 {
        return None;
    }
    let agl_native = agl.max(25.0);
    if let Some(dir_deg) = prior.finite_heading_deg() {
        let best_band = DENSE_NADIR_YAW_BANDS_DEG
            .into_iter()
            .min_by(|&a, &b| heading_diff_deg(a, dir_deg).total_cmp(&heading_diff_deg(b, dir_deg)))
            .unwrap_or(dir_deg);
        let rot = surface.camera_rotation(lat, lon, best_band, 88.0, 0.0);
        let eye = projected_or_native_eye(surface, (lat, lon, dem_h, agl_native), cold_prior, &rot);
        return Some((eye, rot));
    }
    let rot = screened_first.map_or_else(
        || surface.camera_rotation(lat, lon, 0.0, 88.0, 0.0),
        |(_, r)| r,
    );
    let eye = projected_or_native_eye(surface, (lat, lon, dem_h, agl_native), cold_prior, &rot);
    Some((eye, rot))
}

fn projected_or_native_eye(
    surface: &Surface,
    (lat, lon, dem_h, agl_native): (f64, f64, f64, f64),
    cold_prior: Option<&ColdDeadReckonPrior>,
    crot: &Matrix3<f64>,
) -> [f64; 3] {
    if let Some(cp) = cold_prior {
        let center_shift = (cp.center_enu[0] - cp.anchor_prior_enu[0])
            .hypot(cp.center_enu[1] - cp.anchor_prior_enu[1]);
        let cam_shift = cp.cam_offset_m[0].hypot(cp.cam_offset_m[1]);
        if center_shift > 6.0 || cam_shift > 6.0 {
            let raw = cp.projected_eye_for_rotation(crot);
            let [dlat, dlon, _] = surface.geodetic([raw[0], raw[1], 0.0]);
            let ddem = surface.sample_dem(dlat, dlon).unwrap_or(dem_h);
            return surface.local(dlat, dlon, ddem + agl_native);
        }
    }
    surface.local(lat, lon, dem_h + agl_native)
}

/// Screen low-AGL (`<= 250 m`) nadir center, dead-reckoned, confirmatory, and 8-neighbor ring seeds.
///
/// When `spatial_ring_enabled` is true, this function places up to `6` top center poses and up to
/// `6` spatial-ring poses into `out[0..12]` so capped `LiveFlight` sweeps evaluate both basins.
pub(crate) fn maybe_screen_low_agl_nadir_seeds(
    (surface, store): (&Surface, &CorridorTileStore),
    scratch: &mut KashinScratch,
    req: NadirScreenRequest<'_>,
    prepared: &PreparedKashinQuery,
    intrinsics: [f64; 4],
    render_size: [u32; 2],
) -> Result<Option<Vec<SeedPose>>, FlightVisualError> {
    let (Some((lat, lon)), Some(alt_m)) = (
        req.prior.lat_lon(),
        req.prior.alt_msl_m.filter(|a| a.is_finite()),
    ) else {
        return Ok(None);
    };
    let Some(dem_h) = surface.sample_dem(lat, lon) else {
        return Ok(None);
    };
    let agl = (alt_m - dem_h).max(0.0);
    if agl > 250.0 {
        return Ok(None);
    }

    let (center, top_headings) = screen_center_and_dr_yaws(
        (surface, store),
        scratch,
        &req,
        (lat, lon, dem_h, agl),
        prepared,
        intrinsics,
        render_size,
        req.post_screen_cap > 12,
    )?;

    let cap = req.post_screen_cap.max(1);
    if !req.spatial_ring_enabled {
        let mut out = Vec::with_capacity(center.len().min(cap));
        for pose in center {
            push_unique_seed(&mut out, pose, cap);
        }
        return Ok(Some(out));
    }

    let ring = screen_low_agl_ring_seeds(
        (surface, store),
        scratch,
        &req,
        (lat, lon, dem_h, agl),
        prepared,
        intrinsics,
        render_size,
        &top_headings,
    )?;

    let mut out = Vec::with_capacity((center.len() + ring.len()).min(cap.max(12)));
    let (c6, r6) = (center.len().min(6), ring.len().min(6));
    let (c8, r8, c12) = (center.len().min(8), ring.len().min(8), center.len().min(12));
    for slice in [
        &center[..c6],
        &ring[..r6],
        &center[c6..c8],
        &ring[r6..r8],
        &center[c8..c12],
        &ring[r8..],
        &center[c12..],
    ] {
        for &p in slice {
            push_unique_seed(&mut out, p, cap);
        }
    }
    Ok(Some(out))
}

/// Select spatially and angularly diverse nadir fallback seeds (`center[0]`, `ring[0]`, `ring[1]`, `alt_yaw`).
#[must_use]
pub(crate) fn select_nadir_polish_fallbacks(
    screened: &[SeedPose],
    closest_nadir: Option<SeedPose>,
    max_fallbacks: usize,
) -> Vec<(usize, SeedPose)> {
    let mut out = Vec::with_capacity(max_fallbacks.min(4));
    if max_fallbacks == 0 {
        return out;
    }
    let Some(first) = closest_nadir.or_else(|| screened.first().copied()) else {
        return out;
    };
    out.push((0, first));
    let ring0 = (out.len() < max_fallbacks)
        .then(|| {
            screened
                .iter()
                .copied()
                .enumerate()
                .find(|&(_, (e, _))| (e[0] - first.0[0]).hypot(e[1] - first.0[1]) > 35.0)
        })
        .flatten();
    out.extend(ring0);
    let prev_ring = ring0.map(|(_, (e, _))| e);
    if out.len() < max_fallbacks
        && let Some(r1) = screened.iter().copied().enumerate().find(|&(_, (e, _))| {
            (e[0] - first.0[0]).hypot(e[1] - first.0[1]) > 35.0
                && prev_ring.is_none_or(|pr| (e[0] - pr[0]).hypot(e[1] - pr[1]) > 35.0)
        })
    {
        out.push(r1);
    }
    if out.len() < max_fallbacks
        && let Some(ay) = screened.iter().copied().enumerate().find(|&(_, (e, r))| {
            (e[0] - first.0[0]).hypot(e[1] - first.0[1]) <= 25.0
                && (((r * first.1.transpose()).trace() - 1.0) * 0.5) < 0.90
        })
    {
        out.push(ay);
    }
    out
}

#[expect(
    clippy::too_many_arguments,
    reason = "center and dead-reckoned yaw screening shares surface, tile store, scratch buffers, and camera geometry"
)]
fn screen_center_and_dr_yaws(
    (surface, store): (&Surface, &CorridorTileStore),
    scratch: &mut KashinScratch,
    req: &NadirScreenRequest<'_>,
    (lat, lon, dem_h, agl): (f64, f64, f64, f64),
    prepared: &PreparedKashinQuery,
    intrinsics: [f64; 4],
    render_size: [u32; 2],
    include_elevated: bool,
) -> Result<(Vec<SeedPose>, Vec<f64>), FlightVisualError> {
    let prior_hdg = req.prior.finite_heading_deg();
    let mut ordered_yaws = DENSE_NADIR_YAW_BANDS_DEG;
    if let Some(ph) = prior_hdg {
        ordered_yaws.sort_by(|&a, &b| heading_diff_deg(a, ph).total_cmp(&heading_diff_deg(b, ph)));
    }

    let agl_native = agl.max(25.0);
    let agl_low = (agl * 0.68).clamp(48.0, 78.0);
    let agl_high = (agl * 1.32).clamp(105.0, 155.0);
    let alt_agls: &[f64] = if !include_elevated {
        &[]
    } else if agl >= 80.0 {
        &[agl_low, agl_high]
    } else {
        std::slice::from_ref(&agl_high)
    };
    let native_eye = surface.local(lat, lon, dem_h + agl_native);
    let lm_cap = 384;

    let has_dr = req.cold_prior.is_some_and(|cp| {
        let c_shift = (cp.center_enu[0] - cp.anchor_prior_enu[0])
            .hypot(cp.center_enu[1] - cp.anchor_prior_enu[1]);
        let cam_shift = cp.cam_offset_m[0].hypot(cp.cam_offset_m[1]);
        c_shift > 6.0 || cam_shift > 6.0
    });

    let mut primary_yaws: Vec<((bool, i64), f64, SeedPose)> =
        Vec::with_capacity(ordered_yaws.len() * 3);

    for &hdg in &ordered_yaws {
        let crot = surface.camera_rotation(lat, lon, hdg, 88.0, 0.0);
        let near_prior_hdg = prior_hdg.is_some_and(|ph| heading_diff_deg(hdg, ph) <= 18.0);
        let hdg_bonus = if near_prior_hdg { 50_000_i64 } else { 0 };
        if has_dr {
            let dr_eye = projected_or_native_eye(
                surface,
                (lat, lon, dem_h, agl_native),
                req.cold_prior,
                &crot,
            );
            let dr_ranked = score_nadir_yaw(
                (store, prepared),
                scratch,
                (dr_eye, crot),
                intrinsics,
                render_size,
                (near_prior_hdg, hdg_bonus + 220, lm_cap),
            )?;
            primary_yaws.push((dr_ranked.0, hdg, dr_ranked.1));
        }
        let nat_ranked = score_nadir_yaw(
            (store, prepared),
            scratch,
            (native_eye, crot),
            intrinsics,
            render_size,
            (near_prior_hdg, hdg_bonus + 120, lm_cap),
        )?;
        primary_yaws.push((nat_ranked.0, hdg, nat_ranked.1));

        if let Some(conf_c) = req.confirmatory_center_enu {
            let [clat, clon, _] = surface.geodetic([conf_c[0], conf_c[1], 0.0]);
            let cdem = surface.sample_dem(clat, clon).unwrap_or(dem_h);
            let conf_eye = surface.local(clat, clon, cdem + agl_native);
            if (conf_eye[0] - native_eye[0]).hypot(conf_eye[1] - native_eye[1]) > 12.0 {
                let conf_ranked = score_nadir_yaw(
                    (store, prepared),
                    scratch,
                    (conf_eye, crot),
                    intrinsics,
                    render_size,
                    (near_prior_hdg, hdg_bonus + 250, lm_cap),
                )?;
                primary_yaws.push((conf_ranked.0, hdg, conf_ranked.1));
            }
        }

        for &a_agl in alt_agls.iter().filter(|&&a| (a - agl_native).abs() > 8.0) {
            let a_eye = surface.local(lat, lon, dem_h + a_agl);
            let alt_ranked = score_nadir_yaw(
                (store, prepared),
                scratch,
                (a_eye, crot),
                intrinsics,
                render_size,
                (near_prior_hdg, hdg_bonus, lm_cap),
            )?;
            primary_yaws.push((alt_ranked.0, hdg, alt_ranked.1));
        }
    }

    primary_yaws.sort_by_key(|a| Reverse(a.0));
    let hdg_cap = if req.pace_realtime && !req.coarse_sweep {
        3
    } else {
        6
    };
    let mut top_headings: Vec<f64> = Vec::with_capacity(hdg_cap);
    for &(_, h, _) in &primary_yaws {
        if top_headings.len() >= hdg_cap {
            break;
        }
        if !top_headings
            .iter()
            .any(|&prev| heading_diff_deg(prev, h) < 5.0)
        {
            top_headings.push(h);
        }
    }
    let mut out: Vec<SeedPose> = Vec::with_capacity(28);
    if !req.pace_realtime && include_elevated && agl >= 80.0 {
        let low_ceil = dem_h + agl_low + 4.0;
        for &(_, _, pose) in &primary_yaws {
            push_unique_seed(&mut out, pose, 4);
        }
        for &(_, _, pose) in primary_yaws.iter().filter(|p| p.2.0[2] <= low_ceil) {
            push_unique_seed(&mut out, pose, 6);
        }
    }
    for (_, _, pose) in primary_yaws {
        push_unique_seed(&mut out, pose, 28);
    }
    Ok((out, top_headings))
}

#[expect(
    clippy::too_many_arguments,
    reason = "8-neighbor spatial ring screening shares surface, tile store, scratch buffers, and top headings"
)]
fn screen_low_agl_ring_seeds(
    (surface, store): (&Surface, &CorridorTileStore),
    scratch: &mut KashinScratch,
    req: &NadirScreenRequest<'_>,
    (lat, lon, dem_h, agl): (f64, f64, f64, f64),
    prepared: &PreparedKashinQuery,
    intrinsics: [f64; 4],
    render_size: [u32; 2],
    candidate_headings: &[f64],
) -> Result<Vec<SeedPose>, FlightVisualError> {
    let yaws: &[f64] = if candidate_headings.is_empty() {
        &DENSE_NADIR_YAW_BANDS_DEG
    } else {
        candidate_headings
    };
    let lm_cap = if req.pace_realtime { 256 } else { 384 };
    let native_center = surface.local(lat, lon, 0.0);
    let agl_native = agl.max(25.0);
    let (step_xy, agl_high) = if req.pace_realtime {
        (
            (agl * 0.54).clamp(44.0, 68.0),
            (agl * 0.94).clamp(46.0, 130.0),
        )
    } else {
        (
            (agl * 0.58).clamp(45.0, 75.0),
            (agl * 1.08).clamp(45.0, 140.0),
        )
    };
    let agl_low = (agl * 0.66).clamp(46.0, 78.0);
    let ring_agls: &[f64] =
        if req.pace_realtime && !req.coarse_sweep && (agl_high - agl_low).abs() >= 8.0 {
            &[agl_low, agl_high]
        } else {
            std::slice::from_ref(&agl_high)
        };
    let mut ring_primary: Vec<RankedSeedPose> = Vec::with_capacity(8);
    let mut ring_secondary: Vec<RankedSeedPose> = Vec::with_capacity(8);
    for (gx, gy) in [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ] {
        let mut cell_yaws: Vec<RankedSeedPose> = Vec::with_capacity(yaws.len() * 2);
        for (h_idx, &hdg) in yaws.iter().enumerate() {
            let base_rot = surface.camera_rotation(lat, lon, hdg, 88.0, 0.0);
            let anchor_xy = projected_or_native_eye(
                surface,
                (lat, lon, dem_h, agl_native),
                req.cold_prior,
                &base_rot,
            );
            let has_sep_native = (!req.pace_realtime || (h_idx == 0 && (gx == 0 || gy == 0)))
                && (anchor_xy[0] - native_center[0]).hypot(anchor_xy[1] - native_center[1]) > 25.0;
            for base_xy in std::iter::once(anchor_xy).chain(has_sep_native.then_some(native_center))
            {
                let [glat, glon, _] = surface.geodetic([
                    base_xy[0] + f64::from(gx) * step_xy,
                    base_xy[1] + f64::from(gy) * step_xy,
                    0.0,
                ]);
                let Some(gdem) = surface.sample_dem(glat, glon) else {
                    continue;
                };
                let grot = surface.camera_rotation(glat, glon, hdg, 88.0, 0.0);
                for &r_agl in ring_agls {
                    let geye = surface.local(glat, glon, gdem + r_agl);
                    if let Some(ranked) = score_ring_cell_yaw(
                        (store, prepared),
                        scratch,
                        (geye, grot),
                        intrinsics,
                        render_size,
                        lm_cap,
                    )? {
                        cell_yaws.push(ranked);
                    }
                }
            }
        }
        cell_yaws.sort_by_key(|a| Reverse(a.0));
        let mut it = cell_yaws.into_iter();
        ring_primary.extend(it.next());
        ring_secondary.extend(it.next());
    }
    ring_primary.sort_by_key(|a| Reverse(a.0));
    ring_secondary.sort_by_key(|a| Reverse(a.0));
    ring_primary.extend(ring_secondary);
    Ok(ring_primary.into_iter().map(|(_, pose)| pose).collect())
}

fn score_ring_cell_yaw(
    (store, prepared): (&CorridorTileStore, &PreparedKashinQuery),
    scratch: &mut KashinScratch,
    (geye, grot): SeedPose,
    intrinsics: [f64; 4],
    render_size: [u32; 2],
    lm_cap: usize,
) -> Result<Option<RankedSeedPose>, FlightVisualError> {
    let virt = store.project_virtual_kashin(&VirtualViewCamera {
        eye_enu: geye,
        world_to_cam: rot_to_row_major(&grot),
        intrinsics,
        image_size: render_size,
        max_landmarks: lm_cap,
        min_cos_incidence: 0.45,
        min_cos_orientation: 0.94,
    })?;
    if virt.world_points_enu.len() < 16 {
        return Ok(None);
    }
    let (strict_cnt, smooth) = kashin_screen_stats(scratch, &virt, prepared, lm_cap, 32);
    let rank = (
        strict_cnt >= 8,
        (smooth * 1_000.0).round() as i64 + (strict_cnt as i64) * 150,
    );
    Ok(Some((rank, (geye, grot))))
}

fn score_nadir_yaw(
    (store, prepared): (&CorridorTileStore, &PreparedKashinQuery),
    scratch: &mut KashinScratch,
    (eye, rot): SeedPose,
    intrinsics: [f64; 4],
    render_size: [u32; 2],
    (near_prior_hdg, tie_bonus, lm_cap): (bool, i64, usize),
) -> Result<RankedSeedPose, FlightVisualError> {
    let virt = store.project_virtual_kashin(&VirtualViewCamera {
        eye_enu: eye,
        world_to_cam: rot_to_row_major(&rot),
        intrinsics,
        image_size: render_size,
        max_landmarks: lm_cap,
        min_cos_incidence: 0.45,
        min_cos_orientation: 0.94,
    })?;
    let (strict_cnt, smooth) = kashin_screen_stats(scratch, &virt, prepared, lm_cap, 32);
    let rank = (
        strict_cnt >= 8 || near_prior_hdg,
        tie_bonus + (smooth * 1_000.0).round() as i64 + (strict_cnt as i64) * 150,
    );
    Ok((rank, (eye, rot)))
}
