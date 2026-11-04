//! Foreground 3D landmark seeding, frustum top-up, and reprojection-gated anchor merging.

use crate::{anchor::nadir_screen::rot_to_row_major, surface::Surface};
use nalgebra::{Matrix3, Vector3};
use navigate_match::{
    ActiveTrackSet, CorridorTileStore, PreparedKashinQuery, SparseFeatures, VirtualCorridorView,
    VirtualViewCamera,
};
use navigate_visual::PoseCandidate;
use std::collections::HashMap;

/// Frustum retention margin in pixels during warm-track top-up.
pub const TOP_UP_FRUSTUM_MARGIN_PX: f64 = 48.0;

/// Frustum retention margin in pixels when staging background anchor tracks.
pub const ARRIVAL_FRUSTUM_MARGIN_PX: f64 = 64.0;

const TOP_UP_DEDUP_METERS: f64 = 1.5;
const FG_MERGE_MAX_DISCREPANCY_PX: f64 = 4.0;
const FG_MERGE_CAP: usize = 768;

/// Seed a 3D [`ActiveTrackSet`] from raycast query features and optional virtual-view landmarks.
#[must_use]
pub fn seed_tracks(
    surface: &Surface,
    prepared: &PreparedKashinQuery,
    qry_feat: &SparseFeatures,
    cand: &PoseCandidate,
    virt_opt: &Option<VirtualCorridorView>,
    was_polished: bool,
) -> ActiveTrackSet {
    let seed_cap = qry_feat.pixels.len().min(1024);
    let hits = surface.raycast_pixels(
        cand.eye,
        &cand.rotation,
        &cand.k,
        &qry_feat.pixels[..seed_cap],
    );
    let world_opts: Vec<Option<[f64; 3]>> =
        hits.into_iter().map(|h| h.map(|x| x.world_enu)).collect();
    let mut set = ActiveTrackSet::from_prepared_query(prepared, &world_opts, 1024);
    if let Some(virt) = virt_opt {
        let virt_cap = if was_polished || cand.score.holdout_median_px <= 4.5 {
            2048
        } else {
            1024
        };
        set.extend_capped(
            &virt.world_points_enu,
            &virt.sign_bits,
            &virt.kashin_i8,
            virt_cap,
        );
    }
    set
}

/// Top up `tracks` with terrain raycasts at `cand` and visible corridor bank landmarks.
pub fn top_up_tracks_with_bank(
    tracks: &mut ActiveTrackSet,
    surface: &Surface,
    store: &CorridorTileStore,
    prepared: &PreparedKashinQuery,
    qry_feat: &SparseFeatures,
    cand: &PoseCandidate,
    render_size: [u32; 2],
) {
    let intrinsics = [
        cand.k[(0, 0)],
        cand.k[(1, 1)],
        cand.k[(0, 2)],
        cand.k[(1, 2)],
    ];
    tracks.retain_in_frustum(
        cand.eye,
        &rot_to_row_major(&cand.rotation),
        intrinsics,
        render_size,
        TOP_UP_FRUSTUM_MARGIN_PX,
    );
    if tracks.world_points_enu.len() > 1400 {
        let drop_n = tracks.world_points_enu.len() - 1400;
        let sb_drop = drop_n.min(tracks.sign_bits.len());
        let kb_drop = drop_n.min(tracks.kashin_i8.len());
        tracks.world_points_enu.drain(..drop_n);
        tracks.sign_bits.drain(..sb_drop);
        tracks.kashin_i8.drain(..kb_drop);
    }
    let sin_down = -cand.rotation[(2, 2)];
    let ray_cap = if sin_down < 0.08 {
        qry_feat.pixels.len().min(128)
    } else if tracks.len() >= 256 {
        qry_feat.pixels.len().min(160)
    } else {
        qry_feat.pixels.len().min(384)
    };
    let hits = surface.raycast_pixels(
        cand.eye,
        &cand.rotation,
        &cand.k,
        &qry_feat.pixels[..ray_cap],
    );
    let world_opts: Vec<Option<[f64; 3]>> =
        hits.into_iter().map(|h| h.map(|x| x.world_enu)).collect();
    let fresh = ActiveTrackSet::from_prepared_query(prepared, &world_opts, 384);
    extend_dedup_enu(tracks, &fresh, TOP_UP_DEDUP_METERS, 2048);

    let steep_nadir = sin_down > 0.85;
    if let Ok(virt) = store.project_virtual_kashin(&VirtualViewCamera {
        eye_enu: cand.eye,
        world_to_cam: rot_to_row_major(&cand.rotation),
        intrinsics,
        image_size: render_size,
        max_landmarks: 512,
        min_cos_incidence: if steep_nadir {
            0.45
        } else if sin_down < 0.25 {
            0.72
        } else {
            0.84
        },
        min_cos_orientation: if sin_down > 0.95 { 0.94 } else { 0.80 },
    }) && !virt.world_points_enu.is_empty()
    {
        let bank_fresh = ActiveTrackSet {
            world_points_enu: virt.world_points_enu,
            sign_bits: virt.sign_bits,
            kashin_i8: virt.kashin_i8,
        };
        extend_dedup_enu(tracks, &bank_fresh, TOP_UP_DEDUP_METERS, 2048);
    }
}

/// Merge up to `768` arrived anchor landmarks into a healthy foreground track set only when
/// their reprojections under `(prop_eye, prop_rot)` and `(fg_eye, fg_rot)` agree within `4.0 px`.
pub fn merge_consistent_fg_tracks(
    fg_tracks: &mut ActiveTrackSet,
    arrived_tracks: &ActiveTrackSet,
    (prop_eye, prop_rot): ([f64; 3], &Matrix3<f64>),
    (fg_eye, fg_rot): ([f64; 3], &Matrix3<f64>),
    k_render: &Matrix3<f64>,
    render_size: [u32; 2],
) -> bool {
    if arrived_tracks.is_empty() {
        return false;
    }
    let Some(med_px) = median_reprojection_discrepancy_px(
        &arrived_tracks.world_points_enu,
        (prop_eye, prop_rot),
        (fg_eye, fg_rot),
        k_render,
        render_size,
    ) else {
        return false;
    };
    if med_px > FG_MERGE_MAX_DISCREPANCY_PX {
        return false;
    }
    let mut filtered = arrived_tracks.clone();
    let intrinsics = [
        k_render[(0, 0)],
        k_render[(1, 1)],
        k_render[(0, 2)],
        k_render[(1, 2)],
    ];
    filtered.retain_in_frustum(
        fg_eye,
        &rot_to_row_major(fg_rot),
        intrinsics,
        render_size,
        ARRIVAL_FRUSTUM_MARGIN_PX,
    );
    if filtered.world_points_enu.len() > FG_MERGE_CAP {
        filtered.world_points_enu.truncate(FG_MERGE_CAP);
        filtered.sign_bits.truncate(FG_MERGE_CAP);
        filtered.kashin_i8.truncate(FG_MERGE_CAP);
    }
    extend_dedup_enu(fg_tracks, &filtered, TOP_UP_DEDUP_METERS, 2048);
    true
}

fn median_reprojection_discrepancy_px(
    world_points: &[[f64; 3]],
    (eye_a, rot_a): ([f64; 3], &Matrix3<f64>),
    (eye_b, rot_b): ([f64; 3], &Matrix3<f64>),
    k: &Matrix3<f64>,
    render_size: [u32; 2],
) -> Option<f64> {
    let wf = f64::from(render_size[0]);
    let hf = f64::from(render_size[1]);
    let step = (world_points.len() / 64).max(1);
    let mut diffs = Vec::with_capacity(64);
    for &w in world_points.iter().step_by(step).take(64) {
        let Some(pa) = project_px(w, eye_a, rot_a, k) else {
            continue;
        };
        let Some(pb) = project_px(w, eye_b, rot_b, k) else {
            continue;
        };
        let inside = (pa[0] >= -32.0 && pa[0] <= wf + 32.0 && pa[1] >= -32.0 && pa[1] <= hf + 32.0)
            || (pb[0] >= -32.0 && pb[0] <= wf + 32.0 && pb[1] >= -32.0 && pb[1] <= hf + 32.0);
        if inside {
            diffs.push((pa[0] - pb[0]).hypot(pa[1] - pb[1]));
        }
    }
    if diffs.len() < 4 {
        return None;
    }
    diffs.sort_by(f64::total_cmp);
    Some(diffs[diffs.len() / 2])
}

fn project_px(
    w: [f64; 3],
    eye: [f64; 3],
    rot: &Matrix3<f64>,
    k: &Matrix3<f64>,
) -> Option<[f64; 2]> {
    let cam = rot * (Vector3::from(w) - Vector3::from(eye));
    if cam.z <= 1.0 || !cam.z.is_finite() {
        return None;
    }
    let u = k[(0, 0)] * (cam.x / cam.z) + k[(0, 2)];
    let v = k[(1, 1)] * (cam.y / cam.z) + k[(1, 2)];
    (u.is_finite() && v.is_finite()).then_some([u, v])
}

fn extend_dedup_enu(
    tracks: &mut ActiveTrackSet,
    fresh: &ActiveTrackSet,
    min_spacing_m: f64,
    max_tracks: usize,
) {
    let cell = min_spacing_m.max(0.1);
    let min_sq = min_spacing_m * min_spacing_m;
    let voxel = |p: [f64; 3]| {
        (
            (p[0] / cell).floor() as i64,
            (p[1] / cell).floor() as i64,
            (p[2] / cell).floor() as i64,
        )
    };
    let mut occupied: HashMap<(i64, i64, i64), Vec<[f64; 3]>> =
        HashMap::with_capacity(tracks.len() + fresh.len());
    for &p in &tracks.world_points_enu {
        if p.iter().all(|v| v.is_finite()) {
            occupied.entry(voxel(p)).or_default().push(p);
        }
    }

    let mut add_world = Vec::with_capacity(fresh.len());
    let mut add_signs = Vec::with_capacity(fresh.len());
    let mut add_codes = Vec::with_capacity(fresh.len());
    let count = fresh
        .world_points_enu
        .len()
        .min(fresh.sign_bits.len())
        .min(fresh.kashin_i8.len());

    for i in 0..count {
        if tracks.len() + add_world.len() >= max_tracks {
            break;
        }
        let p = fresh.world_points_enu[i];
        if !p.iter().all(|v| v.is_finite()) {
            continue;
        }
        let (vx, vy, vz) = voxel(p);
        let mut near = false;
        'neighbors: for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(bucket) = occupied.get(&(vx + dx, vy + dy, vz + dz))
                        && bucket.iter().any(|q| {
                            let de = p[0] - q[0];
                            let dn = p[1] - q[1];
                            let du = p[2] - q[2];
                            de * de + dn * dn + du * du < min_sq
                        })
                    {
                        near = true;
                        break 'neighbors;
                    }
                }
            }
        }
        if !near {
            occupied.entry((vx, vy, vz)).or_default().push(p);
            add_world.push(p);
            add_signs.push(fresh.sign_bits[i]);
            add_codes.push(fresh.kashin_i8[i]);
        }
    }
    tracks.extend_capped(&add_world, &add_signs, &add_codes, max_tracks);
}
