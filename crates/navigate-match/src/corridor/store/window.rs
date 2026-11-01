//! Frustum-aware spatial window selection and LRU eviction for [`super::CorridorTileStore`].

use super::{MAX_RESIDENT_WINDOW_BYTES, StoreInner, TileAabb};
use crate::corridor::{
    CorridorTileEntry, CorridorTileKey, VirtualViewCamera, codec::aabb_overlaps_frustum,
};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

pub(super) fn select_window_keys_inner(
    guard: &StoreInner,
    tile_size_m: f32,
    max_cached_tiles: usize,
    cam: &VirtualViewCamera,
) -> Vec<CorridorTileKey> {
    if cam.image_size[0] == 0
        || cam.image_size[1] == 0
        || cam.max_landmarks == 0
        || cam.intrinsics[0] <= 1e-6
        || cam.intrinsics[1] <= 1e-6
    {
        return Vec::new();
    }
    let s = f64::from(tile_size_m.max(10.0));
    let fwd = [cam.world_to_cam[6], cam.world_to_cam[7]];
    let fwd_z = cam.world_to_cam[8];
    let fwd_h = (fwd[0] * fwd[0] + fwd[1] * fwd[1]).sqrt();
    let upward_horizon = fwd_h > 0.25 && -fwd_z < 0.05;
    let shallow_horizon = fwd_h > 0.25 && -fwd_z < 0.25;
    let look_dist = if upward_horizon {
        (cam.eye_enu[2].abs() * 8.5).clamp(5_500.0, 13_000.0)
    } else if shallow_horizon {
        (cam.eye_enu[2].abs() * 5.5).clamp(3_500.0, 7_500.0)
    } else if fwd_h > 0.25 {
        (cam.eye_enu[2].abs() * 2.5).clamp(0.0, 3_000.0)
    } else {
        0.0
    };
    let ux = if fwd_h > 1e-6 { fwd[0] / fwd_h } else { 0.0 };
    let uy = if fwd_h > 1e-6 { fwd[1] / fwd_h } else { 0.0 };
    let focus_e = cam.eye_enu[0] + ux * look_dist;
    let focus_n = cam.eye_enu[1] + uy * look_dist;
    let max_range_sq = if upward_horizon {
        (22_000.0_f64 + s).powi(2)
    } else if shallow_horizon {
        (14_000.0_f64 + s).powi(2)
    } else {
        (6_800.0_f64 + s).powi(2)
    };

    let mut ranked: Vec<(f64, CorridorTileKey)> = guard
        .indexed_keys
        .iter()
        .filter_map(|&key| {
            if let Some(&(anchor, b_min, b_max)) = guard.known_bounds.get(&key)
                && !aabb_overlaps_frustum(anchor, b_min, b_max, cam)
            {
                return None;
            }
            let cx = (f64::from(key.tile_x) + 0.5) * s;
            let cy = (f64::from(key.tile_y) + 0.5) * s;
            let de_eye = cx - cam.eye_enu[0];
            let dn_eye = cy - cam.eye_enu[1];
            if de_eye * de_eye + dn_eye * dn_eye > max_range_sq {
                return None;
            }
            if fwd_h > 0.5 && (de_eye * ux + dn_eye * uy) < -s * 1.25 {
                return None;
            }
            let de_foc = cx - focus_e;
            let dn_foc = cy - focus_n;
            Some((de_foc * de_foc + dn_foc * dn_foc, key))
        })
        .collect();
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
    ranked
        .into_iter()
        .take(max_cached_tiles)
        .map(|(_, k)| k)
        .collect()
}

pub(super) fn tile_bytes(t: &CorridorTileEntry) -> usize {
    size_of::<CorridorTileEntry>() + t.len() * 93
}

pub(super) fn record_store_bounds(
    inner: &mut StoreInner,
    key: CorridorTileKey,
    tile: &CorridorTileEntry,
) {
    record_bounds_map(&mut inner.known_bounds, &mut inner.z_span, key, tile);
}

pub(super) fn record_bounds_map(
    bounds: &mut HashMap<CorridorTileKey, TileAabb>,
    z_span: &mut Option<[f32; 2]>,
    key: CorridorTileKey,
    tile: &CorridorTileEntry,
) {
    let _ = bounds.insert(
        key,
        (tile.anchor_enu, tile.bounds_min_rel, tile.bounds_max_rel),
    );
    if !tile.is_empty() {
        let lo = tile.anchor_enu[2] as f32 + tile.bounds_min_rel[2];
        let hi = tile.anchor_enu[2] as f32 + tile.bounds_max_rel[2];
        *z_span = Some(match *z_span {
            Some([cur_lo, cur_hi]) => [cur_lo.min(lo), cur_hi.max(hi)],
            None => [lo, hi],
        });
    }
}

pub(super) fn estimated_key_overlaps_frustum(
    key: CorridorTileKey,
    s: f64,
    [z_min, z_max]: [f32; 2],
    cam: &VirtualViewCamera,
) -> bool {
    let d_eye = ((f64::from(key.tile_x) + 0.5) * s - cam.eye_enu[0])
        .hypot((f64::from(key.tile_y) + 0.5) * s - cam.eye_enu[1]) as f32;
    let pad_xy = (s as f32) * 0.15 + 8.0;
    let pad_z = 200.0 + 0.35 * d_eye;
    let anchor = [f64::from(key.tile_x) * s, f64::from(key.tile_y) * s, 0.0];
    aabb_overlaps_frustum(
        anchor,
        [-pad_xy, -pad_xy, z_min - pad_z],
        [(s as f32) + pad_xy, (s as f32) + pad_xy, z_max + pad_z],
        cam,
    )
}

pub(super) fn promote_key(order: &mut VecDeque<CorridorTileKey>, key: CorridorTileKey) {
    if let Some(pos) = order.iter().position(|&k| k == key) {
        let _ = order.remove(pos);
    }
    order.push_back(key);
}

pub(super) fn cache_window_bytes(inner: &StoreInner) -> usize {
    inner.cache_entries.values().map(|t| tile_bytes(t)).sum()
}

pub(super) fn touch_lru(
    inner: &mut StoreInner,
    key: CorridorTileKey,
    tile: Arc<CorridorTileEntry>,
    cap: usize,
) {
    promote_key(&mut inner.cache_order, key);
    let _ = inner.cache_entries.insert(key, tile);
    while inner.cache_entries.len() > cap
        || (inner.cache_entries.len() > 1 && cache_window_bytes(inner) > MAX_RESIDENT_WINDOW_BYTES)
    {
        let Some(evicted) = inner.cache_order.pop_front() else {
            break;
        };
        let _ = inner.cache_entries.remove(&evicted);
    }
}
