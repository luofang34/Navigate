//! Virtual camera frustum projection across [`CorridorTileEntry`] slices and iterators.

use super::{
    CorridorTileEntry, VirtualCorridorView, VirtualViewCamera,
    codec::{
        has_nearby_pixel, incidence_cosine, mat3_mul, scale_similarity, tile_overlaps_frustum,
        unit_cosine_i8,
    },
};
use crate::{
    HadamardProjector, SparseFeatures,
    kashin::{try_normalize64, unpack_kashin_i8},
};

type ProjectedCand = (f32, f32, usize, usize, usize, [f64; 2], [f64; 3]);

/// Project visible 3D landmarks from `tiles` into `cam` to synthesize a [`VirtualCorridorView`].
#[must_use]
pub fn project_tiles_virtual_view<'a>(
    tiles: impl IntoIterator<Item = &'a CorridorTileEntry>,
    cam: &VirtualViewCamera,
    projector: &HadamardProjector,
) -> VirtualCorridorView {
    let [w, h] = cam.image_size;
    let [fx, fy, cx, cy] = cam.intrinsics;
    if w == 0 || h == 0 || cam.max_landmarks == 0 || fx <= 1e-6 || fy <= 1e-6 {
        return empty_virtual_view();
    }
    let wf = f64::from(w);
    let hf = f64::from(h);
    let live_cam_x = [
        cam.world_to_cam[0],
        cam.world_to_cam[1],
        cam.world_to_cam[2],
    ];
    let zw = (w.div_ceil(8)).max(1) as usize;
    let zh = (h.div_ceil(8)).max(1) as usize;
    let mut min_depth = vec![f32::INFINITY; zw * zh];
    let mut visible_tiles = 0_usize;
    let mut tile_refs: Vec<&'a CorridorTileEntry> = Vec::new();
    let mut cands: Vec<ProjectedCand> = Vec::new();

    for tile in tiles {
        if tile.is_empty() || !tile_overlaps_frustum(tile, cam) {
            continue;
        }
        let t_idx = tile_refs.len();
        tile_refs.push(tile);
        visible_tiles += 1;
        for l_idx in 0..tile.len() {
            let rel = tile.rel_enu[l_idx];
            let world = [
                tile.anchor_enu[0] + f64::from(rel[0]),
                tile.anchor_enu[1] + f64::from(rel[1]),
                tile.anchor_enu[2] + f64::from(rel[2]),
            ];
            let v = [
                world[0] - cam.eye_enu[0],
                world[1] - cam.eye_enu[1],
                world[2] - cam.eye_enu[2],
            ];
            let pc = mat3_mul(&cam.world_to_cam, v);
            if pc[2] <= 1.0 {
                continue;
            }
            let u_px = fx * (pc[0] / pc[2]) + cx;
            let v_px = fy * (pc[1] / pc[2]) + cy;
            if !(0.0..wf).contains(&u_px) || !(0.0..hf).contains(&v_px) {
                continue;
            }
            let z_cell = ((v_px as u32 >> 3) as usize) * zw + ((u_px as u32 >> 3) as usize);
            let depth_f = pc[2] as f32;
            if depth_f < min_depth[z_cell] {
                min_depth[z_cell] = depth_f;
            }
            let cos_ray = incidence_cosine(tile.obs_dirs_i8[l_idx], v);
            if cos_ray < cam.min_cos_incidence {
                continue;
            }
            let is_nadir = tile.obs_dirs_i8[l_idx][2] > 120;
            let cos_ori = unit_cosine_i8(tile.cam_x_i8[l_idx], live_cam_x);
            if cos_ori < cam.min_cos_orientation {
                continue;
            }
            let Some(scale_sim) = scale_similarity(tile.ref_dist_m[l_idx], v, fx, is_nadir) else {
                continue;
            };
            let sal = f32::from(tile.scores_u8[l_idx]) / 255.0;
            let eff_ori = if is_nadir && cam.min_cos_orientation < 0.0 {
                0.85 + 0.15 * cos_ori.max(0.0)
            } else {
                cos_ori.max(0.05)
            };
            let view_sim = cos_ray.max(0.05).powi(3) * eff_ori.powi(4);
            let prio = view_sim * scale_sim * (0.25 + 0.75 * sal);
            cands.push((prio, depth_f, z_cell, t_idx, l_idx, [u_px, v_px], world));
        }
    }

    cands.retain(|c| c.1 <= min_depth[c.2] * 1.18 + 30.0);
    cands.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| a.3.cmp(&b.3))
            .then_with(|| a.4.cmp(&b.4))
    });

    let scale = (800.0 / wf).min(600.0 / hf);
    let ox = (800.0 - wf * scale) * 0.5 + (scale - 1.0) * 0.5;
    let oy = (600.0 - hf * scale) * 0.5 + (scale - 1.0) * 0.5;
    let keep_cap = cam.max_landmarks.min(cands.len());
    let nw = (w.div_ceil(4)).max(1) as usize;
    let nh = (h.div_ceil(4)).max(1) as usize;
    let mut nms_grid: Vec<Option<[f64; 2]>> = vec![None; nw * nh];
    let mut region_counts = [0_usize; 64];
    let region_quota = (cam.max_landmarks / 32).max(8);
    let mut out = VirtualCorridorView {
        features: SparseFeatures {
            pixels: Vec::with_capacity(keep_cap),
            model_pixels: Vec::with_capacity(keep_cap),
            scores: Vec::with_capacity(keep_cap),
            descriptors: Vec::with_capacity(keep_cap * 64),
        },
        world_points_enu: Vec::with_capacity(keep_cap),
        sign_bits: Vec::with_capacity(keep_cap),
        kashin_i8: Vec::with_capacity(keep_cap),
        visible_tiles,
    };

    let mut remaining = cands;
    for enforce_quota in [true, false] {
        let mut next = Vec::new();
        for cand_item in remaining {
            if out.world_points_enu.len() >= cam.max_landmarks {
                break;
            }
            let px = cand_item.5;
            let n_cell = ((px[1] as u32 >> 2) as usize) * nw + ((px[0] as u32 >> 2) as usize);
            if nms_grid[n_cell].is_some() || has_nearby_pixel(&nms_grid, nw, nh, px, 16.0) {
                continue;
            }
            if enforce_quota {
                let rx = (((px[0] / wf) * 8.0) as usize).min(7);
                let ry = (((px[1] / hf) * 8.0) as usize).min(7);
                let reg = ry * 8 + rx;
                if region_counts[reg] >= region_quota {
                    next.push(cand_item);
                    continue;
                }
                region_counts[reg] += 1;
            }
            nms_grid[n_cell] = Some(px);
            push_virtual_landmark(
                &mut out,
                tile_refs[cand_item.3],
                cand_item,
                (scale, ox, oy),
                projector,
            );
        }
        remaining = next;
    }
    out
}

fn push_virtual_landmark(
    out: &mut VirtualCorridorView,
    tile: &CorridorTileEntry,
    (prio, _, _, _, l_idx, px, world): ProjectedCand,
    (scale, ox, oy): (f64, f64, f64),
    projector: &HadamardProjector,
) {
    let code = tile.kashin_i8[l_idx];
    let unit = try_normalize64(&unpack_kashin_i8(&code)).unwrap_or([0.0; 64]);
    out.features.pixels.push(px);
    out.features
        .model_pixels
        .push([(px[0] * scale + ox) as f32, (px[1] * scale + oy) as f32]);
    out.features.scores.push(prio);
    out.features
        .descriptors
        .extend_from_slice(&projector.unproject(&unit));
    out.world_points_enu.push(world);
    out.sign_bits.push(tile.sign_bits[l_idx]);
    out.kashin_i8.push(code);
}

fn empty_virtual_view() -> VirtualCorridorView {
    VirtualCorridorView {
        features: SparseFeatures::empty(),
        world_points_enu: Vec::new(),
        sign_bits: Vec::new(),
        kashin_i8: Vec::new(),
        visible_tiles: 0,
    }
}
