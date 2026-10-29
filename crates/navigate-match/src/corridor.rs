//! Tile-indexed 3D corridor landmark bank (`NVKTILE1`) and virtual frustum projection.
//!
//! [`CorridorLandmarkBank`] stores viewpoint-independent 3D terrain landmarks in
//! East-North-Up (ENU) ground tiles. At runtime, [`CorridorLandmarkBank::project_virtual_view`]
//! projects visible 3D landmarks into any live camera pose and intrinsics without rendering.

mod codec;
mod track;

pub use track::{ActiveTrackSet, TrackMatchConfig, Tracked3dMatch};

use crate::{
    HadamardProjector, MatchError, SparseFeatures, ViewSketch,
    kashin::{pack_kashin_i8, sign_bits64, try_normalize64, unpack_kashin_i8},
};
use codec::{
    decode_corridor_bank, encode_corridor_bank, ensure, has_nearby_pixel, incidence_cosine,
    mat3_mul, quantize_unit_i8, scale_similarity, tile_overlaps_frustum, unit_cosine_i8,
};

/// Eight-byte magic header of a serialized [`CorridorLandmarkBank`].
pub const CORRIDOR_MAGIC: &[u8; 8] = b"NVKTILE1";
/// Binary format version of [`CorridorLandmarkBank`].
pub const CORRIDOR_VERSION: u32 = 1;

pub(super) const MAX_TILES: usize = 65_536;
pub(super) const MAX_TILE_LANDMARKS: usize = 65_536;

/// Integer 2D grid coordinates of one ENU corridor tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CorridorTileKey {
    /// Tile index along the East axis (`floor(east_m / tile_size_m)`).
    pub tile_x: i32,
    /// Tile index along the North axis (`floor(north_m / tile_size_m)`).
    pub tile_y: i32,
}

/// Viewpoint-independent 3D terrain landmarks for one ENU ground cell.
#[derive(Clone, Debug, PartialEq)]
pub struct CorridorTileEntry {
    /// Tile index along the East axis.
    pub tile_x: i32,
    /// Tile index along the North axis.
    pub tile_y: i32,
    /// Tile anchor `[east, north, up]` in ENU meters.
    pub anchor_enu: [f64; 3],
    /// Minimum `[east, north, up]` offset relative to `anchor_enu` in meters.
    pub bounds_min_rel: [f32; 3],
    /// Maximum `[east, north, up]` offset relative to `anchor_enu` in meters.
    pub bounds_max_rel: [f32; 3],
    /// Tile-level 160-byte [`ViewSketch`] summary.
    pub sketch: ViewSketch,
    /// Landmark 3D offsets `[de, dn, du]` relative to `anchor_enu` (`12` bytes each).
    pub rel_enu: Vec<[f32; 3]>,
    /// Quantized unit ENU rays from each landmark to its canonical camera (`3` bytes each).
    pub obs_dirs_i8: Vec<[i8; 3]>,
    /// Quantized unit ENU sensor `+X` axes of each landmark's canonical camera (`3` bytes each).
    pub cam_x_i8: Vec<[i8; 3]>,
    /// Canonical camera 3D viewing ranges in meters (`0` disables scale gating, `2` bytes each).
    pub ref_dist_m: Vec<u16>,
    /// Quantized keypoint saliency scores (`0..=255`, `1` byte each).
    pub scores_u8: Vec<u8>,
    /// Contiguous 64-bit Kashin sign sketches (`8` bytes each).
    pub sign_bits: Vec<u64>,
    /// Bounded 64-byte rotated Kashin `i8` descriptors (`64` bytes each).
    pub kashin_i8: Vec<[u8; 64]>,
}

impl CorridorTileEntry {
    /// Encode 3D terrain landmarks and [`SparseFeatures`] into a [`CorridorTileEntry`].
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when feature shapes, coordinates, or landmark counts are invalid.
    pub fn from_landmarks(
        key: CorridorTileKey,
        anchor_enu: [f64; 3],
        projector: &HadamardProjector,
        world_points_enu: &[[f64; 3]],
        view_meta: (&[[f32; 3]], &[[f32; 3]], &[u16]),
        features: &SparseFeatures,
    ) -> Result<Self, MatchError> {
        let (obs_dirs_enu, cam_x_enu, dists_m) = view_meta;
        let n = features.pixels.len();
        let ok = n <= MAX_TILE_LANDMARKS
            && anchor_enu.iter().all(|v| v.is_finite())
            && world_points_enu.len() == n
            && (obs_dirs_enu.is_empty() || obs_dirs_enu.len() == n)
            && (cam_x_enu.is_empty() || cam_x_enu.len() == n)
            && (dists_m.is_empty() || dists_m.len() == n)
            && features.descriptors.len() == n * 64
            && (features.scores.is_empty() || features.scores.len() == n);
        ensure(ok, "invalid corridor tile landmarks")?;

        let mut rel_enu = Vec::with_capacity(n);
        let mut obs_dirs_i8 = Vec::with_capacity(n);
        let mut cam_x_i8 = Vec::with_capacity(n);
        let mut ref_dist_m = Vec::with_capacity(n);
        let mut scores_u8 = Vec::with_capacity(n);
        let mut sign_bits = Vec::with_capacity(n);
        let mut kashin_i8 = Vec::with_capacity(n);
        let mut b_min = [f32::INFINITY; 3];
        let mut b_max = [f32::NEG_INFINITY; 3];

        for (i, chunk) in features.descriptors.as_chunks::<64>().0.iter().enumerate() {
            let u = try_normalize64(chunk).ok_or(MatchError::Output("invalid descriptor"))?;
            let w = world_points_enu[i];
            ensure(w.iter().all(|v| v.is_finite()), "non-finite landmark ENU")?;
            let rel = [
                (w[0] - anchor_enu[0]) as f32,
                (w[1] - anchor_enu[1]) as f32,
                (w[2] - anchor_enu[2]) as f32,
            ];
            ensure(
                rel.iter().all(|v| v.is_finite()),
                "landmark offset overflow",
            )?;
            for axis in 0..3 {
                b_min[axis] = b_min[axis].min(rel[axis]);
                b_max[axis] = b_max[axis].max(rel[axis]);
            }
            let rot = projector.project(&u);
            rel_enu.push(rel);
            obs_dirs_i8.push(quantize_unit_i8(
                obs_dirs_enu.get(i).copied().unwrap_or([0.0, 0.0, 1.0]),
                [0, 0, 127],
            ));
            cam_x_i8.push(quantize_unit_i8(
                cam_x_enu.get(i).copied().unwrap_or([1.0, 0.0, 0.0]),
                [127, 0, 0],
            ));
            ref_dist_m.push(dists_m.get(i).copied().unwrap_or(0));
            let raw_score = features.scores.get(i).copied().unwrap_or(1.0);
            let s_clamped = if raw_score.is_finite() {
                raw_score.clamp(0.0, 1.0)
            } else {
                0.5
            };
            scores_u8.push((s_clamped * 255.0).round() as u8);
            sign_bits.push(sign_bits64(&rot));
            kashin_i8.push(pack_kashin_i8(&rot));
        }
        if n == 0 {
            b_min = [0.0; 3];
            b_max = [0.0; 3];
        }
        let entry = Self {
            tile_x: key.tile_x,
            tile_y: key.tile_y,
            anchor_enu,
            bounds_min_rel: b_min,
            bounds_max_rel: b_max,
            sketch: ViewSketch::from_features(projector, features),
            rel_enu,
            obs_dirs_i8,
            cam_x_i8,
            ref_dist_m,
            scores_u8,
            sign_bits,
            kashin_i8,
        };
        entry.validate().map(|()| entry)
    }

    /// Return the number of 3D landmarks stored in this tile.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rel_enu.len()
    }

    /// Return `true` when this tile stores zero landmarks.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rel_enu.is_empty()
    }

    /// Reconstruct the ENU world coordinates of landmark `index`.
    #[must_use]
    pub fn world_point(&self, index: usize) -> Option<[f64; 3]> {
        let rel = *self.rel_enu.get(index)?;
        Some([
            self.anchor_enu[0] + f64::from(rel[0]),
            self.anchor_enu[1] + f64::from(rel[1]),
            self.anchor_enu[2] + f64::from(rel[2]),
        ])
    }

    pub(super) fn validate(&self) -> Result<(), MatchError> {
        let n = self.rel_enu.len();
        let lens_ok = n <= MAX_TILE_LANDMARKS
            && self.obs_dirs_i8.len() == n
            && self.cam_x_i8.len() == n
            && self.ref_dist_m.len() == n
            && self.scores_u8.len() == n
            && self.sign_bits.len() == n
            && self.kashin_i8.len() == n;
        let finite_ok = self.anchor_enu.iter().all(|v| v.is_finite())
            && self.bounds_min_rel.iter().all(|v| v.is_finite())
            && self.bounds_max_rel.iter().all(|v| v.is_finite())
            && (0..3).all(|a| self.bounds_min_rel[a] <= self.bounds_max_rel[a])
            && self.rel_enu.iter().flatten().all(|v| v.is_finite());
        ensure(lens_ok && finite_ok, "invalid corridor tile entry")
    }
}

/// Camera pose, pinhole intrinsics, and culling bounds for virtual view synthesis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VirtualViewCamera {
    /// Camera optical center `[east, north, up]` in ENU meters.
    pub eye_enu: [f64; 3],
    /// Row-major `3 x 3` rotation matrix from ENU world coordinates to camera coordinates.
    pub world_to_cam: [f64; 9],
    /// Pinhole intrinsics `[fx, fy, cx, cy]` in pixels.
    pub intrinsics: [f64; 4],
    /// Virtual image dimensions `[width, height]` in pixels.
    pub image_size: [u32; 2],
    /// Maximum landmarks retained in the virtual view.
    pub max_landmarks: usize,
    /// Minimum cosine between canonical and live 3D viewing rays (`-1.0` disables ray gating).
    pub min_cos_incidence: f32,
    /// Minimum cosine between canonical and live sensor `+X` axes (`-1.0` disables orientation gating).
    pub min_cos_orientation: f32,
}

/// Projected 2D-3D virtual reference view synthesized from [`CorridorLandmarkBank`].
#[derive(Clone, Debug, PartialEq)]
pub struct VirtualCorridorView {
    /// Reconstructed [`SparseFeatures`] with exact live `pixels` and `model_pixels`.
    pub features: SparseFeatures,
    /// Aligned 3D ENU world coordinates (`features.len()` elements).
    pub world_points_enu: Vec<[f64; 3]>,
    /// Aligned 64-bit Kashin sign sketches (`features.len()` elements).
    pub sign_bits: Vec<u64>,
    /// Aligned 64-byte Kashin `i8` codes (`features.len()` elements).
    pub kashin_i8: Vec<[u8; 64]>,
    /// Number of corridor tiles overlapping the camera frustum.
    pub visible_tiles: usize,
}

/// Tile-indexed bank of 3D [`CorridorTileEntry`] cells (`NVKTILE1`).
#[derive(Clone, Debug, PartialEq)]
pub struct CorridorLandmarkBank {
    /// SHA-256 digest of the source terrain and imagery catalog.
    pub catalog_sha256: [u8; 32],
    /// SHA-256 digest of the [`XFeat`](crate::XFeat) encoder contract.
    pub encoder_contract: [u8; 32],
    /// Seed of the [`HadamardProjector`] for all tiles in this bank.
    pub seed: u64,
    /// Horizontal ground tile edge length in ENU meters.
    pub tile_size_m: f32,
    /// Tile entries stored in this corridor bank.
    pub tiles: Vec<CorridorTileEntry>,
}

impl CorridorLandmarkBank {
    /// Create and validate a [`CorridorLandmarkBank`].
    pub fn new(
        catalog_sha256: [u8; 32],
        encoder_contract: [u8; 32],
        seed: u64,
        tile_size_m: f32,
        tiles: Vec<CorridorTileEntry>,
    ) -> Result<Self, MatchError> {
        let bank = Self {
            catalog_sha256,
            encoder_contract,
            seed,
            tile_size_m,
            tiles,
        };
        bank.check().map(|()| bank)
    }

    pub(super) fn check(&self) -> Result<(), MatchError> {
        let size_ok = self.tile_size_m.is_finite() && self.tile_size_m >= 10.0;
        ensure(
            size_ok && self.tiles.len() <= MAX_TILES,
            "invalid corridor bank header",
        )?;
        self.tiles.iter().try_for_each(CorridorTileEntry::validate)
    }

    /// Return the total number of 3D landmarks across all tiles.
    #[must_use]
    pub fn total_landmarks(&self) -> usize {
        self.tiles.iter().map(CorridorTileEntry::len).sum()
    }

    /// Serialize this corridor bank into deterministic `NVKTILE1` bytes with a SHA-256 trailer.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        encode_corridor_bank(self)
    }

    /// Decode and verify an `NVKTILE1` byte slice.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, MatchError> {
        decode_corridor_bank(bytes)
    }

    /// Write this corridor bank to `path` on native targets.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_file(&self, path: &std::path::Path) -> Result<(), MatchError> {
        self.check()?;
        std::fs::write(path, self.to_bytes()).map_err(|e| MatchError::Device {
            stage: "corridor bank write",
            message: e.to_string(),
        })
    }

    /// Read a [`CorridorLandmarkBank`] from `path` on native targets.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_file(path: &std::path::Path) -> Result<Self, MatchError> {
        let bytes = std::fs::read(path).map_err(|e| MatchError::Device {
            stage: "corridor bank read",
            message: e.to_string(),
        })?;
        Self::from_bytes(&bytes)
    }

    /// Project visible 3D landmarks into `cam` to synthesize an exact [`VirtualCorridorView`].
    #[must_use]
    pub fn project_virtual_view(
        &self,
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
        let mut cands: Vec<ProjectedCand> = Vec::new();

        for (t_idx, tile) in self.tiles.iter().enumerate() {
            if tile.is_empty() || !tile_overlaps_frustum(tile, cam) {
                continue;
            }
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
                let cos_ori = unit_cosine_i8(tile.cam_x_i8[l_idx], live_cam_x);
                if cos_ori < cam.min_cos_orientation {
                    continue;
                }
                let Some(scale_sim) = scale_similarity(tile.ref_dist_m[l_idx], v, fx) else {
                    continue;
                };
                let sal = f32::from(tile.scores_u8[l_idx]) / 255.0;
                let view_sim = cos_ray.max(0.05).powi(3) * cos_ori.max(0.05).powi(4);
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
                    &self.tiles[cand_item.3],
                    cand_item,
                    (scale, ox, oy),
                    projector,
                );
            }
            remaining = next;
        }
        out
    }
}

type ProjectedCand = (f32, f32, usize, usize, usize, [f64; 2], [f64; 3]);

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

#[cfg(test)]
mod tests;
