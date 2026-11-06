//! Shared voxel-deduplicated 3D landmark bucket assembly for corridor tiles.

use crate::error::FlightVisualError;
use navigate_match::{CorridorTileEntry, CorridorTileKey, HadamardProjector, SparseFeatures};
use std::collections::BTreeMap;

pub(crate) type BandVoxelMap = BTreeMap<u8, BTreeMap<(i32, i32), VoxelLandmark>>;

pub(crate) struct VoxelLandmark {
    pub world_enu: [f64; 3],
    pub obs_dir_enu: [f32; 3],
    pub cam_x_enu: [f32; 3],
    pub ref_dist_m: u16,
    pub score: f32,
    pub descriptor: [f32; 64],
}

pub(crate) fn unit_obs_dir(eye_enu: [f64; 3], world_enu: [f64; 3]) -> [f32; 3] {
    let d = [
        (eye_enu[0] - world_enu[0]) as f32,
        (eye_enu[1] - world_enu[1]) as f32,
        (eye_enu[2] - world_enu[2]) as f32,
    ];
    let norm = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    if norm > 1e-6 {
        d.map(|v| v / norm)
    } else {
        [0.0, 0.0, 1.0]
    }
}

pub(crate) fn build_tile_entry_from_bands(
    key: CorridorTileKey,
    tile_m: f64,
    max_per_tile: usize,
    projector: &HadamardProjector,
    band_map: BandVoxelMap,
) -> Result<Option<CorridorTileEntry>, FlightVisualError> {
    if band_map.is_empty() {
        return Ok(None);
    }
    let num_bands = band_map.len().max(1);
    let per_band_cap = (max_per_tile / num_bands).max(64);
    let mut lms = Vec::new();
    for (_, vox_map) in band_map {
        let mut band_lms: Vec<VoxelLandmark> = vox_map.into_values().collect();
        band_lms.sort_by(|a, b| b.score.total_cmp(&a.score));
        band_lms.truncate(per_band_cap);
        lms.extend(band_lms);
    }
    if lms.is_empty() {
        return Ok(None);
    }
    lms.sort_by(|a, b| b.score.total_cmp(&a.score));
    lms.truncate(max_per_tile);
    let mean_up = lms.iter().map(|l| l.world_enu[2]).sum::<f64>() / (lms.len() as f64);
    let anchor_enu = [
        (f64::from(key.tile_x) + 0.5) * tile_m,
        (f64::from(key.tile_y) + 0.5) * tile_m,
        mean_up,
    ];
    let mut feat = SparseFeatures {
        pixels: Vec::with_capacity(lms.len()),
        model_pixels: Vec::with_capacity(lms.len()),
        scores: Vec::with_capacity(lms.len()),
        descriptors: Vec::with_capacity(lms.len() * 64),
    };
    let mut world_pts = Vec::with_capacity(lms.len());
    let mut obs_dirs = Vec::with_capacity(lms.len());
    let mut cam_xs = Vec::with_capacity(lms.len());
    let mut ref_dists = Vec::with_capacity(lms.len());
    for lm in &lms {
        feat.pixels.push([0.0, 0.0]);
        feat.model_pixels.push([0.0, 0.0]);
        feat.scores.push(lm.score);
        feat.descriptors.extend_from_slice(&lm.descriptor);
        world_pts.push(lm.world_enu);
        obs_dirs.push(lm.obs_dir_enu);
        cam_xs.push(lm.cam_x_enu);
        ref_dists.push(lm.ref_dist_m);
    }
    Ok(Some(CorridorTileEntry::from_landmarks(
        key,
        anchor_enu,
        projector,
        &world_pts,
        (&obs_dirs, &cam_xs, &ref_dists),
        &feat,
    )?))
}
