//! On-demand 3D landmark tile synthesis and background `corridor-tile-builder` worker.

use super::{
    background::BackgroundWorker,
    canonical_tile_poses,
    nadir::{is_nadir_pose, synthesize_rotated_bands},
    unit_obs_dir,
};
use crate::{
    PipelineError,
    stages::scale_k,
    surface::gpu::GpuQueueRole,
    surface::{Surface, SurfaceGpuRenderer},
    xfeat_runtime::LoadedXFeat,
};
use image::DynamicImage;
use nalgebra::Matrix3;
use navigate_match::{
    CorridorTileEntry, CorridorTileKey, CorridorTileStore, HadamardProjector, SparseFeatures,
};
use std::{
    collections::{BTreeMap, HashSet},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
const LIVE_RENDER_SIZE: [u32; 2] = [800, 600];

type BandVoxelMap = BTreeMap<u8, BTreeMap<(i32, i32), SynthLandmark>>;

struct SynthLandmark {
    world_enu: [f64; 3],
    obs_dir_enu: [f32; 3],
    cam_x_enu: [f32; 3],
    ref_dist_m: u16,
    score: f32,
    descriptor: [f32; 64],
}

#[derive(Clone, Copy)]
pub(super) struct TileSynthParams {
    pub tile_m: f64,
    pub agl_m: f64,
    pub max_per_tile: usize,
    pub stride: u32,
    pub include_obliques: bool,
}

fn live_k_render() -> Matrix3<f64> {
    let k_base = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
    scale_k(&k_base, 1.0, 1.0)
}

pub(super) fn synthesize_corridor_tiles(
    surface: &Surface,
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    xfeat: &LoadedXFeat,
    projector: &HadamardProjector,
    key: CorridorTileKey,
    params: TileSynthParams,
) -> Result<Vec<CorridorTileEntry>, PipelineError> {
    let tile_m = params.tile_m;
    let ce = (f64::from(key.tile_x) + 0.5) * tile_m;
    let cn = (f64::from(key.tile_y) + 0.5) * tile_m;
    let [clat, clon, _] = surface.geodetic([ce, cn, 0.0]);
    let Some(ground_msl) = surface.sample_dem(clat, clon) else {
        return Ok(Vec::new());
    };

    let k_render = live_k_render();
    let views = canonical_tile_poses(
        surface,
        (ce, cn, ground_msl),
        tile_m,
        params.agl_m,
        params.include_obliques,
    );
    let voxel_m = 1.5_f64;
    let max_oblique_dist_m = (params.agl_m * 14.0).max(4_000.0);
    let mut tile_buckets: BTreeMap<CorridorTileKey, BandVoxelMap> = BTreeMap::new();

    for (band, eye, rot) in views {
        let mut gpu_view_opt = None;
        if let Some(lock) = gpu_renderer
            && let Ok(mut guard) = lock.lock()
        {
            if guard.is_none() {
                *guard = SurfaceGpuRenderer::try_new_with_role(
                    surface,
                    &k_render,
                    LIVE_RENDER_SIZE,
                    GpuQueueRole::BackgroundTileBuilder,
                );
            }
            if let Some(gpu) = guard.as_mut() {
                gpu_view_opt = gpu
                    .render_view(surface, eye, &rot, &k_render, LIVE_RENDER_SIZE)
                    .ok();
            }
        }

        let (gray_img, raw, world_hits) = if let Some(ref gv) = gpu_view_opt {
            let raw = xfeat.extract(&gv.image, 1536)?;
            let unproj = gv.unproject_pixels(surface, &raw.pixels);
            (gv.image.clone(), raw, unproj)
        } else {
            let ((rgb, _, _), _) = surface.render_adaptive(
                eye,
                &rot,
                &k_render,
                LIVE_RENDER_SIZE,
                params.stride,
                8,
                0.025,
            );
            let gray_img = DynamicImage::ImageRgb8(rgb).into_luma8();
            let raw = xfeat.extract(&gray_img, 1536)?;
            let hits = surface.raycast_pixels(eye, &rot, &k_render, &raw.pixels);
            let unproj = hits.into_iter().map(|h| h.map(|x| x.world_enu)).collect();
            (gray_img, raw, unproj)
        };

        let mut insert_band = |band_idx: u8,
                               cam_x: [f32; 3],
                               features: &SparseFeatures,
                               world_hits: &[Option<[f64; 3]>]| {
            let chunks = features.descriptors.as_chunks::<64>().0;
            let n_raw = (world_hits.len().max(1)) as f32;
            for (idx, w_opt) in world_hits.iter().enumerate() {
                let Some(w) = *w_opt else {
                    continue;
                };
                let dist_m =
                    ((eye[0] - w[0]).powi(2) + (eye[1] - w[1]).powi(2) + (eye[2] - w[2]).powi(2))
                        .sqrt();
                if params.include_obliques && !(20.0..=max_oblique_dist_m).contains(&dist_m) {
                    continue;
                }
                let dst_key = CorridorTileKey {
                    tile_x: (w[0] / tile_m).floor() as i32,
                    tile_y: (w[1] / tile_m).floor() as i32,
                };
                if dst_key != key {
                    continue;
                }
                let vx = (w[0] / voxel_m).floor() as i32;
                let vy = (w[1] / voxel_m).floor() as i32;
                let score = (1.0 - (idx as f32) / n_raw).clamp(0.01, 1.0);
                let vox_map = tile_buckets
                    .entry(dst_key)
                    .or_default()
                    .entry(band_idx)
                    .or_default();
                if vox_map.get(&(vx, vy)).is_none_or(|prev| score > prev.score) {
                    vox_map.insert(
                        (vx, vy),
                        SynthLandmark {
                            world_enu: w,
                            obs_dir_enu: unit_obs_dir(eye, w),
                            cam_x_enu: cam_x,
                            ref_dist_m: dist_m.round().clamp(1.0, 65_535.0) as u16,
                            score,
                            descriptor: chunks[idx],
                        },
                    );
                }
            }
        };

        let cam_x = [rot[(0, 0)] as f32, rot[(0, 1)] as f32, rot[(0, 2)] as f32];
        insert_band(band, cam_x, &raw, &world_hits);

        if is_nadir_pose(&rot)
            && let Ok(rot_bands) =
                synthesize_rotated_bands(&gray_img, surface, clat, clon, xfeat, 1536)
        {
            for (b_idx, r_band) in rot_bands.into_iter().enumerate() {
                let r_band_idx = band.saturating_add((b_idx as u8) + 1);
                let r_world_hits: Vec<Option<[f64; 3]>> = if let Some(ref gv) = gpu_view_opt {
                    gv.unproject_pixels(surface, &r_band.unrotated_pixels)
                } else {
                    surface
                        .raycast_pixels(eye, &rot, &k_render, &r_band.unrotated_pixels)
                        .into_iter()
                        .map(|h| h.map(|x| x.world_enu))
                        .collect()
                };
                insert_band(
                    r_band_idx,
                    r_band.cam_x_enu,
                    &r_band.features,
                    &r_world_hits,
                );
            }
        }
    }

    let mut entries = Vec::with_capacity(tile_buckets.len());
    for (dst_key, band_buckets) in tile_buckets {
        if let Some(entry) = build_tile_entry_from_bands(
            dst_key,
            tile_m,
            params.max_per_tile,
            projector,
            band_buckets,
        )? {
            entries.push(entry);
        }
    }
    Ok(entries)
}

fn build_tile_entry_from_bands(
    key: CorridorTileKey,
    tile_m: f64,
    max_per_tile: usize,
    projector: &HadamardProjector,
    band_map: BandVoxelMap,
) -> Result<Option<CorridorTileEntry>, PipelineError> {
    if band_map.is_empty() {
        return Ok(None);
    }
    let num_bands = band_map.len().max(1);
    let per_band_cap = (max_per_tile / num_bands).max(64);
    let mut lms = Vec::new();
    for (_, vox_map) in band_map {
        let mut band_lms: Vec<SynthLandmark> = vox_map.into_values().collect();
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

pub(super) struct TileBuilderContext {
    pub surface: Arc<Surface>,
    pub store: Arc<CorridorTileStore>,
    pub projector: HadamardProjector,
    pub xfeat: LoadedXFeat,
    pub params: TileSynthParams,
}

pub(super) struct TileBuildOutcome {
    pub inserted_count: usize,
    pub elapsed_ms: f64,
}

/// Low-priority background worker that synthesizes missing [`CorridorTileEntry`] cells on demand.
pub(super) struct BackgroundTileBuilder {
    worker: BackgroundWorker<CorridorTileKey, Result<TileBuildOutcome, PipelineError>>,
    seen: HashSet<CorridorTileKey>,
    in_flight: usize,
    built_tiles: usize,
}

impl BackgroundTileBuilder {
    pub(super) fn spawn(ctx: TileBuilderContext, prewarm_gpu: bool) -> Result<Self, PipelineError> {
        let init_gpu = prewarm_gpu
            .then(|| {
                SurfaceGpuRenderer::try_new_with_role(
                    &ctx.surface,
                    &live_k_render(),
                    LIVE_RENDER_SIZE,
                    GpuQueueRole::BackgroundTileBuilder,
                )
            })
            .flatten();
        let gpu_renderer = Mutex::new(init_gpu);
        let worker = BackgroundWorker::spawn(
            "corridor-tile-builder",
            SHUTDOWN_TIMEOUT,
            move |key: CorridorTileKey| {
                if ctx.store.has_tile(key) {
                    return Ok(TileBuildOutcome {
                        inserted_count: 0,
                        elapsed_ms: 0.0,
                    });
                }
                let t0 = Instant::now();
                let entries = synthesize_corridor_tiles(
                    &ctx.surface,
                    Some(&gpu_renderer),
                    &ctx.xfeat,
                    &ctx.projector,
                    key,
                    ctx.params,
                )?;
                let mut inserted_count = 0_usize;
                for entry in entries {
                    ctx.store.insert_tile(entry, true)?;
                    inserted_count += 1;
                }
                Ok(TileBuildOutcome {
                    inserted_count,
                    elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
                })
            },
        )?;
        Ok(Self {
            worker,
            seen: HashSet::new(),
            in_flight: 0,
            built_tiles: 0,
        })
    }

    pub(super) fn request_neighborhood(
        &mut self,
        surface: &Surface,
        store: &CorridorTileStore,
        lookahead_enu: [f64; 2],
    ) -> Result<(), PipelineError> {
        let tile_m = f64::from(store.tile_size_m.max(10.0));
        let center = store.tile_key_at(lookahead_enu[0], lookahead_enu[1]);
        let mut candidates = Vec::with_capacity(9);
        for dy in -1..=1_i32 {
            for dx in -1..=1_i32 {
                let key = CorridorTileKey {
                    tile_x: center.tile_x + dx,
                    tile_y: center.tile_y + dy,
                };
                if self.seen.contains(&key) || store.has_tile(key) {
                    continue;
                }
                let ce = (f64::from(key.tile_x) + 0.5) * tile_m;
                let cn = (f64::from(key.tile_y) + 0.5) * tile_m;
                let [clat, clon, _] = surface.geodetic([ce, cn, 0.0]);
                if surface.sample_dem(clat, clon).is_none() {
                    let _ = self.seen.insert(key);
                    continue;
                }
                let dist_sq = (ce - lookahead_enu[0]).powi(2) + (cn - lookahead_enu[1]).powi(2);
                candidates.push((dist_sq, key));
            }
        }
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (_, key) in candidates {
            if self.in_flight >= 4 {
                break;
            }
            let _ = self.seen.insert(key);
            self.worker.submit(key)?;
            self.in_flight += 1;
        }
        Ok(())
    }

    pub(super) fn poll(&mut self, samples: &mut Vec<f64>) -> Result<(), PipelineError> {
        while let Some(res) = self.worker.try_recv()? {
            self.in_flight = self.in_flight.saturating_sub(1);
            let out = res?;
            if out.inserted_count > 0 {
                self.built_tiles += out.inserted_count;
                samples.push(out.elapsed_ms);
            }
        }
        Ok(())
    }

    pub(super) fn built_tiles(&self) -> usize {
        self.built_tiles
    }
}
