//! On-demand 3D landmark tile synthesis and bounded `corridor-tile-builder` worker.

use crate::{
    canonical::canonical_tile_poses,
    error::FlightVisualError,
    nadir::{is_nadir_pose, synthesize_rotated_bands},
    runtime::LoadedXFeat,
    surface::{GpuQueueRole, SharedSurfaceGpu, Surface, SurfaceGpuRenderer},
    tile_assemble::{BandVoxelMap, VoxelLandmark, build_tile_entry_from_bands, unit_obs_dir},
    types::ExecutionPolicy,
    worker::BoundedFifoWorker,
};
use image::DynamicImage;
use nalgebra::Matrix3;
use navigate_match::{
    CorridorTileEntry, CorridorTileKey, CorridorTileStore, HadamardProjector, SparseFeatures,
};
use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
const LIVE_RENDER_SIZE: [u32; 2] = [800, 600];
const TILE_FIFO_CAPACITY: usize = 8;

/// Parameters controlling on-demand 3D landmark synthesis per corridor tile.
#[derive(Clone, Copy, Debug)]
pub struct TileSynthParams {
    /// Horizontal `ENU` tile edge length in meters.
    pub tile_m: f64,
    /// Nominal height above ground level in meters.
    pub agl_m: f64,
    /// Maximum deduplicated landmarks stored per tile.
    pub max_per_tile: usize,
    /// Coarse stride for fallback adaptive CPU ray-marching.
    pub stride: u32,
    /// Include canonical oblique rings alongside the nadir view.
    pub include_obliques: bool,
}

pub(crate) fn live_k_render() -> Matrix3<f64> {
    Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0)
}

/// Synthesize 3D landmark entries for `key` using shared GPU rendering or adaptive CPU fallback.
///
/// # Errors
///
/// Returns [`FlightVisualError`] when feature extraction or tile encoding fails.
pub fn synthesize_corridor_tiles(
    surface: &Surface,
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    xfeat: &LoadedXFeat,
    projector: &HadamardProjector,
    key: CorridorTileKey,
    params: TileSynthParams,
) -> Result<Vec<CorridorTileEntry>, FlightVisualError> {
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
    let voxel_m = (tile_m / 400.0).clamp(1.5, 8.0);
    let max_oblique_dist_m = (params.agl_m * 18.0).max(tile_m * 5.0).max(6_000.0);
    let mut tile_buckets: BTreeMap<CorridorTileKey, BandVoxelMap> = BTreeMap::new();

    let mut rendered = Vec::with_capacity(views.len());
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
        let gray_img = if let Some(ref gv) = gpu_view_opt {
            gv.image.clone()
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
            DynamicImage::ImageRgb8(rgb).into_luma8()
        };
        rendered.push((band, eye, rot, gray_img, gpu_view_opt));
    }

    let mut batch_feats = Vec::with_capacity(rendered.len());
    for chunk in rendered.chunks(16) {
        let pairs: Vec<(&image::GrayImage, Option<&image::GrayImage>)> =
            chunk.iter().map(|(_, _, _, img, _)| (img, None)).collect();
        batch_feats.extend(xfeat.extract_batch_masked(&pairs, 1536)?);
    }

    for ((band, eye, rot, gray_img, gpu_view_opt), raw) in rendered.into_iter().zip(batch_feats) {
        let world_hits: Vec<Option<[f64; 3]>> = if let Some(ref gv) = gpu_view_opt {
            gv.unproject_pixels(surface, &raw.pixels)
        } else {
            surface
                .raycast_pixels(eye, &rot, &k_render, &raw.pixels)
                .into_iter()
                .map(|h| h.map(|x| x.world_enu))
                .collect()
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
                        VoxelLandmark {
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

        if !params.include_obliques
            && is_nadir_pose(&rot)
            && let Ok(rot_bands) =
                synthesize_rotated_bands(&gray_img, surface, clat, clon, xfeat, 1536, false)
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

pub(crate) struct TileBuilderContext {
    pub surface: Arc<Surface>,
    pub shared_gpu: Option<Arc<SharedSurfaceGpu>>,
    pub store: Arc<CorridorTileStore>,
    pub projector: HadamardProjector,
    pub xfeat: LoadedXFeat,
    pub params: TileSynthParams,
}

pub(crate) struct TileBuildOutcome {
    pub entries: Vec<CorridorTileEntry>,
    pub elapsed_ms: f64,
}

/// Bounded FIFO background worker that synthesizes missing [`CorridorTileEntry`] cells on demand.
pub(crate) struct BackgroundTileBuilder {
    worker: BoundedFifoWorker<CorridorTileKey, Result<TileBuildOutcome, FlightVisualError>>,
    seen: HashSet<CorridorTileKey>,
    submit_virtual_ms: VecDeque<u64>,
    built_tiles: usize,
}

impl BackgroundTileBuilder {
    pub(crate) fn spawn(
        ctx: TileBuilderContext,
        prewarm_gpu: bool,
    ) -> Result<Self, FlightVisualError> {
        let init_gpu = prewarm_gpu
            .then(|| {
                if let Some(ref shared) = ctx.shared_gpu {
                    SurfaceGpuRenderer::from_shared(
                        shared,
                        &live_k_render(),
                        LIVE_RENDER_SIZE,
                        GpuQueueRole::BackgroundTileBuilder,
                    )
                } else {
                    SurfaceGpuRenderer::try_new_with_role(
                        &ctx.surface,
                        &live_k_render(),
                        LIVE_RENDER_SIZE,
                        GpuQueueRole::BackgroundTileBuilder,
                    )
                }
            })
            .flatten();
        let gpu_renderer = Mutex::new(init_gpu);
        let shared_gpu = ctx.shared_gpu.clone();
        let worker = BoundedFifoWorker::spawn(
            "corridor-tile-builder",
            TILE_FIFO_CAPACITY,
            SHUTDOWN_TIMEOUT,
            move |key: CorridorTileKey| {
                if ctx.store.has_tile(key) {
                    return Ok(TileBuildOutcome {
                        entries: Vec::new(),
                        elapsed_ms: 0.0,
                    });
                }
                if let Ok(mut guard) = gpu_renderer.lock()
                    && guard.is_none()
                    && let Some(ref shared) = shared_gpu
                {
                    *guard = SurfaceGpuRenderer::from_shared(
                        shared,
                        &live_k_render(),
                        LIVE_RENDER_SIZE,
                        GpuQueueRole::BackgroundTileBuilder,
                    );
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
                Ok(TileBuildOutcome {
                    entries,
                    elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
                })
            },
        )?;
        Ok(Self {
            worker,
            seen: HashSet::new(),
            submit_virtual_ms: VecDeque::with_capacity(TILE_FIFO_CAPACITY),
            built_tiles: 0,
        })
    }

    pub(crate) fn request_neighborhood(
        &mut self,
        surface: &Surface,
        store: &CorridorTileStore,
        lookahead_enu: [f64; 2],
        virtual_now_ms: u64,
    ) -> Result<(), FlightVisualError> {
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
            if self.submit_virtual_ms.len() >= 4 {
                break;
            }
            if self.worker.try_submit(key)? {
                let _ = self.seen.insert(key);
                self.submit_virtual_ms.push_back(virtual_now_ms);
            } else {
                break;
            }
        }
        Ok(())
    }

    pub(crate) fn poll_with_policy(
        &mut self,
        store: &CorridorTileStore,
        policy: ExecutionPolicy,
        virtual_now_ms: u64,
        samples: &mut Vec<f64>,
    ) -> Result<(), FlightVisualError> {
        match policy {
            ExecutionPolicy::LiveFlight => {
                while let Some(res) = self.worker.try_recv()? {
                    let _ = self.submit_virtual_ms.pop_front();
                    self.commit_outcome(store, res, samples)?;
                }
            }
            ExecutionPolicy::DeterministicReplay { tile_budget_ms, .. } => {
                while let Some(&submit_ms) = self.submit_virtual_ms.front() {
                    if virtual_now_ms < submit_ms.saturating_add(tile_budget_ms) {
                        break;
                    }
                    let _ = self.submit_virtual_ms.pop_front();
                    let res = self.worker.recv_blocking()?;
                    self.commit_outcome(store, res, samples)?;
                }
            }
        }
        Ok(())
    }

    pub(crate) fn drain_shutdown(
        &mut self,
        store: &CorridorTileStore,
        samples: &mut Vec<f64>,
    ) -> Result<(), FlightVisualError> {
        while !self.submit_virtual_ms.is_empty() {
            let _ = self.submit_virtual_ms.pop_front();
            let res = self.worker.recv_blocking()?;
            self.commit_outcome(store, res, samples)?;
        }
        Ok(())
    }

    fn commit_outcome(
        &mut self,
        store: &CorridorTileStore,
        res: Result<TileBuildOutcome, FlightVisualError>,
        samples: &mut Vec<f64>,
    ) -> Result<(), FlightVisualError> {
        let out = match res {
            Ok(v) => v,
            Err(err) => {
                self.worker.mark_faulted();
                return Err(err);
            }
        };
        if !out.entries.is_empty() {
            for entry in out.entries {
                store.insert_tile(entry, true)?;
                self.built_tiles = self.built_tiles.saturating_add(1);
            }
            samples.push(out.elapsed_ms);
        }
        Ok(())
    }

    pub(crate) const fn built_tiles(&self) -> usize {
        self.built_tiles
    }

    pub(crate) fn is_alive(&self) -> bool {
        self.worker.is_alive()
    }

    pub(crate) fn is_faulted(&self) -> bool {
        self.worker.is_faulted()
    }

    pub(crate) fn dropped_count(&self) -> u32 {
        self.worker.dropped_count()
    }
}
