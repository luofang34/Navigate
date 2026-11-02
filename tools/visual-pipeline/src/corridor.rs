//! Tile-indexed 3D corridor landmark bank construction (`corridor-build`) and
//! `25-30 Hz` two-tier temporal corridor tracking (`corridor-track`).

mod admission;
pub mod anchor;
pub mod anchor_worker;
mod background;
pub(crate) mod canonical;
pub mod kinematics;
mod latency;
pub mod masking;
mod motion;
pub mod nadir;
pub mod session;
mod tile_builder;
mod tracker;

pub(crate) use canonical::{canonical_tile_poses, select_corridor_tiles};

pub use anchor_worker::AnchorWorkerState;
pub use kinematics::{
    KinematicState, TrackingSearchStrategy, VehicleDynamicsProfile, can_update_vertical_velocity,
    compute_dt_s,
};
pub use session::{
    DeadReckonEstimate, DeadReckonPath, DualPathDeadReckoner, FilterIntegritySnapshot,
    SessionAnchorReport, VisualCorridorSession,
};
pub use tracker::run_corridor_track;

use crate::{
    PipelineError,
    realtime::helpers::{hex, raw_file_sha256},
    stages::{load_profile_cases, scale_k},
    surface::{Surface, SurfaceGpuRenderer},
    xfeat_runtime::{LoadedXFeat, XFeatRuntimeConfig},
};
use image::DynamicImage;
use nalgebra::Matrix3;
use navigate_match::{
    CorridorLandmarkBank, CorridorTileEntry, CorridorTileKey, CorridorTileStore,
    DEFAULT_KASHIN_SEED, DEFAULT_WINDOW_TILES, HadamardProjector, SparseFeatures,
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

/// Options that control offline [`CorridorLandmarkBank`] or sharded [`CorridorTileStore`] construction (`corridor-build`).
#[derive(Clone, Debug)]
pub struct CorridorBuildOptions {
    /// Primary terrain and orthoimagery `catalog.json` path.
    pub catalog_path: PathBuf,
    /// Optional case profile JSON (`with-prior.json`) to focus corridor bounds.
    pub profile_path: Option<PathBuf>,
    /// Path to `XFeat` `.safetensors` weights.
    pub xfeat_weights_path: PathBuf,
    /// Output `.nvktile` file path or sharded `.nvktiledir` directory path.
    pub output_bank_path: PathBuf,
    /// Horizontal ENU tile edge length in metres (`>= 100.0`).
    pub tile_size_m: f64,
    /// Horizontal radius around profile priors included in the corridor in metres.
    pub corridor_radius_m: f64,
    /// Nominal height above ground level for canonical oblique views in metres.
    pub nominal_agl_m: f64,
    /// Maximum deduplicated 3D landmarks stored per tile (`16..=32768`).
    pub max_landmarks_per_tile: usize,
    /// Coarse pixel stride for [`Surface::render_adaptive`].
    pub adaptive_stride: u32,
    /// Include canonical oblique rings alongside the nadir view per tile.
    pub include_obliques: bool,
    /// Sparse `XFeat` runtime configuration (`burn-f32`, `burn-f16`, `coreml`, `tensorrt`).
    pub xfeat_runtime: XFeatRuntimeConfig,
}

/// Options that control two-tier real-time corridor localization and `25-30 Hz` tracking (`corridor-track`).
#[derive(Clone, Debug)]
pub struct CorridorTrackOptions {
    /// Dataset root directory (`manifest.json`, `images/`, `masks/`).
    pub dataset_dir: PathBuf,
    /// Case profile JSON (`profiles/with-prior.json` or video testset profile).
    pub profile_path: PathBuf,
    /// Primary terrain and orthoimagery `catalog.json` path.
    pub catalog_path: PathBuf,
    /// Input `.nvktile` file path or sharded [`CorridorTileStore`] directory path.
    pub corridor_bank_path: PathBuf,
    /// Path to `XFeat` `.safetensors` weights.
    pub xfeat_weights_path: PathBuf,
    /// Optional path to `LighterGlue` `.safetensors` weights for anchor cross-attention.
    pub lighterglue_weights_path: Option<PathBuf>,
    /// Optional directory containing a `Segformer` checkpoint for amortized live masking.
    pub seg_model_dir: Option<PathBuf>,
    /// Optional path to `DINOv2` `.safetensors` weights for global place retrieval.
    pub dinov2_weights_path: Option<PathBuf>,
    /// Optional path to a `navigate-place` manifest, directory, or `.npls` shard for global retrieval.
    pub place_index_path: Option<PathBuf>,
    /// Vehicle dynamics constraints and platform parameter gates.
    pub dynamics_profile: VehicleDynamicsProfile,
    /// Semantic mask policy name (`"airframe-occlusions"` by default).
    pub mask_policy: String,
    /// Frame interval between full `Segformer` refreshes (`10` = `3 Hz` at `30 fps`).
    pub mask_interval: usize,
    /// Optional time interval in seconds between full `Segformer` refreshes.
    pub mask_interval_s: Option<f64>,
    /// Stage output directory (`run.json`, `{id}.json`, `{id}/candidate.npz`).
    pub output_dir: PathBuf,
    /// Frame interval between map-anchor refresh checks (`0` keeps warm track until loss).
    pub anchor_interval: usize,
    /// Optional time interval in seconds between map-anchor refresh checks.
    pub anchor_interval_s: Option<f64>,
    /// Optional horizontal uncertainty threshold in meters to trigger an anchor.
    pub anchor_uncertainty_m: Option<f64>,
    /// Trigger an anchor check when entering a new EKF independence map cell.
    pub anchor_on_cell_boundary: bool,
    /// Minimum PnP inliers required to accept a warm-tracked frame.
    pub min_track_inliers: usize,
    /// Horizontal edge length of each EKF independence [`navigate_visual_session::MapCell`] in metres.
    pub map_cell_m: f64,
    /// Maximum seed hypotheses evaluated during cold map-anchor acquisition.
    pub anchor_attempts: usize,
    /// Heading sweep step in degrees during cold map-anchor acquisition (`0.0` uses prior sweep).
    pub heading_step_deg: f64,
    /// Height multipliers evaluated during cold map-anchor acquisition.
    pub height_factors: Vec<f64>,
    /// Maximum `XFeat` keypoints extracted per query frame (`16..=4096`).
    pub query_keypoints: usize,
    /// Run one GPU (`wgpu-terrain-v1`) or adaptive CPU render and depth-unproject polish pass on map anchors.
    pub polish_anchors: bool,
    /// Optional `[lat_deg, lon_deg, alt_m]` and optional `heading_deg` prior override.
    pub prior_override: Option<([f64; 3], Option<f64>)>,
    /// Optional CIPA 35 mm equivalent focal length override in millimetres.
    pub focal_35mm_override: Option<f64>,
    /// Optional case ID allow-list.
    pub only_cases: Option<BTreeSet<String>>,
    /// Run `Segformer` refreshes, map anchors, and on-demand tile synthesis on background threads.
    pub background_refresh: bool,
    /// Pace offline frame replay to real-time capture timestamps (`case.time_s`).
    pub pace_realtime: bool,
    /// Sparse `XFeat` runtime configuration (`burn-f32`, `burn-f16`, `coreml`, `tensorrt`).
    pub xfeat_runtime: XFeatRuntimeConfig,
}

type BandVoxelMap = BTreeMap<u8, BTreeMap<(i32, i32), RawLandmark>>;

struct RawLandmark {
    world_enu: [f64; 3],
    obs_dir_enu: [f32; 3],
    cam_x_enu: [f32; 3],
    ref_dist_m: u16,
    score: f32,
    descriptor: [f32; 64],
}

/// Build and save a [`CorridorLandmarkBank`] (`.nvktile`) or sharded [`CorridorTileStore`] directory (`corridor-build`).
///
/// # Errors
///
/// Returns [`PipelineError`] when catalog loading, feature extraction, or file writing fails.
pub fn build_corridor_bank(opts: &CorridorBuildOptions) -> Result<Value, PipelineError> {
    let t0 = Instant::now();
    let surface = Surface::from_catalog(&opts.catalog_path)?;
    let catalog_digest = raw_file_sha256(&opts.catalog_path)?;
    let xfeat = LoadedXFeat::load(&opts.xfeat_weights_path, &opts.xfeat_runtime)?;
    let projector = HadamardProjector::new(DEFAULT_KASHIN_SEED);

    let tile_m = opts.tile_size_m.clamp(100.0, 20_000.0);
    let agl_m = opts.nominal_agl_m.clamp(25.0, 8_000.0);
    let max_per_tile = opts.max_landmarks_per_tile.clamp(16, 65_536);
    let stride = opts.adaptive_stride.max(1);
    let cases = opts
        .profile_path
        .as_deref()
        .map(load_profile_cases)
        .transpose()?
        .unwrap_or_default();

    let mut tile_keys =
        select_corridor_tiles(&surface, &cases, tile_m, opts.corridor_radius_m, agl_m);
    tile_keys.sort_by_key(|&(tx, ty)| ((tx.div_euclid(16), ty.div_euclid(16)), tx, ty));
    let prior_points_enu: Vec<[f64; 2]> = cases
        .iter()
        .filter_map(|c| c.prior.as_ref().and_then(|p| p.position.as_ref()))
        .map(|pos| {
            let p = surface.local(pos.latitude, pos.longitude, 0.0);
            [p[0], p[1]]
        })
        .collect();
    let dense_nadir_radius_sq = opts
        .corridor_radius_m
        .max(tile_m * 2.0)
        .max(agl_m * 3.0)
        .powi(2);
    let render_size = [800_u32, 600_u32];
    let k_base = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
    let k_render = scale_k(&k_base, 1.0, 1.0);
    let mut gpu_renderer = SurfaceGpuRenderer::try_new(&surface, &k_render, render_size);
    if gpu_renderer.is_none() {
        tracing::warn!(
            catalog_sha256 = %surface.catalog_sha256,
            "corridor-build GPU terrain renderer unavailable; using adaptive CPU raycasting"
        );
    }

    let mut tile_buckets: BTreeMap<(i32, i32), BandVoxelMap> = BTreeMap::new();
    let mut canonical_views = 0_usize;
    let mut total_pixels = 0_usize;
    let mut total_rays = 0_usize;
    let voxel_m = 1.5_f64;
    let max_oblique_dist_m = (agl_m * 14.0).max(4_000.0);

    for &(tx, ty) in &tile_keys {
        let ce = (f64::from(tx) + 0.5) * tile_m;
        let cn = (f64::from(ty) + 0.5) * tile_m;
        let [clat, clon, _] = surface.geodetic([ce, cn, 0.0]);
        let Some(ground_msl) = surface.sample_dem(clat, clon) else {
            continue;
        };
        let dense_nadir = prior_points_enu.is_empty()
            || prior_points_enu
                .iter()
                .any(|p| (ce - p[0]).powi(2) + (cn - p[1]).powi(2) <= dense_nadir_radius_sq);
        let views = canonical_tile_poses(
            &surface,
            (ce, cn, ground_msl),
            tile_m,
            agl_m,
            opts.include_obliques,
        );
        let mut rendered = Vec::with_capacity(views.len());
        for (band, eye, rot) in views {
            canonical_views += 1;
            let gpu_view_opt = gpu_renderer.as_mut().and_then(|gpu| {
                gpu.render_view(&surface, eye, &rot, &k_render, render_size)
                    .ok()
            });
            let gray_img = if let Some(ref gv) = gpu_view_opt {
                total_pixels += (render_size[0] as usize) * (render_size[1] as usize);
                gv.image.clone()
            } else {
                let ((rgb, _, _), stats) =
                    surface.render_adaptive(eye, &rot, &k_render, render_size, stride, 8, 0.025);
                total_pixels += stats.total_pixels;
                total_rays += stats.rays_cast;
                DynamicImage::ImageRgb8(rgb).into_luma8()
            };
            rendered.push((band, eye, rot, gray_img, gpu_view_opt));
        }

        let batch_pairs: Vec<(&image::GrayImage, Option<&image::GrayImage>)> = rendered
            .iter()
            .map(|(_, _, _, img, _)| (img, None))
            .collect();
        let batch_feats = xfeat.extract_batch_masked(&batch_pairs, 2048)?;

        for ((band, eye, rot, gray_img, gpu_view_opt), raw) in rendered.into_iter().zip(batch_feats)
        {
            let world_hits: Vec<Option<[f64; 3]>> = if let Some(ref gv) = gpu_view_opt {
                gv.unproject_pixels(&surface, &raw.pixels)
            } else {
                surface
                    .raycast_pixels(eye, &rot, &k_render, &raw.pixels)
                    .into_iter()
                    .map(|h| h.map(|x| x.world_enu))
                    .collect()
            };

            let mut insert_band =
                |band_idx: u8,
                 cam_x: [f32; 3],
                 features: &SparseFeatures,
                 world_hits: &[Option<[f64; 3]>]| {
                    let chunks = features.descriptors.as_chunks::<64>().0;
                    let n_raw = (world_hits.len().max(1)) as f32;

                    for (idx, w_opt) in world_hits.iter().enumerate() {
                        let Some(w) = *w_opt else {
                            continue;
                        };
                        let dist_m = ((eye[0] - w[0]).powi(2)
                            + (eye[1] - w[1]).powi(2)
                            + (eye[2] - w[2]).powi(2))
                        .sqrt();
                        if opts.include_obliques && !(20.0..=max_oblique_dist_m).contains(&dist_m) {
                            continue;
                        }
                        let dst_tx = (w[0] / tile_m).floor() as i32;
                        let dst_ty = (w[1] / tile_m).floor() as i32;
                        let vx = (w[0] / voxel_m).floor() as i32;
                        let vy = (w[1] / voxel_m).floor() as i32;
                        let score = (1.0 - (idx as f32) / n_raw).clamp(0.01, 1.0);
                        let band_map = tile_buckets
                            .entry((dst_tx, dst_ty))
                            .or_default()
                            .entry(band_idx)
                            .or_default();
                        if band_map
                            .get(&(vx, vy))
                            .is_none_or(|prev| score > prev.score)
                        {
                            band_map.insert(
                                (vx, vy),
                                RawLandmark {
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

            if !opts.include_obliques
                && nadir::is_nadir_pose(&rot)
                && let Ok(rot_bands) = nadir::synthesize_rotated_bands(
                    &gray_img,
                    &surface,
                    clat,
                    clon,
                    &xfeat,
                    2048,
                    dense_nadir,
                )
            {
                for (b_idx, r_band) in rot_bands.into_iter().enumerate() {
                    let r_band_idx = band.saturating_add((b_idx as u8) + 1);
                    let r_world_hits: Vec<Option<[f64; 3]>> = if let Some(ref gv) = gpu_view_opt {
                        gv.unproject_pixels(&surface, &r_band.unrotated_pixels)
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
    }

    let tiles = assemble_tiles(tile_buckets, tile_m, max_per_tile, &projector)?;
    let bank = CorridorLandmarkBank::new(
        catalog_digest,
        xfeat.encoder_contract(),
        DEFAULT_KASHIN_SEED,
        tile_m as f32,
        tiles,
    )?;

    let sharded = should_write_sharded_dir(&opts.output_bank_path);
    let bank_bytes = if sharded {
        let _store =
            CorridorTileStore::write_bank_dir(&bank, &opts.output_bank_path, DEFAULT_WINDOW_TILES)?;
        bank.to_bytes().len()
    } else {
        if let Some(parent) = opts.output_bank_path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|source| PipelineError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let encoded = bank.to_bytes();
        fs::write(&opts.output_bank_path, &encoded).map_err(|source| PipelineError::Io {
            path: opts.output_bank_path.clone(),
            source,
        })?;
        encoded.len()
    };

    let ray_reduction = if total_rays > 0 {
        (total_pixels as f64) / (total_rays as f64)
    } else {
        1.0
    };
    Ok(json!({
        "corridor_bank_path": opts.output_bank_path.display().to_string(),
        "sharded_store": sharded,
        "bank_bytes": bank_bytes,
        "xfeat_runtime": xfeat.runtime_name(),
        "xfeat_weights_sha256": hex(xfeat.weights_sha256()),
        "xfeat_encoder_contract": hex(&xfeat.encoder_contract()),
        "tile_size_m": tile_m,
        "tiles": bank.tiles.len(),
        "canonical_views": canonical_views,
        "landmarks_3d": bank.total_landmarks(),
        "adaptive_rays_cast": total_rays,
        "adaptive_ray_reduction": ray_reduction,
        "elapsed_ms": t0.elapsed().as_secs_f64() * 1000.0,
    }))
}

fn should_write_sharded_dir(path: &Path) -> bool {
    if path.is_dir() {
        return true;
    }
    match path.extension().and_then(|s| s.to_str()) {
        Some(ext) => {
            ext.eq_ignore_ascii_case("nvktiledir") || ext.eq_ignore_ascii_case("nvktstore")
        }
        None => true,
    }
}

fn assemble_tiles(
    tile_buckets: BTreeMap<(i32, i32), BandVoxelMap>,
    tile_m: f64,
    max_per_tile: usize,
    projector: &HadamardProjector,
) -> Result<Vec<CorridorTileEntry>, PipelineError> {
    let mut tiles = Vec::with_capacity(tile_buckets.len());
    for ((tx, ty), band_map) in tile_buckets {
        let num_bands = band_map.len().max(1);
        let per_band_cap = (max_per_tile / num_bands).max(64);
        let mut lms = Vec::new();
        for (_, vox_map) in band_map {
            let mut band_lms: Vec<RawLandmark> = vox_map.into_values().collect();
            band_lms.sort_by(|a, b| b.score.total_cmp(&a.score));
            band_lms.truncate(per_band_cap);
            lms.extend(band_lms);
        }
        if lms.is_empty() {
            continue;
        }
        lms.sort_by(|a, b| b.score.total_cmp(&a.score));
        lms.truncate(max_per_tile);
        let mean_up = lms.iter().map(|l| l.world_enu[2]).sum::<f64>() / (lms.len() as f64);
        let anchor_enu = [
            (f64::from(tx) + 0.5) * tile_m,
            (f64::from(ty) + 0.5) * tile_m,
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
        tiles.push(CorridorTileEntry::from_landmarks(
            CorridorTileKey {
                tile_x: tx,
                tile_y: ty,
            },
            anchor_enu,
            projector,
            &world_pts,
            (&obs_dirs, &cam_xs, &ref_dists),
            &feat,
        )?);
    }
    Ok(tiles)
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
