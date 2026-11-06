//! Offline 3D corridor landmark bank builder with scale-adaptive voxels and cross-tile batching.

use crate::{
    canonical::{PriorWaypoint, canonical_tile_poses, select_corridor_tiles_from_priors},
    error::FlightVisualError,
    nadir::{
        DENSE_NADIR_YAW_BANDS_DEG, NADIR_DOWN_DEG, NADIR_YAW_BANDS_DEG, cached_800x600_masks,
        is_nadir_pose, rotate_image_gray, rotated_valid_mask, unrotate_pixel,
    },
    runtime::{LoadedXFeat, XFeatRuntimeConfig},
    surface::{GpuSurfaceView, Surface, SurfaceGpuRenderer},
    tile_assemble::{BandVoxelMap, VoxelLandmark, build_tile_entry_from_bands, unit_obs_dir},
};
use image::{DynamicImage, GrayImage};
use nalgebra::Matrix3;
use navigate_match::{
    CorridorLandmarkBank, CorridorTileEntry, CorridorTileKey, CorridorTileStore,
    DEFAULT_KASHIN_SEED, DEFAULT_WINDOW_TILES, HadamardProjector, SparseFeatures,
};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

const CROSS_TILE_BATCH_SIZE: usize = 16;

enum PatchKind {
    Base {
        view_idx: usize,
        band: u8,
        cam_x: [f32; 3],
    },
    RotatedNadir {
        view_idx: usize,
        band: u8,
        angle_deg: f64,
        cam_x: [f32; 3],
    },
}

struct RenderedBaseView {
    eye: [f64; 3],
    rot: Matrix3<f64>,
    gpu_view: Option<GpuSurfaceView>,
}

/// Configuration for building a [`CorridorLandmarkBank`] or sharded [`CorridorTileStore`].
#[derive(Clone, Debug)]
pub struct CorridorBuildSpec {
    /// Path to `catalog.json`.
    pub catalog_path: PathBuf,
    /// Optional waypoint priors to focus corridor tile selection.
    pub priors: Vec<PriorWaypoint>,
    /// Path to `XFeat` `.safetensors` weights.
    pub xfeat_weights_path: PathBuf,
    /// Output `.nvktile` file or sharded `.nvktiledir` directory path.
    pub output_bank_path: PathBuf,
    /// Horizontal `ENU` tile edge length in meters (`>= 100.0`).
    pub tile_size_m: f64,
    /// Horizontal corridor radius around priors in meters.
    pub corridor_radius_m: f64,
    /// Nominal height above ground level in meters.
    pub nominal_agl_m: f64,
    /// Maximum deduplicated 3D landmarks stored per tile.
    pub max_landmarks_per_tile: usize,
    /// Adaptive CPU ray-march stride when GPU rendering is unavailable.
    pub adaptive_stride: u32,
    /// Include canonical oblique view rings alongside nadir views.
    pub include_obliques: bool,
    /// Sparse `XFeat` runtime configuration.
    pub xfeat_runtime: XFeatRuntimeConfig,
}

/// Summary statistics emitted by [`build_corridor_bank_with_priors`].
#[derive(Clone, Debug)]
pub struct CorridorBuildStats {
    /// Output bank path string.
    pub corridor_bank_path: String,
    /// True when written as a sharded directory store.
    pub sharded_store: bool,
    /// Encoded bank byte length.
    pub bank_bytes: usize,
    /// Active `XFeat` runtime identifier.
    pub xfeat_runtime: String,
    /// SHA-256 digest of `XFeat` weights.
    pub xfeat_weights_sha256: [u8; 32],
    /// Encoder contract digest.
    pub xfeat_encoder_contract: [u8; 32],
    /// Tile edge size in meters.
    pub tile_size_m: f64,
    /// Built tile count.
    pub tiles: usize,
    /// Rendered canonical view count.
    pub canonical_views: usize,
    /// Total deduplicated 3D landmarks across all tiles.
    pub landmarks_3d: usize,
    /// CPU rays cast when adaptive fallback was used.
    pub adaptive_rays_cast: usize,
    /// Ray reduction ratio (`total_pixels / adaptive_rays_cast`).
    pub adaptive_ray_reduction: f64,
    /// Total build wall time in milliseconds.
    pub elapsed_ms: f64,
}

/// Build and write a [`CorridorLandmarkBank`] or sharded [`CorridorTileStore`] using cross-tile batching.
///
/// # Errors
///
/// Returns [`FlightVisualError`] when catalog loading, feature extraction, or file output fails.
pub fn build_corridor_bank_with_priors(
    spec: &CorridorBuildSpec,
) -> Result<CorridorBuildStats, FlightVisualError> {
    let t0 = Instant::now();
    let surface = Surface::from_catalog(&spec.catalog_path)?;
    let catalog_bytes = fs::read(&spec.catalog_path).map_err(|source| FlightVisualError::Io {
        path: spec.catalog_path.clone(),
        source,
    })?;
    let catalog_digest: [u8; 32] = Sha256::digest(&catalog_bytes).into();
    let xfeat = LoadedXFeat::load(&spec.xfeat_weights_path, &spec.xfeat_runtime)?;
    let projector = HadamardProjector::new(DEFAULT_KASHIN_SEED);

    let tile_m = spec.tile_size_m.clamp(100.0, 20_000.0);
    let agl_m = spec.nominal_agl_m.clamp(25.0, 8_000.0);
    let max_per_tile = spec.max_landmarks_per_tile.clamp(16, 65_536);
    let stride = spec.adaptive_stride.max(1);

    let mut tile_keys = select_corridor_tiles_from_priors(
        &surface,
        &spec.priors,
        tile_m,
        spec.corridor_radius_m,
        agl_m,
    );
    tile_keys.sort_by_key(|&(tx, ty)| ((tx.div_euclid(16), ty.div_euclid(16)), tx, ty));
    let prior_points_enu: Vec<[f64; 2]> = spec
        .priors
        .iter()
        .map(|wp| {
            let p = surface.local(wp.lat_deg, wp.lon_deg, 0.0);
            [p[0], p[1]]
        })
        .collect();
    let dense_nadir_radius_sq = spec
        .corridor_radius_m
        .max(tile_m * 2.0)
        .max(agl_m * 3.0)
        .powi(2);

    let render_size = [800_u32, 600_u32];
    let k_render = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
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
    let voxel_m = (tile_m / 400.0).clamp(1.5, 8.0);
    let max_oblique_dist_m = (agl_m * 18.0).max(tile_m * 5.0).max(6_000.0);

    let tile_group_len = if spec.include_obliques { 1 } else { 4 };
    for tile_group in tile_keys.chunks(tile_group_len) {
        let mut base_views: Vec<RenderedBaseView> = Vec::new();
        let mut patches: Vec<(GrayImage, Option<usize>, PatchKind)> = Vec::new();
        let mut custom_masks: Vec<GrayImage> = Vec::new();

        for &(tx, ty) in tile_group {
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
                spec.include_obliques,
            );
            for (band, eye, rot) in views {
                canonical_views += 1;
                let gpu_view = gpu_renderer.as_mut().and_then(|gpu| {
                    gpu.render_view(&surface, eye, &rot, &k_render, render_size)
                        .ok()
                });
                let gray_img = if let Some(ref gv) = gpu_view {
                    total_pixels += (render_size[0] as usize) * (render_size[1] as usize);
                    gv.image.clone()
                } else {
                    let ((rgb, _, _), stats) = surface.render_adaptive(
                        eye,
                        &rot,
                        &k_render,
                        render_size,
                        stride,
                        8,
                        0.025,
                    );
                    total_pixels += stats.total_pixels;
                    total_rays += stats.rays_cast;
                    DynamicImage::ImageRgb8(rgb).into_luma8()
                };
                let view_idx = base_views.len();
                let cam_x = [rot[(0, 0)] as f32, rot[(0, 1)] as f32, rot[(0, 2)] as f32];
                let want_rot_nadir = !spec.include_obliques && is_nadir_pose(&rot);

                if want_rot_nadir {
                    let angles: &[f64] = if dense_nadir {
                        &DENSE_NADIR_YAW_BANDS_DEG[1..]
                    } else {
                        &NADIR_YAW_BANDS_DEG[1..]
                    };
                    let cached = cached_800x600_masks(dense_nadir);
                    for (b_idx, &angle) in angles.iter().enumerate() {
                        let r_img = rotate_image_gray(&gray_img, angle);
                        let mask_idx = if gray_img.dimensions() == (800, 600) {
                            custom_masks.push(cached[b_idx].clone());
                            custom_masks.len() - 1
                        } else {
                            let (w, h) = gray_img.dimensions();
                            custom_masks.push(rotated_valid_mask(w, h, angle, 12.0));
                            custom_masks.len() - 1
                        };
                        let rot_angle =
                            surface.camera_rotation(clat, clon, angle, NADIR_DOWN_DEG, 0.0);
                        let r_cam_x = [
                            rot_angle[(0, 0)] as f32,
                            rot_angle[(0, 1)] as f32,
                            rot_angle[(0, 2)] as f32,
                        ];
                        patches.push((
                            r_img,
                            Some(mask_idx),
                            PatchKind::RotatedNadir {
                                view_idx,
                                band: band.saturating_add((b_idx as u8) + 1),
                                angle_deg: angle,
                                cam_x: r_cam_x,
                            },
                        ));
                    }
                }
                patches.push((
                    gray_img,
                    None,
                    PatchKind::Base {
                        view_idx,
                        band,
                        cam_x,
                    },
                ));
                base_views.push(RenderedBaseView { eye, rot, gpu_view });
            }
        }

        for chunk in patches.chunks(CROSS_TILE_BATCH_SIZE) {
            let pairs: Vec<(&GrayImage, Option<&GrayImage>)> = chunk
                .iter()
                .map(|(img, m_idx, _)| (img, m_idx.map(|i| &custom_masks[i])))
                .collect();
            let feats = xfeat.extract_batch_masked(&pairs, 2048)?;
            for ((img, _, kind), feat) in chunk.iter().zip(feats) {
                let (view_idx, band_idx, cam_x, query_px): (usize, u8, [f32; 3], Vec<[f64; 2]>) =
                    match *kind {
                        PatchKind::Base {
                            view_idx,
                            band,
                            cam_x,
                        } => (view_idx, band, cam_x, feat.pixels.clone()),
                        PatchKind::RotatedNadir {
                            view_idx,
                            band,
                            angle_deg,
                            cam_x,
                        } => {
                            let (w, h) = img.dimensions();
                            let unrot = feat
                                .pixels
                                .iter()
                                .map(|&p| unrotate_pixel(p, angle_deg, w, h))
                                .collect();
                            (view_idx, band, cam_x, unrot)
                        }
                    };
                let bv = &base_views[view_idx];
                let world_hits: Vec<Option<[f64; 3]>> = if let Some(ref gv) = bv.gpu_view {
                    gv.unproject_pixels(&surface, &query_px)
                } else {
                    surface
                        .raycast_pixels(bv.eye, &bv.rot, &k_render, &query_px)
                        .into_iter()
                        .map(|h| h.map(|x| x.world_enu))
                        .collect()
                };
                insert_landmarks(
                    &mut tile_buckets,
                    bv.eye,
                    (tile_m, voxel_m, max_oblique_dist_m, spec.include_obliques),
                    (band_idx, cam_x),
                    &feat,
                    &world_hits,
                );
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

    let sharded = should_write_sharded_dir(&spec.output_bank_path);
    let bank_bytes = if sharded {
        let _store =
            CorridorTileStore::write_bank_dir(&bank, &spec.output_bank_path, DEFAULT_WINDOW_TILES)?;
        bank.to_bytes().len()
    } else {
        if let Some(parent) = spec.output_bank_path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).map_err(|source| FlightVisualError::Io {
                path: parent.to_path_buf(),
                source,
            })?;
        }
        let encoded = bank.to_bytes();
        fs::write(&spec.output_bank_path, &encoded).map_err(|source| FlightVisualError::Io {
            path: spec.output_bank_path.clone(),
            source,
        })?;
        encoded.len()
    };

    let ray_reduction = if total_rays > 0 {
        (total_pixels as f64) / (total_rays as f64)
    } else {
        1.0
    };
    Ok(CorridorBuildStats {
        corridor_bank_path: spec.output_bank_path.display().to_string(),
        sharded_store: sharded,
        bank_bytes,
        xfeat_runtime: xfeat.runtime_name().to_owned(),
        xfeat_weights_sha256: *xfeat.weights_sha256(),
        xfeat_encoder_contract: xfeat.encoder_contract(),
        tile_size_m: tile_m,
        tiles: bank.tiles.len(),
        canonical_views,
        landmarks_3d: bank.total_landmarks(),
        adaptive_rays_cast: total_rays,
        adaptive_ray_reduction: ray_reduction,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
    })
}

fn insert_landmarks(
    tile_buckets: &mut BTreeMap<(i32, i32), BandVoxelMap>,
    eye: [f64; 3],
    (tile_m, voxel_m, max_oblique_dist_m, include_obliques): (f64, f64, f64, bool),
    (band_idx, cam_x): (u8, [f32; 3]),
    features: &SparseFeatures,
    world_hits: &[Option<[f64; 3]>],
) {
    let chunks = features.descriptors.as_chunks::<64>().0;
    let n_raw = (world_hits.len().max(1)) as f32;
    for (idx, w_opt) in world_hits.iter().enumerate() {
        let Some(w) = *w_opt else {
            continue;
        };
        let dist_m =
            ((eye[0] - w[0]).powi(2) + (eye[1] - w[1]).powi(2) + (eye[2] - w[2]).powi(2)).sqrt();
        if include_obliques && !(20.0..=max_oblique_dist_m).contains(&dist_m) {
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
) -> Result<Vec<CorridorTileEntry>, FlightVisualError> {
    let mut tiles = Vec::with_capacity(tile_buckets.len());
    for ((tx, ty), band_map) in tile_buckets {
        let key = CorridorTileKey {
            tile_x: tx,
            tile_y: ty,
        };
        if let Some(entry) =
            build_tile_entry_from_bands(key, tile_m, max_per_tile, projector, band_map)?
        {
            tiles.push(entry);
        }
    }
    Ok(tiles)
}
