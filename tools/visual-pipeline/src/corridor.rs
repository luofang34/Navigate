//! Tile-indexed 3D corridor landmark bank construction (`corridor-build`) and
//! `25-30 Hz` two-tier temporal corridor tracking (`corridor-track`).

mod admission;
mod anchor;
pub mod masking;
mod tracker;

pub use tracker::run_corridor_track;

use crate::{
    PipelineError,
    realtime::helpers::{hex, load_xfeat_model, raw_file_sha256},
    stages::{load_profile_cases, scale_k},
    surface::Surface,
    testset::PhotoCase,
};
use image::DynamicImage;
use nalgebra::Matrix3;
use navigate_match::{
    CorridorLandmarkBank, CorridorTileEntry, CorridorTileKey, DEFAULT_KASHIN_SEED,
    HadamardProjector, SparseFeatures,
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
    time::Instant,
};

/// Options that control offline [`CorridorLandmarkBank`] construction (`corridor-build`).
#[derive(Clone, Debug)]
pub struct CorridorBuildOptions {
    /// Primary terrain and orthoimagery `catalog.json` path.
    pub catalog_path: PathBuf,
    /// Optional case profile JSON (`with-prior.json`) to focus corridor bounds.
    pub profile_path: Option<PathBuf>,
    /// Path to `XFeat` `.safetensors` weights.
    pub xfeat_weights_path: PathBuf,
    /// Output `NVKTILE1` file path.
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
    /// Include four cardinal oblique views per tile in addition to the nadir view.
    pub include_obliques: bool,
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
    /// Input `NVKTILE1` corridor bank path.
    pub corridor_bank_path: PathBuf,
    /// Path to `XFeat` `.safetensors` weights.
    pub xfeat_weights_path: PathBuf,
    /// Optional path to `LighterGlue` `.safetensors` weights for anchor cross-attention.
    pub lighterglue_weights_path: Option<PathBuf>,
    /// Optional directory containing a `Segformer` checkpoint for amortized live masking.
    pub seg_model_dir: Option<PathBuf>,
    /// Semantic mask policy name (`"airframe-occlusions"` by default).
    pub mask_policy: String,
    /// Frame interval between full `Segformer` refreshes (`10` = `3 Hz` at `30 fps`).
    pub mask_interval: usize,
    /// Stage output directory (`run.json`, `{id}.json`, `{id}/candidate.npz`).
    pub output_dir: PathBuf,
    /// Frame interval between map-anchor refresh checks (`0` keeps warm track until loss).
    pub anchor_interval: usize,
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
    /// Run one parallel adaptive render and sparse-ray polish pass on map anchors.
    pub polish_anchors: bool,
    /// Optional `[lat_deg, lon_deg, alt_m]` and optional `heading_deg` prior override.
    pub prior_override: Option<([f64; 3], Option<f64>)>,
    /// Optional CIPA 35 mm equivalent focal length override in millimetres.
    pub focal_35mm_override: Option<f64>,
    /// Optional case ID allow-list.
    pub only_cases: Option<BTreeSet<String>>,
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

/// Build and save a [`CorridorLandmarkBank`] (`NVKTILE1`) across terrain corridor tiles.
///
/// # Errors
///
/// Returns [`PipelineError`] when catalog loading, feature extraction, or file writing fails.
pub fn build_corridor_bank(opts: &CorridorBuildOptions) -> Result<Value, PipelineError> {
    let t0 = Instant::now();
    let surface = Surface::from_catalog(&opts.catalog_path)?;
    let catalog_digest = raw_file_sha256(&opts.catalog_path)?;
    let (xfeat, encoder_digest) = load_xfeat_model(&opts.xfeat_weights_path)?;
    let projector = HadamardProjector::new(DEFAULT_KASHIN_SEED);

    let tile_m = opts.tile_size_m.clamp(100.0, 20_000.0);
    let agl_m = opts.nominal_agl_m.clamp(120.0, 8_000.0);
    let max_per_tile = opts.max_landmarks_per_tile.clamp(16, 65_536);
    let stride = opts.adaptive_stride.max(1);
    let cases = opts
        .profile_path
        .as_deref()
        .map(load_profile_cases)
        .transpose()?
        .unwrap_or_default();

    let tile_keys = select_corridor_tiles(&surface, &cases, tile_m, opts.corridor_radius_m);
    let render_size = [800_u32, 600_u32];
    let k_base = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
    let k_render = scale_k(&k_base, 1.0, 1.0);

    // Key: (tile_x, tile_y) -> view_band -> (voxel_x, voxel_y) -> RawLandmark
    let mut tile_buckets: BTreeMap<(i32, i32), BandVoxelMap> = BTreeMap::new();
    let mut canonical_views = 0_usize;
    let mut total_pixels = 0_usize;
    let mut total_rays = 0_usize;
    let voxel_m = 1.5_f64;
    let max_oblique_dist_m = (agl_m * 4.5).clamp(1_600.0, 2_400.0);

    for &(tx, ty) in &tile_keys {
        let ce = (f64::from(tx) + 0.5) * tile_m;
        let cn = (f64::from(ty) + 0.5) * tile_m;
        let [clat, clon, _] = surface.geodetic([ce, cn, 0.0]);
        let Some(ground_msl) = surface.sample_dem(clat, clon) else {
            continue;
        };

        let views = canonical_tile_poses(
            &surface,
            (ce, cn, ground_msl),
            tile_m,
            agl_m,
            opts.include_obliques,
        );
        for (band, eye, rot) in views {
            canonical_views += 1;
            let ((rgb, _, _), stats) =
                surface.render_adaptive(eye, &rot, &k_render, render_size, stride, 8, 0.025);
            total_pixels += stats.total_pixels;
            total_rays += stats.rays_cast;
            let gray = DynamicImage::ImageRgb8(rgb).into_luma8();
            let raw = pollster::block_on(xfeat.extract(&gray, 2048))?;
            let hits = surface.raycast_pixels(eye, &rot, &k_render, &raw.pixels);
            let chunks = raw.descriptors.as_chunks::<64>().0;
            let cam_x = [rot[(0, 0)] as f32, rot[(0, 1)] as f32, rot[(0, 2)] as f32];
            let n_raw = (hits.len().max(1)) as f32;

            for (idx, hit_opt) in hits.iter().enumerate() {
                let Some(hit) = hit_opt else {
                    continue;
                };
                let w = hit.world_enu;
                let dist_m =
                    ((eye[0] - w[0]).powi(2) + (eye[1] - w[1]).powi(2) + (eye[2] - w[2]).powi(2))
                        .sqrt();
                if opts.include_obliques && !(180.0..=max_oblique_dist_m).contains(&dist_m) {
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
                    .entry(band)
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
        }
    }

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
            &projector,
            &world_pts,
            (&obs_dirs, &cam_xs, &ref_dists),
            &feat,
        )?);
    }

    let bank = CorridorLandmarkBank::new(
        catalog_digest,
        xfeat.encoder_contract(),
        DEFAULT_KASHIN_SEED,
        tile_m as f32,
        tiles,
    )?;
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
    let ray_reduction = if total_rays > 0 {
        (total_pixels as f64) / (total_rays as f64)
    } else {
        1.0
    };
    Ok(json!({
        "corridor_bank_path": opts.output_bank_path.display().to_string(),
        "bank_bytes": encoded.len(),
        "xfeat_weights_sha256": hex(&encoder_digest),
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

fn select_corridor_tiles(
    surface: &Surface,
    cases: &[PhotoCase],
    tile_m: f64,
    radius_m: f64,
) -> Vec<(i32, i32)> {
    let sw = surface.local(surface.bbox[1], surface.bbox[0], 0.0);
    let ne = surface.local(surface.bbox[3], surface.bbox[2], 0.0);
    let cat_min_x = (sw[0].min(ne[0]) / tile_m).floor() as i32;
    let cat_max_x = (sw[0].max(ne[0]) / tile_m).floor() as i32;
    let cat_min_y = (sw[1].min(ne[1]) / tile_m).floor() as i32;
    let cat_max_y = (sw[1].max(ne[1]) / tile_m).floor() as i32;

    let mut selected = BTreeSet::new();
    let r = radius_m.max(0.0);
    for case in cases {
        if let Some(pos) = case.prior.as_ref().and_then(|p| p.position.as_ref()) {
            let p = surface.local(pos.latitude, pos.longitude, 0.0);
            let x0 = (((p[0] - r) / tile_m).floor() as i32).max(cat_min_x);
            let x1 = (((p[0] + r) / tile_m).floor() as i32).min(cat_max_x);
            let y0 = (((p[1] - r) / tile_m).floor() as i32).max(cat_min_y);
            let y1 = (((p[1] + r) / tile_m).floor() as i32).min(cat_max_y);
            for ty in y0..=y1 {
                for tx in x0..=x1 {
                    selected.insert((tx, ty));
                }
            }
        }
    }
    if selected.is_empty() {
        for ty in cat_min_y..=cat_max_y {
            for tx in cat_min_x..=cat_max_x {
                selected.insert((tx, ty));
            }
        }
    }
    selected.into_iter().collect()
}

fn canonical_tile_poses(
    surface: &Surface,
    (ce, cn, ground_msl): (f64, f64, f64),
    tile_m: f64,
    agl_m: f64,
    include_obliques: bool,
) -> Vec<(u8, [f64; 3], Matrix3<f64>)> {
    if !include_obliques {
        let [clat, clon, _] = surface.geodetic([ce, cn, 0.0]);
        let nadir_eye = surface.local(clat, clon, ground_msl + (tile_m * 0.9).max(agl_m));
        let nadir_rot = surface.camera_rotation(clat, clon, 0.0, 89.0, 0.0);
        return vec![(0, nadir_eye, nadir_rot)];
    }

    let mut out = Vec::with_capacity(24);
    let mut band = 0_u8;
    for down_deg in [26.0_f64, 50.0_f64] {
        let horiz_back = agl_m / down_deg.to_radians().tan();
        for step in 0..12_u32 {
            let heading_deg = f64::from(step) * 30.0;
            let h = heading_deg.to_radians();
            let eye_e = ce - h.sin() * horiz_back;
            let eye_n = cn - h.cos() * horiz_back;
            let [elat, elon, _] = surface.geodetic([eye_e, eye_n, 0.0]);
            let base_dem = surface.sample_dem(elat, elon).unwrap_or(ground_msl);
            let eye_alt = (ground_msl + agl_m).max(base_dem + 140.0);
            let eye = surface.local(elat, elon, eye_alt);
            let rot = surface.camera_rotation(elat, elon, heading_deg, down_deg, 0.0);
            out.push((band, eye, rot));
            band += 1;
        }
    }
    out
}

fn unit_obs_dir(eye_enu: [f64; 3], world_enu: [f64; 3]) -> [f32; 3] {
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
