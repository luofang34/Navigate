//! Command-line and `JSONL` adapters over `navigate-visual-flight`.

pub mod anchor;
mod latency;
pub mod serve;
mod tracker;

pub use navigate_locate::corridor::{
    kinematics,
    kinematics::{
        KinematicState, TrackingSearchStrategy, VehicleDynamicsProfile,
        can_update_vertical_velocity, compute_dt_s,
    },
    session,
    session::{
        AnchorWorkerState, DeadReckonEstimate, DeadReckonPath, DualPathDeadReckoner,
        FilterIntegritySnapshot, SessionAnchorReport, VisualCorridorSession,
    },
};
pub use navigate_visual_flight::{masking, nadir};
pub use serve::{CorridorServeOptions, CorridorServeSession, run_corridor_serve};
pub use tracker::run_corridor_track;

use crate::{
    PipelineError, realtime::bank_match::hex, stages::load_profile_cases,
    xfeat_runtime::XFeatRuntimeConfig,
};
use navigate_visual_flight::{
    CorridorBuildSpec, build_corridor_bank_with_priors, canonical::PriorWaypoint,
};
use serde_json::{Value, json};
use std::{collections::BTreeSet, path::PathBuf};

/// Options that control offline `NVKTILE1` or sharded `NVKTIDX1` construction (`corridor-build`).
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
    /// Horizontal `ENU` tile edge length in meters (`>= 100.0`).
    pub tile_size_m: f64,
    /// Horizontal radius around profile priors included in the corridor in meters.
    pub corridor_radius_m: f64,
    /// Nominal height above ground level for canonical oblique views in meters.
    pub nominal_agl_m: f64,
    /// Maximum deduplicated 3D landmarks stored per tile (`16..=65536`).
    pub max_landmarks_per_tile: usize,
    /// Coarse pixel stride for adaptive CPU raycasting fallback.
    pub adaptive_stride: u32,
    /// Include canonical oblique rings alongside nadir views per tile.
    pub include_obliques: bool,
    /// Sparse `XFeat` runtime configuration (`burn-f32`, `burn-f16`, `coreml`, `tensorrt`).
    pub xfeat_runtime: XFeatRuntimeConfig,
}

/// Options that control two-tier corridor localization and `30 Hz` tracking (`corridor-track`).
#[derive(Clone, Debug)]
pub struct CorridorTrackOptions {
    /// Dataset root directory (`manifest.json`, `images/`, `masks/`).
    pub dataset_dir: PathBuf,
    /// Case profile JSON (`profiles/with-prior.json` or video testset profile).
    pub profile_path: PathBuf,
    /// Primary terrain and orthoimagery `catalog.json` path.
    pub catalog_path: PathBuf,
    /// Input `.nvktile` file path or sharded `NVKTIDX1` directory path.
    pub corridor_bank_path: PathBuf,
    /// Path to `XFeat` `.safetensors` weights.
    pub xfeat_weights_path: PathBuf,
    /// Optional path to `LighterGlue` `.safetensors` or `.onnx` weights.
    pub lighterglue_weights_path: Option<PathBuf>,
    /// Optional directory containing a `Segformer` checkpoint for amortized live masking.
    pub seg_model_dir: Option<PathBuf>,
    /// Optional path to `DINOv2` `.safetensors` weights for global place retrieval.
    pub dinov2_weights_path: Option<PathBuf>,
    /// Optional path to a place index manifest, directory, or `.npls` shard.
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
    /// Minimum `PnP` inliers required to accept a warm-tracked frame.
    pub min_track_inliers: usize,
    /// Horizontal edge length of each EKF independence [`navigate_visual::session::MapCell`] in meters.
    pub map_cell_m: f64,
    /// Maximum seed hypotheses evaluated during cold map-anchor acquisition.
    pub anchor_attempts: usize,
    /// Heading sweep step in degrees during cold map-anchor acquisition (`0.0` uses prior sweep).
    pub heading_step_deg: f64,
    /// Height multipliers evaluated during cold map-anchor acquisition.
    pub height_factors: Vec<f64>,
    /// Maximum `XFeat` keypoints extracted per query frame (`16..=4096`).
    pub query_keypoints: usize,
    /// Run GPU (`wgpu-terrain-v1`) or adaptive CPU render polish on map anchors.
    pub polish_anchors: bool,
    /// Optional `[lat_deg, lon_deg, alt_m]` and optional `heading_deg` prior override.
    pub prior_override: Option<([f64; 3], Option<f64>)>,
    /// Optional CIPA 35 mm equivalent focal length override in millimeters.
    pub focal_35mm_override: Option<f64>,
    /// Optional case ID allow-list.
    pub only_cases: Option<BTreeSet<String>>,
    /// Run `Segformer` refreshes, map anchors, and on-demand tile synthesis on background threads.
    pub background_refresh: bool,
    /// Pace offline frame replay to real-time capture timestamps (`ExecutionPolicy::LiveFlight`).
    pub pace_realtime: bool,
    /// Sparse `XFeat` runtime configuration (`burn-f32`, `burn-f16`, `coreml`, `tensorrt`).
    pub xfeat_runtime: XFeatRuntimeConfig,
}

/// Build and write a `CorridorLandmarkBank` (`.nvktile`) or sharded `CorridorTileStore` directory.
///
/// # Errors
///
/// Returns [`PipelineError`] when catalog loading, feature extraction, or file writing fails.
pub fn build_corridor_bank(opts: &CorridorBuildOptions) -> Result<Value, PipelineError> {
    let cases = opts
        .profile_path
        .as_deref()
        .map(load_profile_cases)
        .transpose()?
        .unwrap_or_default();
    let priors: Vec<PriorWaypoint> = cases
        .iter()
        .filter_map(|c| {
            let p = c.prior.as_ref()?;
            let pos = p.position.as_ref()?;
            Some(PriorWaypoint {
                lat_deg: pos.latitude,
                lon_deg: pos.longitude,
                alt_msl_m: pos.altitude_m.filter(|a| a.is_finite()),
                heading_deg: p.direction_deg.filter(|d| d.is_finite()),
            })
        })
        .collect();

    let stats = build_corridor_bank_with_priors(&CorridorBuildSpec {
        catalog_path: opts.catalog_path.clone(),
        priors,
        xfeat_weights_path: opts.xfeat_weights_path.clone(),
        output_bank_path: opts.output_bank_path.clone(),
        tile_size_m: opts.tile_size_m,
        corridor_radius_m: opts.corridor_radius_m,
        nominal_agl_m: opts.nominal_agl_m,
        max_landmarks_per_tile: opts.max_landmarks_per_tile,
        adaptive_stride: opts.adaptive_stride,
        include_obliques: opts.include_obliques,
        xfeat_runtime: opts.xfeat_runtime.clone(),
    })?;

    Ok(json!({
        "corridor_bank_path": stats.corridor_bank_path,
        "sharded_store": stats.sharded_store,
        "bank_bytes": stats.bank_bytes,
        "xfeat_runtime": stats.xfeat_runtime,
        "xfeat_weights_sha256": hex(&stats.xfeat_weights_sha256),
        "xfeat_encoder_contract": hex(&stats.xfeat_encoder_contract),
        "tile_size_m": stats.tile_size_m,
        "tiles": stats.tiles,
        "canonical_views": stats.canonical_views,
        "landmarks_3d": stats.landmarks_3d,
        "adaptive_rays_cast": stats.adaptive_rays_cast,
        "adaptive_ray_reduction": stats.adaptive_ray_reduction,
        "elapsed_ms": stats.elapsed_ms,
    }))
}
