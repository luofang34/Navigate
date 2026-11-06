//! Shared resource loader (`FlightEngineResources`) and configuration (`FlightEngineConfig`).

use crate::{
    anchor::{AnchorSearchOpts, SeedPose},
    error::FlightVisualError,
    place_loader::load_search_tier_matcher,
    runtime::{LoadedMatcher, LoadedXFeat, MatcherRuntimeConfig, XFeatRuntimeConfig},
    surface::{GpuQueueRole, SharedSurfaceGpu, Surface, SurfaceGpuRenderer},
    types::ExecutionPolicy,
};
use nalgebra::Matrix3;
use navigate_locate::corridor::{VehicleDynamicsProfile, multitier::SearchTierMatcher};
use navigate_match::{CorridorTileStore, HadamardProjector};
use navigate_visual::{LocalFrame, fusion_bridge::VisualAttitudeBudget};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

/// Startup paths and cache limits for [`FlightEngineResources::open`].
#[derive(Clone, Debug)]
pub struct FlightEngineResourcePaths<'a> {
    /// Path to `catalog.json`.
    pub catalog_path: &'a Path,
    /// Path to the `NVKTILE1` file or `NVKTIDX1` directory.
    pub corridor_bank_path: &'a Path,
    /// Path to `XFeat` `.safetensors` weights.
    pub xfeat_weights_path: &'a Path,
    /// Runtime configuration for `XFeat`.
    pub xfeat_runtime: &'a XFeatRuntimeConfig,
    /// Optional path to `LighterGlue` `.safetensors` or `.onnx` weights.
    pub matcher_weights_path: Option<&'a Path>,
    /// Runtime configuration for `SparseMatcher`.
    pub matcher_runtime: &'a MatcherRuntimeConfig,
    /// Optional path to `DINOv2` `.safetensors` weights.
    pub dinov2_weights_path: Option<&'a Path>,
    /// Optional path to an `NVKPLAC1` place index.
    pub place_index_path: Option<&'a Path>,
    /// Maximum resident tiles retained in the LRU corridor cache (`9` by default).
    pub max_resident_tiles: usize,
    /// Compile neural and `wgpu` shader pipelines once during `open`.
    pub prewarm_gpu: bool,
}

/// Immutable terrain, tile store, neural runtimes, and shared `Arc<GpuScene>` resources.
pub struct FlightEngineResources {
    /// Loaded spherical terrain and orthoimagery surface.
    pub surface: Arc<Surface>,
    /// Validated `ENU` tangent frame (`Wgs84Enu` or `BodyTangentEnu`) at the surface origin.
    pub local_frame: LocalFrame,
    /// Windowed LRU 3D landmark tile store.
    pub store: Arc<CorridorTileStore>,
    /// Deterministic Hadamard descriptor projector matching `store.seed`.
    pub projector: HadamardProjector,
    /// Hardware-portable `XFeat` sparse extractor.
    pub xfeat: LoadedXFeat,
    /// Optional hardware-portable `SparseMatcher` (`Burn` `LighterGlue` or `ORT`).
    pub matcher: Option<LoadedMatcher>,
    /// Optional multi-tier `DINOv2` place matcher.
    pub search_tier_matcher: Option<Arc<SearchTierMatcher>>,
    /// Shared GPU scene textures (`Arc<GpuScene>`) reused across all viewports.
    pub shared_gpu: Option<Arc<SharedSurfaceGpu>>,
    /// True when shader pipelines were compiled at resource open time.
    pub prewarmed: bool,
    /// Startup shader warmup duration in milliseconds.
    pub startup_warmup_ms: f64,
}

impl FlightEngineResources {
    /// Load and cross-validate terrain, corridor bank, neural models, and shared GPU scene.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] when local frame validation, contract checks, or I/O fails.
    pub fn open(paths: &FlightEngineResourcePaths<'_>) -> Result<Self, FlightVisualError> {
        let surface = Arc::new(Surface::from_catalog(paths.catalog_path)?);
        let local_frame = surface.local_frame()?;
        let catalog_bytes =
            fs::read(paths.catalog_path).map_err(|source| FlightVisualError::Io {
                path: paths.catalog_path.to_path_buf(),
                source,
            })?;
        let catalog_digest: [u8; 32] = Sha256::digest(&catalog_bytes).into();
        let xfeat = LoadedXFeat::load(paths.xfeat_weights_path, paths.xfeat_runtime)?;
        let matcher = paths
            .matcher_weights_path
            .map(|p| LoadedMatcher::load(p, xfeat.device(), paths.matcher_runtime))
            .transpose()?;
        let store = Arc::new(CorridorTileStore::open(
            paths.corridor_bank_path,
            paths.max_resident_tiles.max(1),
        )?);
        if store.catalog_sha256 != catalog_digest {
            return Err(FlightVisualError::Invalid(
                "corridor bank catalog_sha256 does not match catalog.json".into(),
            ));
        }
        if !xfeat.accepts_encoder_contract(&store.encoder_contract) {
            return Err(FlightVisualError::Invalid(
                "corridor bank encoder_contract does not match loaded XFeat weights".into(),
            ));
        }
        let search_tier_matcher = load_search_tier_matcher(
            paths.dinov2_weights_path,
            paths.place_index_path,
            Arc::clone(&store),
            xfeat.device(),
        )?;
        let projector = HadamardProjector::new(store.seed);
        let shared_gpu = SharedSurfaceGpu::try_new(&surface);
        let mut startup_warmup_ms = 0.0;
        if paths.prewarm_gpu {
            let t0 = Instant::now();
            let _ = xfeat.warmup(matcher.as_ref().map(LoadedMatcher::as_matcher))?;
            if let Some(ref shared) = shared_gpu {
                let k0 = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
                if let Some(mut g) = SurfaceGpuRenderer::from_shared(
                    shared,
                    &k0,
                    [800, 600],
                    GpuQueueRole::ForegroundTracking,
                ) {
                    let lat = surface.origin_lat;
                    let lon = surface.origin_lon;
                    let dem = surface.sample_dem(lat, lon).unwrap_or(surface.min_elev_m);
                    let eye = surface.local(lat, lon, dem + 200.0);
                    let rot = surface.camera_rotation(lat, lon, 0.0, 45.0, 0.0);
                    drop(g.render_view(&surface, eye, &rot, &k0, [800, 600]));
                }
            }
            startup_warmup_ms = t0.elapsed().as_secs_f64() * 1000.0;
        }
        Ok(Self {
            surface,
            local_frame,
            store,
            projector,
            xfeat,
            matcher,
            search_tier_matcher,
            shared_gpu,
            prewarmed: paths.prewarm_gpu,
            startup_warmup_ms,
        })
    }

    /// Construct [`FlightEngineResources`] from in-memory components without disk I/O.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError::MissingLocalFrame`] when `surface.local_frame()` fails.
    pub fn from_parts(
        surface: Arc<Surface>,
        store: Arc<CorridorTileStore>,
        xfeat: LoadedXFeat,
        matcher: Option<LoadedMatcher>,
        search_tier_matcher: Option<Arc<SearchTierMatcher>>,
        shared_gpu: Option<Arc<SharedSurfaceGpu>>,
        prewarmed: bool,
    ) -> Result<Self, FlightVisualError> {
        let local_frame = surface.local_frame()?;
        let projector = HadamardProjector::new(store.seed);
        Ok(Self {
            surface,
            local_frame,
            store,
            projector,
            xfeat,
            matcher,
            search_tier_matcher,
            shared_gpu,
            prewarmed,
            startup_warmup_ms: 0.0,
        })
    }

    /// Return the resident GPU texture allocation in bytes for the shared scene atlases.
    #[must_use]
    pub fn shared_gpu_texture_bytes(&self) -> usize {
        self.shared_gpu
            .as_ref()
            .map_or(0, |s| s.resident_texture_bytes())
    }
}

/// Per-session configuration for [`crate::FlightVisualEngine`].
#[derive(Clone, Debug)]
pub struct FlightEngineConfig {
    /// Worker scheduling policy (`LiveFlight` vs `DeterministicReplay`).
    pub policy: ExecutionPolicy,
    /// Map-anchor seed and refinement configuration.
    pub search_opts: AnchorSearchOpts,
    /// Platform dynamics profile (`rotary-survey`, `fixed-wing`, or `high-performance`).
    pub dynamics_profile: VehicleDynamicsProfile,
    /// Map-cell edge size in meters for EKF independence gating.
    pub map_cell_m: f64,
    /// Minimum PnP inliers required to accept a foreground warm track.
    pub min_track_inliers: usize,
    /// Periodic map-anchor interval in frames (`0` disables frame-count refresh).
    pub anchor_interval_frames: usize,
    /// Periodic map-anchor interval in seconds.
    pub anchor_interval_s: Option<f64>,
    /// Trigger a background map anchor when crossing into an unvisited map cell.
    pub anchor_on_cell_boundary: bool,
    /// Spawn bounded background workers for map anchoring and tile synthesis.
    pub background_workers: bool,
    /// Synthesize missing corridor tiles on demand ahead of the vehicle.
    pub enable_live_tile_synth: bool,
    /// Prewarm the tile-builder GPU viewport at session startup.
    pub prewarm_tile_gpu: bool,
    /// Optional `Segformer` checkpoint directory for airframe/sky masking.
    pub seg_model_dir: Option<PathBuf>,
    /// Semantic mask policy string (`airframe-occlusions` by default).
    pub mask_policy: String,
    /// Refresh interval in frames for `AmortizedMasker`.
    pub mask_interval_frames: usize,
    /// Initial render dimensions `[width, height]` for GPU viewport warmup.
    pub init_render_size: [u32; 2],
    /// Initial render calibration matrix `K` for GPU viewport warmup.
    pub init_k_render: Matrix3<f64>,
    /// Optional initial camera pose `(eye_enu, world_to_cam)` to warm the tile cache and GPU.
    pub init_prior_pose: Option<SeedPose>,
    /// Optional mounting and attitude budget for 6-DoF `ObservationValue::VisualPose` fixes.
    pub attitude_budget: Option<VisualAttitudeBudget>,
}

impl Default for FlightEngineConfig {
    fn default() -> Self {
        Self {
            policy: ExecutionPolicy::LiveFlight,
            search_opts: AnchorSearchOpts::default(),
            dynamics_profile: VehicleDynamicsProfile::rotary_survey(),
            map_cell_m: 400.0,
            min_track_inliers: 24,
            anchor_interval_frames: 90,
            anchor_interval_s: Some(3.0),
            anchor_on_cell_boundary: true,
            background_workers: true,
            enable_live_tile_synth: true,
            prewarm_tile_gpu: false,
            seg_model_dir: None,
            mask_policy: "airframe-occlusions".into(),
            mask_interval_frames: 10,
            init_render_size: [800, 600],
            init_k_render: Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0),
            init_prior_pose: None,
            attitude_budget: None,
        }
    }
}
