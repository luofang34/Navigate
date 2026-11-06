//! Zero-disk-IO `30 Hz` [`FlightVisualEngine`] lifecycle and worker supervision.

mod frame_step;
mod prep;
mod resources;
mod solve;
mod warm_track;

pub use resources::{FlightEngineConfig, FlightEngineResourcePaths, FlightEngineResources};

use crate::{
    anchor::FocalSweepEstimator,
    anchor_arrival::{ActiveAnchorStatus, AnchorApplyCtx, RecentAnchorBackup},
    anchor_worker::{AnchorContext, BackgroundAnchor},
    diagnostics::{AnchorArrivalRecord, FrameDiagnostics},
    error::FlightVisualError,
    masking::AmortizedMasker,
    runtime::LoadedMatcher,
    surface::{GpuQueueRole, SurfaceGpuRenderer},
    tile_builder::{BackgroundTileBuilder, TileBuilderContext, TileSynthParams},
    types::{ExecutionPolicy, FlightFrameOutcome, LiveFrameInput, WorkerHealth},
};
use navigate_locate::corridor::{CorridorTracker, FilterIntegritySnapshot};
use navigate_match::VirtualViewCamera;
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

/// Real-time `30 Hz` zero-disk-IO visual localization and corridor tracking engine.
pub struct FlightVisualEngine {
    pub(crate) resources: Arc<FlightEngineResources>,
    pub(crate) config: FlightEngineConfig,
    pub(crate) state: CorridorTracker,
    pub(super) session_configured: bool,
    pub(crate) focal_estimator: FocalSweepEstimator,
    pub(crate) masker: AmortizedMasker,
    pub(crate) sync_gpu_renderer: Mutex<Option<SurfaceGpuRenderer>>,
    pub(crate) background_anchor: Option<BackgroundAnchor>,
    pub(crate) tile_builder: Option<BackgroundTileBuilder>,
    pub(crate) tile_params: TileSynthParams,
    pub(crate) active_status: Option<ActiveAnchorStatus>,
    pub(crate) recent_backup: Option<RecentAnchorBackup>,
    pub(crate) independent_fixes: usize,
    pub(super) sync_tiles_built: usize,
    pub(super) vo_accum_dt_s: f64,
    shader_warmup_ms: f64,
}

impl FlightVisualEngine {
    /// Start a [`FlightVisualEngine`] over shared `resources` and `config`.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] when masker initialization, warmup, or worker spawn fails.
    pub fn new(
        resources: Arc<FlightEngineResources>,
        mut config: FlightEngineConfig,
    ) -> Result<Self, FlightVisualError> {
        config.search_opts.pace_realtime = matches!(config.policy, ExecutionPolicy::LiveFlight);
        let seg_device = burn::tensor::Device::flex();
        let masker = AmortizedMasker::new(
            config.seg_model_dir.as_deref(),
            &config.mask_policy,
            6,
            config.mask_interval_frames.max(1),
            &seg_device,
        )?;

        let mut shader_warmup_ms = 0.0;
        if !resources.prewarmed && config.background_workers {
            let t0 = Instant::now();
            if matches!(config.policy, ExecutionPolicy::LiveFlight) {
                let _ = resources
                    .xfeat
                    .warmup(resources.matcher.as_ref().map(LoadedMatcher::as_matcher))?;
            } else {
                let _ = resources.xfeat.warmup_xfeat_only(
                    config.init_render_size,
                    config.search_opts.query_keypoints.clamp(16, 4096),
                )?;
            }
            shader_warmup_ms = t0.elapsed().as_secs_f64() * 1000.0;
        }

        if let Some((eye, rot)) = config.init_prior_pose {
            let k = &config.init_k_render;
            drop(resources.store.project_virtual_kashin(&VirtualViewCamera {
                eye_enu: eye,
                world_to_cam: crate::anchor::nadir_screen::rot_to_row_major(&rot),
                intrinsics: [k[(0, 0)], k[(1, 1)], k[(0, 2)], k[(1, 2)]],
                image_size: config.init_render_size,
                max_landmarks: 256,
                min_cos_incidence: 0.45,
                min_cos_orientation: -1.0,
            }));
        }

        let sync_gpu = (!config.background_workers && config.search_opts.polish_anchors)
            .then(|| {
                resources
                    .shared_gpu
                    .as_ref()
                    .and_then(|shared| {
                        SurfaceGpuRenderer::from_shared(
                            shared,
                            &config.init_k_render,
                            config.init_render_size,
                            GpuQueueRole::ForegroundTracking,
                        )
                    })
                    .or_else(|| {
                        SurfaceGpuRenderer::try_new_with_role(
                            &resources.surface,
                            &config.init_k_render,
                            config.init_render_size,
                            GpuQueueRole::ForegroundTracking,
                        )
                    })
                    .map(|mut g| {
                        if let Some((eye, ref rot)) = config.init_prior_pose {
                            drop(g.render_view(
                                &resources.surface,
                                eye,
                                rot,
                                &config.init_k_render,
                                config.init_render_size,
                            ));
                        }
                        g
                    })
            })
            .flatten();

        let tile_params = TileSynthParams {
            tile_m: f64::from(resources.store.tile_size_m.max(10.0)),
            agl_m: 480.0,
            max_per_tile: 4096,
            stride: 4,
            include_obliques: !cfg!(debug_assertions),
        };
        let background_anchor = config
            .background_workers
            .then(|| {
                BackgroundAnchor::spawn(
                    AnchorContext {
                        surface: Arc::clone(&resources.surface),
                        store: Arc::clone(&resources.store),
                        projector: resources.projector,
                        xfeat: resources.xfeat.clone(),
                        matcher: resources.matcher.clone(),
                        search_tier_matcher: resources.search_tier_matcher.clone(),
                        search_opts: config.search_opts.clone(),
                    },
                    resources.shared_gpu.as_ref(),
                    &config.init_k_render,
                    config.init_render_size,
                    config.init_prior_pose,
                )
            })
            .transpose()?;
        let tile_builder = (config.background_workers && config.enable_live_tile_synth)
            .then(|| {
                BackgroundTileBuilder::spawn(
                    TileBuilderContext {
                        surface: Arc::clone(&resources.surface),
                        shared_gpu: resources.shared_gpu.clone(),
                        store: Arc::clone(&resources.store),
                        projector: resources.projector,
                        xfeat: resources.xfeat.clone(),
                        params: tile_params,
                    },
                    config.prewarm_tile_gpu,
                )
            })
            .transpose()?;

        Ok(Self {
            resources,
            config,
            state: CorridorTracker::new(),
            session_configured: false,
            focal_estimator: FocalSweepEstimator::default(),
            masker,
            sync_gpu_renderer: Mutex::new(sync_gpu),
            background_anchor,
            tile_builder,
            tile_params,
            active_status: None,
            recent_backup: None,
            independent_fixes: 0,
            sync_tiles_built: 0,
            vo_accum_dt_s: 0.0,
            shader_warmup_ms,
        })
    }

    /// Process one live camera frame without disk I/O or heap-allocated JSON diagnostics.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] on non-monotonic timestamps, invalid buffers, or worker faults.
    pub fn process_frame(
        &mut self,
        frame: &LiveFrameInput<'_>,
    ) -> Result<FlightFrameOutcome, FlightVisualError> {
        let (outcome, _) = self.process_frame_inner(frame, "live", false)?;
        Ok(outcome)
    }

    /// Process one camera frame and return typed [`FrameDiagnostics`] for CLI debriefing.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] on non-monotonic timestamps, invalid buffers, or worker faults.
    pub fn process_frame_with_diagnostics(
        &mut self,
        frame: &LiveFrameInput<'_>,
        frame_label: &str,
    ) -> Result<(FlightFrameOutcome, FrameDiagnostics), FlightVisualError> {
        let (outcome, diag_opt) = self.process_frame_inner(frame, frame_label, true)?;
        let diag = diag_opt.ok_or_else(|| {
            FlightVisualError::Invalid("diagnostics missing on diagnostic frame call".into())
        })?;
        Ok((outcome, diag))
    }

    /// Drain in-flight background anchor and tile jobs at the end of an offline replay session.
    ///
    /// # Errors
    ///
    /// Returns [`FlightVisualError`] when a background worker faulted during execution.
    pub fn shutdown_drain(
        &mut self,
    ) -> Result<(Vec<AnchorArrivalRecord>, Vec<f64>), FlightVisualError> {
        let mut arrivals = Vec::new();
        let mut tile_ms = Vec::new();
        let cell_m = self.config.map_cell_m.clamp(50.0, 20_000.0);
        if let Some(worker) = self.background_anchor.as_mut()
            && let Some(outcome) = worker.drain_shutdown()?
        {
            let mut ctx = AnchorApplyCtx {
                state: &mut self.state,
                active_status: &mut self.active_status,
                recent_backup: &mut self.recent_backup,
                independent_fixes: &mut self.independent_fixes,
                focal_estimator: &mut self.focal_estimator,
                catalog_sha256: &self.resources.surface.catalog_sha256,
                local_frame: self.resources.local_frame,
                cell_m,
                dynamics_profile: self.config.dynamics_profile,
            };
            let (_, rec) = ctx.apply(outcome, None);
            arrivals.push(rec);
        }
        if let Some(tb) = self.tile_builder.as_mut() {
            tb.drain_shutdown(&self.resources.store, &mut tile_ms)?;
        }
        Ok((arrivals, tile_ms))
    }

    /// Reset transient session state before serving an independent single-image request.
    pub fn reset_session(&mut self) {
        self.state = CorridorTracker::new();
        self.session_configured = false;
        self.focal_estimator = FocalSweepEstimator::default();
        self.masker.reset_session();
        self.active_status = None;
        self.recent_backup = None;
        self.independent_fixes = 0;
        self.sync_tiles_built = 0;
        self.vo_accum_dt_s = 0.0;
    }

    /// Update synchronous anchor search options before a single-image `corridor-serve` request.
    pub fn set_search_opts(&mut self, mut search_opts: crate::anchor::AnchorSearchOpts) {
        search_opts.pace_realtime = matches!(self.config.policy, ExecutionPolicy::LiveFlight);
        self.config.search_opts = search_opts;
    }

    /// Cap inline `Segformer` input resolution on synchronous single-image engines.
    pub fn clamp_masker_input_side(&mut self, max_side: u32) {
        self.masker.clamp_input_side(max_side);
    }

    /// Return current watchdog health and bounded-queue counters.
    #[must_use]
    pub fn worker_health(&self) -> WorkerHealth {
        WorkerHealth {
            anchor_alive: self
                .background_anchor
                .as_ref()
                .is_some_and(BackgroundAnchor::is_alive),
            anchor_faulted: self
                .background_anchor
                .as_ref()
                .is_some_and(BackgroundAnchor::is_faulted),
            anchor_in_flight: self
                .background_anchor
                .as_ref()
                .is_some_and(BackgroundAnchor::is_in_flight),
            anchor_superseded: self
                .background_anchor
                .as_ref()
                .map_or(0, BackgroundAnchor::superseded_count),
            anchor_dropped: self
                .background_anchor
                .as_ref()
                .map_or(0, BackgroundAnchor::dropped_count),
            tile_alive: self
                .tile_builder
                .as_ref()
                .is_some_and(BackgroundTileBuilder::is_alive),
            tile_faulted: self
                .tile_builder
                .as_ref()
                .is_some_and(BackgroundTileBuilder::is_faulted),
            tile_dropped: self
                .tile_builder
                .as_ref()
                .map_or(0, BackgroundTileBuilder::dropped_count),
            mask_superseded: self.masker.superseded_count(),
        }
    }

    /// Return total live corridor tiles synthesized across foreground and background builders.
    #[must_use]
    pub fn tiles_built_live(&self) -> usize {
        self.sync_tiles_built
            + self
                .tile_builder
                .as_ref()
                .map_or(0, BackgroundTileBuilder::built_tiles)
    }

    /// Return total independent map-cell fixes admitted in this session.
    #[must_use]
    pub const fn independent_fixes(&self) -> usize {
        self.independent_fixes
    }

    /// Return the startup shader warmup duration in milliseconds.
    #[must_use]
    pub const fn shader_warmup_ms(&self) -> f64 {
        self.shader_warmup_ms
    }

    /// Return the locked full-frame pixel focal length if estimated by an uncalibrated sweep.
    #[must_use]
    pub const fn locked_focal_px(&self) -> Option<f64> {
        self.focal_estimator.locked_focal_px
    }

    /// Refresh and return the current navigation filter integrity snapshot.
    pub fn filter_snapshot(&mut self) -> FilterIntegritySnapshot {
        self.state.filter_snapshot(Some(self.resources.local_frame))
    }
}
