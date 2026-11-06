//! Persistent JSON-Lines (`JSONL`) worker session (`corridor-serve`) over [`FlightVisualEngine`].

mod request;

use super::{
    VehicleDynamicsProfile,
    latency::StageLatencies,
    tracker::{
        apply_case_overrides, build_live_frame_input, build_search_opts,
        matcher_runtime_for_options, mount_profile_for_dynamics,
        reporting::{
            RunSummaryParams, build_run_summary, emit_corridor_case_report, record_frame_telemetry,
        },
        run_corridor_track_with_resources,
    },
};
use crate::{
    PipelineError,
    json_file::write_json,
    stages::{focal::FocalLengthEstimator, load_case_mask, load_case_rgb},
    testset::PhotoCase,
    xfeat_runtime::XFeatRuntimeConfig,
};
use nalgebra::Matrix3;
use navigate_match::DEFAULT_WINDOW_TILES;
use navigate_visual_flight::{
    ExecutionPolicy, FlightEngineConfig, FlightEngineResourcePaths, FlightEngineResources,
    FlightVisualEngine, FlightVisualError, GpuQueueRole, LoadedMatcher, SurfaceGpuRenderer,
    TrackingMode,
};
use request::ServeRequest;
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, Write},
    path::{Path, PathBuf},
    sync::Arc,
    thread::{self, JoinHandle},
    time::Instant,
};

/// Startup configuration for a resident `corridor-serve` worker session.
#[derive(Clone, Debug)]
pub struct CorridorServeOptions {
    /// Path to the primary terrain and orthoimagery `catalog.json`.
    pub catalog_path: PathBuf,
    /// Path to the `NVKTILE1` file or sharded `NVKTIDX1` directory.
    pub corridor_bank_path: PathBuf,
    /// Path to `XFeat` `.safetensors` weights.
    pub xfeat_weights_path: PathBuf,
    /// Optional path to `LighterGlue` `.safetensors` or `.onnx` weights.
    pub lighterglue_weights_path: Option<PathBuf>,
    /// Optional path to a `Segformer` checkpoint directory.
    pub seg_model_dir: Option<PathBuf>,
    /// Optional path to `DINOv2` `.safetensors` weights for global fallback.
    pub dinov2_weights_path: Option<PathBuf>,
    /// Optional path to an `NVKPLAC1` place index.
    pub place_index_path: Option<PathBuf>,
    /// Semantic mask policy name (`"airframe-occlusions"` by default).
    pub mask_policy: String,
    /// Default vehicle dynamics profile for requests.
    pub dynamics_profile: VehicleDynamicsProfile,
    /// Default query keypoint cap per frame.
    pub query_keypoints: usize,
    /// Enable orthophoto polish by default on requests.
    pub polish_anchors: bool,
    /// Sparse `XFeat` extractor runtime configuration.
    pub xfeat_runtime: XFeatRuntimeConfig,
}

/// Resident corridor localization worker reusing [`FlightEngineResources`] and [`FlightVisualEngine`].
pub struct CorridorServeSession {
    opts: CorridorServeOptions,
    resources: Arc<FlightEngineResources>,
    single_engine: FlightVisualEngine,
    warmup_handle: Option<JoinHandle<Result<f64, FlightVisualError>>>,
    warmup_ms: f64,
    requests_served: u64,
}

impl CorridorServeSession {
    /// Open a resident worker session and start asynchronous GPU pipeline warmup.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when catalog loading, bank verification, or engine setup fails.
    pub fn open(opts: CorridorServeOptions) -> Result<Self, PipelineError> {
        let matcher_rt = matcher_runtime_for_options(
            opts.lighterglue_weights_path.as_deref(),
            &opts.xfeat_runtime,
        );
        let resources = Arc::new(FlightEngineResources::open(&FlightEngineResourcePaths {
            catalog_path: &opts.catalog_path,
            corridor_bank_path: &opts.corridor_bank_path,
            xfeat_weights_path: &opts.xfeat_weights_path,
            xfeat_runtime: &opts.xfeat_runtime,
            matcher_weights_path: opts.lighterglue_weights_path.as_deref(),
            matcher_runtime: &matcher_rt,
            dinov2_weights_path: opts.dinov2_weights_path.as_deref(),
            place_index_path: opts.place_index_path.as_deref(),
            max_resident_tiles: DEFAULT_WINDOW_TILES,
            prewarm_gpu: false,
        })?);

        let init_k = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
        let mut cfg = FlightEngineConfig {
            policy: ExecutionPolicy::DeterministicReplay {
                anchor_budget_ms: 200,
                tile_budget_ms: 150,
            },
            dynamics_profile: opts.dynamics_profile,
            seg_model_dir: opts.seg_model_dir.clone(),
            mask_policy: opts.mask_policy.clone(),
            mask_interval_frames: 10,
            background_workers: false,
            enable_live_tile_synth: resources.store.indexed_tile_count() == 0,
            prewarm_tile_gpu: false,
            init_render_size: [800, 600],
            init_k_render: init_k,
            init_prior_pose: None,
            ..FlightEngineConfig::default()
        };
        cfg.search_opts.query_keypoints = opts.query_keypoints.clamp(16, 4096);
        cfg.search_opts.polish_anchors = opts.polish_anchors;
        let mut single_engine = FlightVisualEngine::new(Arc::clone(&resources), cfg)?;
        single_engine.clamp_masker_input_side(256);

        let bg_res = Arc::clone(&resources);
        let warmup_handle = thread::Builder::new()
            .name("corridor-serve-warmup".into())
            .spawn(move || warmup_resources_async(&bg_res))
            .map_err(|source| PipelineError::Io {
                path: PathBuf::from("corridor-serve-warmup"),
                source,
            })?;

        Ok(Self {
            opts,
            resources,
            single_engine,
            warmup_handle: Some(warmup_handle),
            warmup_ms: 0.0,
            requests_served: 0,
        })
    }

    fn ensure_warmup_complete(&mut self) -> Result<(), PipelineError> {
        if let Some(handle) = self.warmup_handle.take() {
            let ms = handle.join().map_err(|_| {
                PipelineError::Invalid("corridor-serve warmup thread panicked".into())
            })??;
            self.warmup_ms = ms;
        }
        Ok(())
    }

    /// Build the startup handshake JSON value for the worker stream.
    #[must_use]
    pub fn ready_banner(&self) -> Value {
        json!({
            "status": "ready",
            "catalog_sha256": self.resources.surface.catalog_sha256,
            "xfeat_runtime": self.resources.xfeat.runtime_name(),
            "indexed_tiles": self.resources.store.indexed_tile_count(),
            "shared_gpu_texture_bytes": self.resources.shared_gpu_texture_bytes(),
            "shader_warmup_ms": self.resources.startup_warmup_ms,
        })
    }

    /// Execute one JSON request line and return the response JSON value plus a shutdown flag.
    pub fn handle_line(&mut self, raw_line: &str) -> (Value, bool) {
        let req: ServeRequest = match serde_json::from_str(raw_line) {
            Ok(r) => r,
            Err(err) => {
                return (
                    json!({ "ok": false, "error": format!("invalid JSON request: {err}") }),
                    false,
                );
            }
        };
        let req_id = req.request_id.clone();
        match req.op.as_deref() {
            Some("ping") => (
                json!({ "ok": true, "request_id": req_id, "status": "pong" }),
                false,
            ),
            Some("shutdown" | "quit" | "exit") => {
                drop(self.ensure_warmup_complete());
                (
                    json!({
                        "ok": true,
                        "request_id": req_id,
                        "status": "shutdown",
                        "requests_served": self.requests_served,
                    }),
                    true,
                )
            }
            None | Some("localize" | "track") => {
                let t0 = Instant::now();
                let res = self
                    .ensure_warmup_complete()
                    .and_then(|()| self.execute_localize(req));
                match res {
                    Ok((summary, case_report)) => {
                        self.requests_served = self.requests_served.wrapping_add(1);
                        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                        (
                            json!({
                                "ok": true,
                                "request_id": req_id,
                                "elapsed_ms": elapsed_ms,
                                "summary": summary,
                                "case_report": case_report,
                            }),
                            false,
                        )
                    }
                    Err(err) => (
                        json!({
                            "ok": false,
                            "request_id": req_id,
                            "error": err.to_string(),
                        }),
                        false,
                    ),
                }
            }
            Some(other) => (
                json!({
                    "ok": false,
                    "request_id": req_id,
                    "error": format!("unsupported op `{other}`"),
                }),
                false,
            ),
        }
    }

    fn execute_localize(&mut self, req: ServeRequest) -> Result<(Value, Value), PipelineError> {
        let prep = req.into_prepared(&self.opts)?;
        if let Some(raw_case) = prep.single_case {
            return self.execute_single_case(
                &prep.track_opts,
                &raw_case,
                prep.persist_output.as_deref(),
            );
        }
        let summary = run_corridor_track_with_resources(
            &prep.track_opts,
            None,
            Arc::clone(&self.resources),
            prep.persist_output.as_deref(),
        )?;
        Ok((summary, Value::Null))
    }

    fn execute_single_case(
        &mut self,
        track_opts: &crate::CorridorTrackOptions,
        raw_case: &PhotoCase,
        persist_dir: Option<&Path>,
    ) -> Result<(Value, Value), PipelineError> {
        if let Some(dir) = persist_dir {
            fs::create_dir_all(dir).map_err(|source| PipelineError::Io {
                path: dir.to_path_buf(),
                source,
            })?;
            write_json(
                &dir.join("run.json"),
                &json!({ "catalog_sha256": self.resources.surface.catalog_sha256 }),
            )?;
        }

        self.single_engine.reset_session();
        self.single_engine
            .set_search_opts(build_search_opts(track_opts));

        let case = apply_case_overrides(raw_case, track_opts);
        let query_rgb = load_case_rgb(&track_opts.dataset_dir, &case.image)?;
        let disk_mask = (!case.valid_mask.is_empty())
            .then(|| load_case_mask(&track_opts.dataset_dir, &case.valid_mask, case.size))
            .transpose()?;
        let focal_est = FocalLengthEstimator::default();
        let mount = mount_profile_for_dynamics(&track_opts.dynamics_profile);
        let live_input = build_live_frame_input(
            &track_opts.dataset_dir,
            &case,
            &query_rgb,
            disk_mask.as_ref(),
            mount,
            0,
            1_000_000_000,
            &focal_est,
        );

        let (outcome, diag) = self
            .single_engine
            .process_frame_with_diagnostics(&live_input, &case.id)?;
        let tel = &outcome.telemetry;
        let mut latencies = StageLatencies::default();
        latencies.seg_refresh_ms.extend(tel.segformer_ms);
        if matches!(
            tel.tracking_mode,
            TrackingMode::CorridorAnchor | TrackingMode::Unresolved
        ) {
            latencies.anchor_ms.push(tel.solve_ms);
        }

        let mut telemetry = Vec::new();
        record_frame_telemetry(
            &mut telemetry,
            &case.id,
            self.resources.xfeat.runtime_name(),
            tel,
            &diag.filter_snapshot,
        );
        let mut accepted = 0_usize;
        let case_report = emit_corridor_case_report(
            persist_dir,
            &self.resources.surface,
            &case.id,
            tel,
            outcome.anchor_confidence,
            &diag,
            &mut accepted,
        )?;
        let final_snap = self.single_engine.filter_snapshot();
        let out_str = persist_dir.map_or_else(String::new, |d| d.display().to_string());
        let summary = build_run_summary(RunSummaryParams {
            evaluated: 1,
            accepted,
            acquiring_frames: usize::from(tel.tracking_mode == TrackingMode::AcquiringAnchor),
            warm_tracked_frames: usize::from(tel.tracking_mode == TrackingMode::WarmTrack),
            independent_fixes: self.single_engine.independent_fixes(),
            seg_refreshes: usize::from(tel.segformer_refreshed),
            runtime_name: self.resources.xfeat.runtime_name(),
            indexed_tile_count: self.resources.store.indexed_tile_count(),
            cached_tile_count: self.resources.store.cached_tile_count(),
            resident_window_bytes: self.resources.store.resident_window_bytes(),
            shared_gpu_texture_bytes: self.resources.shared_gpu_texture_bytes(),
            tiles_built_live: self.single_engine.tiles_built_live(),
            tile_build_ms: &diag.tile_build_ms,
            shader_warmup_ms: 0.0,
            output_dir: &out_str,
            telemetry,
            background_refresh: track_opts.background_refresh,
            anchor_results: Vec::new(),
            latencies: &latencies,
            filter_snap: &final_snap,
            locked_focal_px: self.single_engine.locked_focal_px(),
        });

        Ok((summary, case_report))
    }
}

/// Run a persistent `JSONL` corridor worker loop over `reader` and `writer`.
///
/// The function writes one `"ready"` JSON line first. Next, it processes one JSON request per
/// input line until EOF or `"op": "shutdown"`.
///
/// # Errors
///
/// Returns [`PipelineError`] when session startup or stream output fails.
pub fn run_corridor_serve<R: BufRead, W: Write>(
    opts: &CorridorServeOptions,
    reader: R,
    mut writer: W,
) -> Result<Value, PipelineError> {
    let mut session = CorridorServeSession::open(opts.clone())?;
    write_json_line(&mut writer, &session.ready_banner())?;
    for line_res in reader.lines() {
        let line = line_res.map_err(|source| PipelineError::Io {
            path: PathBuf::from("stdin"),
            source,
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (resp, should_stop) = session.handle_line(trimmed);
        write_json_line(&mut writer, &resp)?;
        if should_stop {
            break;
        }
    }
    drop(session.ensure_warmup_complete());
    Ok(json!({
        "status": "stopped",
        "requests_served": session.requests_served,
        "shared_gpu_texture_bytes": session.resources.shared_gpu_texture_bytes(),
        "shader_warmup_ms": session.warmup_ms,
    }))
}

fn warmup_resources_async(res: &FlightEngineResources) -> Result<f64, FlightVisualError> {
    let t0 = Instant::now();
    let _ = res
        .xfeat
        .warmup(res.matcher.as_ref().map(LoadedMatcher::as_matcher))?;
    if let Some(ref shared) = res.shared_gpu {
        let k0 = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
        if let Some(mut g) = SurfaceGpuRenderer::from_shared(
            shared,
            &k0,
            [800, 600],
            GpuQueueRole::ForegroundTracking,
        ) {
            let lat = res.surface.origin_lat;
            let lon = res.surface.origin_lon;
            let dem = res
                .surface
                .sample_dem(lat, lon)
                .unwrap_or(res.surface.min_elev_m);
            let eye = res.surface.local(lat, lon, dem + 200.0);
            let rot = res.surface.camera_rotation(lat, lon, 0.0, 45.0, 0.0);
            drop(g.render_view(&res.surface, eye, &rot, &k0, [800, 600]));
        }
    }
    Ok(t0.elapsed().as_secs_f64() * 1000.0)
}

fn write_json_line<W: Write>(writer: &mut W, value: &Value) -> Result<(), PipelineError> {
    let line = serde_json::to_string(value).map_err(|source| PipelineError::Json {
        context: "corridor-serve JSONL response".into(),
        source,
    })?;
    writeln!(writer, "{line}").map_err(|source| PipelineError::Io {
        path: PathBuf::from("stdout"),
        source,
    })?;
    writer.flush().map_err(|source| PipelineError::Io {
        path: PathBuf::from("stdout"),
        source,
    })
}
