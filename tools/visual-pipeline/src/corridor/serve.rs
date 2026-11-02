//! Persistent JSON-Lines (`JSONL`) worker session (`corridor-serve`) for corridor localization.
//!
//! A resident worker loads the terrain catalog, the corridor landmark bank, and the neural
//! weights one time per operating-system process. It compiles GPU shaders at startup so each
//! single-image or profile request runs without cold shader compilation.

mod request;

use super::{
    CorridorTrackOptions, VehicleDynamicsProfile,
    anchor::multitier::{SearchTierMatcher, load_search_tier_matcher},
    masking::AmortizedMasker,
    tracker::run_corridor_track_resident,
};
use crate::{
    PipelineError,
    realtime::helpers::{check_encoder_contract, load_lighterglue_model, raw_file_sha256},
    surface::{Surface, SurfaceGpuRenderer, gpu::GpuQueueRole},
    xfeat_runtime::{LoadedXFeat, XFeatRuntimeConfig},
};
use nalgebra::Matrix3;
use navigate_match::{CorridorTileStore, DEFAULT_WINDOW_TILES, HadamardProjector, LighterGlue};
use navigate_visual::LocalFrame;
use request::ServeRequest;
use serde_json::{Value, json};
use std::{
    fs,
    io::{BufRead, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
};

pub(super) struct CorridorResidentResources {
    pub surface: Arc<Surface>,
    pub local_frame: Option<LocalFrame>,
    pub store: Arc<CorridorTileStore>,
    pub projector: HadamardProjector,
    pub xfeat: LoadedXFeat,
    pub lighterglue: Option<LighterGlue>,
    pub search_tier_matcher: Option<Arc<SearchTierMatcher>>,
    pub prewarmed: bool,
}

impl CorridorResidentResources {
    pub(super) fn open(
        catalog_path: &Path,
        corridor_bank_path: &Path,
        xfeat_weights_path: &Path,
        xfeat_runtime: &XFeatRuntimeConfig,
        lighterglue_weights_path: Option<&Path>,
        (dinov2_weights_path, place_index_path): (Option<&Path>, Option<&Path>),
        prewarmed: bool,
    ) -> Result<Self, PipelineError> {
        let surface = Arc::new(Surface::from_catalog(catalog_path)?);
        let local_frame = LocalFrame::wgs84_enu(surface.origin_lat, surface.origin_lon).ok();
        let catalog_digest = raw_file_sha256(catalog_path)?;
        let xfeat = LoadedXFeat::load(xfeat_weights_path, xfeat_runtime)?;
        let lighterglue = lighterglue_weights_path
            .map(|p| load_lighterglue_model(p, xfeat.device()))
            .transpose()?;
        let store = Arc::new(CorridorTileStore::open(
            corridor_bank_path,
            DEFAULT_WINDOW_TILES,
        )?);
        if store.catalog_sha256 != catalog_digest {
            return Err(PipelineError::Invalid(
                "corridor bank catalog_sha256 does not match --catalog".into(),
            ));
        }
        check_encoder_contract("corridor bank", &store.encoder_contract, &xfeat)?;
        let search_tier_matcher = load_search_tier_matcher(
            dinov2_weights_path,
            place_index_path,
            Arc::clone(&store),
            xfeat.device(),
        )?;
        let projector = HadamardProjector::new(store.seed);
        Ok(Self {
            surface,
            local_frame,
            store,
            projector,
            xfeat,
            lighterglue,
            search_tier_matcher,
            prewarmed,
        })
    }

    pub(super) fn open_for_track(
        opts: &CorridorTrackOptions,
        prewarmed: bool,
    ) -> Result<Self, PipelineError> {
        Self::open(
            &opts.catalog_path,
            &opts.corridor_bank_path,
            &opts.xfeat_weights_path,
            &opts.xfeat_runtime,
            opts.lighterglue_weights_path.as_deref(),
            (
                opts.dinov2_weights_path.as_deref(),
                opts.place_index_path.as_deref(),
            ),
            prewarmed,
        )
    }
}

/// Startup configuration for a resident `corridor-serve` worker session.
#[derive(Clone, Debug)]
pub struct CorridorServeOptions {
    /// Path to the primary terrain and orthoimagery `catalog.json`.
    pub catalog_path: PathBuf,
    /// Path to the `NVKTILE1` file or sharded `NVKTIDX1` directory.
    pub corridor_bank_path: PathBuf,
    /// Path to `XFeat` `.safetensors` weights.
    pub xfeat_weights_path: PathBuf,
    /// Optional path to `LighterGlue` `.safetensors` weights.
    pub lighterglue_weights_path: Option<PathBuf>,
    /// Optional path to a `Segformer` checkpoint directory.
    pub seg_model_dir: Option<PathBuf>,
    /// Optional path to `DINOv2` `.safetensors` weights for global fallback.
    pub dinov2_weights_path: Option<PathBuf>,
    /// Optional path to an `NVKPLAC1` place index.
    pub place_index_path: Option<PathBuf>,
    /// Semantic mask policy name.
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

/// Resident corridor localization worker that reuses loaded models and compiled GPU pipelines.
pub struct CorridorServeSession {
    opts: CorridorServeOptions,
    res: CorridorResidentResources,
    gpu_renderer: Mutex<Option<SurfaceGpuRenderer>>,
    masker: AmortizedMasker,
    startup_warmup_ms: f64,
    requests_served: u64,
}

impl CorridorServeSession {
    /// Open a resident worker session and compile neural and terrain GPU pipelines one time.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when catalog loading, bank verification, or warmup fails.
    pub fn open(opts: CorridorServeOptions) -> Result<Self, PipelineError> {
        let res = CorridorResidentResources::open(
            &opts.catalog_path,
            &opts.corridor_bank_path,
            &opts.xfeat_weights_path,
            &opts.xfeat_runtime,
            opts.lighterglue_weights_path.as_deref(),
            (
                opts.dinov2_weights_path.as_deref(),
                opts.place_index_path.as_deref(),
            ),
            true,
        )?;
        let t0 = Instant::now();
        let _ = res.xfeat.warmup(res.lighterglue.as_ref())?;
        let init_rs = [800_u32, 600_u32];
        let init_k = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
        let init_gpu = opts
            .polish_anchors
            .then(|| {
                let mut g = SurfaceGpuRenderer::try_new_with_role(
                    &res.surface,
                    &init_k,
                    init_rs,
                    GpuQueueRole::ForegroundTracking,
                )?;
                let lat = res.surface.origin_lat;
                let lon = res.surface.origin_lon;
                let dem = res
                    .surface
                    .sample_dem(lat, lon)
                    .unwrap_or(res.surface.min_elev_m);
                let eye = res.surface.local(lat, lon, dem + 200.0);
                let rot = res.surface.camera_rotation(lat, lon, 0.0, 45.0, 0.0);
                g.render_view(&res.surface, eye, &rot, &init_k, init_rs)
                    .ok();
                Some(g)
            })
            .flatten();
        let startup_warmup_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let seg_device = burn::tensor::Device::flex();
        let masker = AmortizedMasker::new(
            opts.seg_model_dir.as_deref(),
            &opts.mask_policy,
            6,
            10,
            &seg_device,
        )?;
        Ok(Self {
            opts,
            res,
            gpu_renderer: Mutex::new(init_gpu),
            masker,
            startup_warmup_ms,
            requests_served: 0,
        })
    }

    /// Build the startup handshake JSON value for the worker stream.
    #[must_use]
    pub fn ready_banner(&self) -> Value {
        json!({
            "status": "ready",
            "catalog_sha256": self.res.surface.catalog_sha256,
            "xfeat_runtime": self.res.xfeat.runtime_name(),
            "indexed_tiles": self.res.store.indexed_tile_count(),
            "shader_warmup_ms": self.startup_warmup_ms,
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
            Some("shutdown" | "quit" | "exit") => (
                json!({
                    "ok": true,
                    "request_id": req_id,
                    "status": "shutdown",
                    "requests_served": self.requests_served,
                }),
                true,
            ),
            None | Some("localize" | "track") => {
                let t0 = Instant::now();
                match self.execute_localize(req) {
                    Ok((summary, case_report)) => {
                        self.requests_served += 1;
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
        let prep = req.into_prepared(&self.opts, self.requests_served)?;
        self.masker.reset_session();
        let gpu_ref = prep.track_opts.polish_anchors.then_some(&self.gpu_renderer);
        let summary_res = run_corridor_track_resident(
            &prep.track_opts,
            prep.single_case.as_ref().map(|c| c.as_slice()),
            &self.res,
            gpu_ref,
            Some(&mut self.masker),
        );
        let case_report = prep
            .single_case
            .as_ref()
            .and_then(|c| {
                fs::read_to_string(prep.output_dir.join(format!("{}.json", c[0].id))).ok()
            })
            .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
            .unwrap_or(Value::Null);
        if let Some(tmp) = prep.temp_out {
            fs::remove_dir_all(tmp).ok();
        }
        Ok((summary_res?, case_report))
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
    Ok(json!({
        "status": "stopped",
        "requests_served": session.requests_served,
        "shader_warmup_ms": session.startup_warmup_ms,
    }))
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
