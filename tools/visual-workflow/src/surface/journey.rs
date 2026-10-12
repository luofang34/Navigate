//! Resident bounded acquisition and conditional tracking on one immutable source map.
mod models;
mod processing;
use super::{
    acquire::{self, engine::Engine},
    job::SurfaceJob,
    sequence::{continuity::Continuity, input::Manifest, tracks::Tracks},
};
use crate::{
    Error, assets,
    config::{Config, Runtime},
    evaluate::tracker::{Tracker, TrackerBackend},
};
use navigate_visual::{
    Frame, FrameStamp, LocalizerConfig, OrthographicReference, PosePrior, PoseVerifier,
};
use navigate_visual_workflow::FeatureBackend;
use serde_json::json;
use std::{path::Path, time::Instant};

struct Tracking {
    region: Option<OrthographicReference>,
    window: Option<[u32; 4]>,
    tracks: Tracks,
}
struct Runner {
    models: models::Models,
    reference: OrthographicReference,
    prior: PosePrior,
    verifier: PoseVerifier,
    tracker: Tracker,
    continuity: Continuity,
    state: Option<Tracking>,
    interval: usize,
    record_matches: bool,
}
pub fn run_blocking(
    config: &Config,
    root: &Path,
    manifest: &Path,
    output: &Path,
    reload_model_on_recovery: bool,
    reference_cache_mib: usize,
) -> Result<(), Error> {
    let total = Instant::now();
    if reference_cache_mib > 64 {
        return Err(Error::Input(
            "reference cache must be at most 64 MiB".into(),
        ));
    }
    let spec: Manifest = assets::read_json_blocking(manifest)?;
    spec.validate()?;
    if spec.keyframe_interval.is_some() {
        return Err(Error::Input(
            "acquisition sequence requires keyframe correction disabled".into(),
        ));
    }
    let manifest_root = manifest.parent().unwrap_or(Path::new("."));
    let job_path = manifest_root.join(&spec.job);
    let job: SurfaceJob = assets::read_json_blocking(&job_path)?;
    acquire::validate(config, &job)?;
    let (initial, reference, prior) =
        job.load_blocking(job_path.parent().unwrap_or(manifest_root), config)?;
    let loading_ms = total.elapsed().as_secs_f64() * 1000.0;
    let started = Instant::now();
    let runtime = Runtime::new_blocking(config, root)?;
    let engine = Engine::new(runtime.load_blocking(config, root)?)?;
    let model_load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let policy = cache_policy(&job, reference_cache_mib);
    let mut runner = Runner {
        models: models::Models::new(engine, runtime, root, reload_model_on_recovery, policy),
        reference,
        prior,
        verifier: PoseVerifier::new(LocalizerConfig::default())?,
        tracker: Tracker::new_blocking(TrackerBackend::Wgpu)?,
        continuity: Continuity::new(initial.camera)?,
        state: None,
        interval: spec.refresh_interval,
        record_matches: job.record_matches,
    };
    let setup_ms = total.elapsed().as_secs_f64() * 1000.0;
    let frames = runner.frames_blocking(&spec, initial.camera, manifest_root, config)?;
    let identity = runner.models.engine_mut()?.backend.identity().to_owned();
    assets::write_json_blocking(
        output,
        &json!({"backend":identity,"tracker":runner.tracker.identity,
        "job":job,"refresh_interval":spec.refresh_interval,"setup_ms":setup_ms,"source_loading_ms":loading_ms,"model_load_ms":model_load_ms,
        "reference_cache_mib":reference_cache_mib,"model_loads":runner.models.loads,"query_encodings":runner.models.queries,"reload_model_on_recovery":reload_model_on_recovery,"source_loads":1,"frames":frames,"total_ms":total.elapsed().as_secs_f64()*1000.0,
        "scope":"One immutable full source map with resident inference models. Each acquisition shares one query encoding across bounded coarse/local/fine search; exact map features can be retained in a bounded cache. The first geometry pass is a conditional hypothesis, not a unique navigation fix. Track only the selected crop's map identities. A gap, failed geometry or scheduled refresh discards that state and searches from the original blind prior. Reacquisition starts a separate motion segment. No inferred motion across gaps, geodetic fusion, or truth input. Full source and at most one crop remain resident; runtime/model memory is separate. Offline replay processes every declared frame and does not establish real-time deadline feasibility."}),
    )?;
    tracing::info!(path=%output.display(),"Saved resident acquisition and tracking sequence");
    Ok(())
}

fn cache_policy(job: &SurfaceJob, reference_cache_mib: usize) -> acquire::engine::CachePolicy {
    let source_identity = json!({"image":job.reference.image.sha256,"heights":job.reference.heights.sha256,
        "origin":job.reference.origin,"column":job.reference.column,"row":job.reference.row,
        "map_release":job.reference.map_release,"map_sha256":job.reference.map_sha256});
    use sha2::{Digest, Sha256};
    let source: [u8; 32] = Sha256::digest(source_identity.to_string().as_bytes()).into();
    acquire::engine::CachePolicy {
        bytes: reference_cache_mib * 1024 * 1024,
        source,
    }
}

impl Runner {
    fn frames_blocking(
        &mut self,
        spec: &Manifest,
        camera: navigate_visual::CameraModel,
        manifest_root: &Path,
        config: &Config,
    ) -> Result<Vec<serde_json::Value>, Error> {
        let mut frames = Vec::new();
        for sample in &spec.frames {
            let started = Instant::now();
            let frame = Frame {
                camera,
                stamp: FrameStamp {
                    sequence: sample.frame_id,
                    capture_time_ns: sample.capture_time_ns,
                },
                image: assets::image_verified_blocking(
                    &manifest_root.join(&sample.image.path),
                    config.limits.image_pixels,
                    Some(&sample.image.sha256),
                )?,
            };
            if frame.image.dimensions() != (frame.camera.width, frame.camera.height) {
                return Err(Error::Input(
                    "acquisition sequence image does not match calibration".into(),
                ));
            }
            let frame_loading_ms = started.elapsed().as_secs_f64() * 1000.0;
            let mut record = self.process_blocking(frame, config)?;
            record["frame_loading_ms"] = json!(frame_loading_ms);
            record["pipeline_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
            record["image"] = json!(sample.image);
            frames.push(record);
        }
        Ok(frames)
    }
}
