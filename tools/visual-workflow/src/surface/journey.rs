//! Resident bounded acquisition and conditional tracking on one immutable source map.
pub(crate) mod arrival;
mod models;
mod processing;
mod replay;
use super::{
    acquire,
    job::SurfaceJob,
    sequence::{continuity::Continuity, input::Manifest, tracks::Tracks},
};
use crate::{
    Error, assets,
    config::Config,
    evaluate::tracker::{Tracker, TrackerBackend},
};
use navigate_visual::{LocalizerConfig, OrthographicReference, PosePrior, PoseVerifier};
use navigate_visual_workflow::FeatureBackend;
use serde_json::json;
use std::{
    path::{Path, PathBuf},
    time::Instant,
};
pub struct Options {
    pub reload_model_on_recovery: bool,
    pub reference_cache_mib: usize,
    pub arrival: arrival::Options,
    pub feature_catalog: Option<PathBuf>,
}

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
    options: Options,
) -> Result<(), Error> {
    let Options {
        reload_model_on_recovery,
        reference_cache_mib,
        arrival: arrival_options,
        feature_catalog,
    } = options;
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
    let policy = cache_policy(&job, reference_cache_mib);
    let (models, model_load_ms) = models::Models::load_blocking(
        config,
        root,
        reload_model_on_recovery,
        policy,
        feature_catalog.as_deref(),
    )?;
    let mut runner = Runner {
        models,
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
    let (frames, arrivals) = runner.frames_blocking(
        &spec,
        initial.camera,
        manifest_root,
        config,
        arrival::Arrivals::new(arrival_options, total),
    )?;
    let identity = runner.models.engine_mut()?.backend.identity().to_owned();
    assets::write_json_blocking(
        output,
        &json!({"backend":identity,"tracker":runner.tracker.identity,
        "job":job,"refresh_interval":spec.refresh_interval,"setup_ms":setup_ms,"source_loading_ms":loading_ms,"model_load_ms":model_load_ms,
        "reference_cache_mib":reference_cache_mib,"feature_catalog":feature_catalog,"model_loads":runner.models.loads,"query_encodings":runner.models.queries,"reload_model_on_recovery":reload_model_on_recovery,"source_loads":1,"arrivals":arrivals,"frames":frames,"total_ms":total.elapsed().as_secs_f64()*1000.0,
        "scope":"One immutable full source map with resident inference models. Each acquisition shares one query encoding across bounded coarse/local/fine search; exact map features can be retained in a bounded cache. The first geometry pass is a conditional hypothesis, not a unique navigation fix. Track only the selected crop's map identities. A gap, failed geometry or scheduled refresh discards that state and searches from the original blind prior. Reacquisition starts a separate motion segment. No inferred motion across gaps, geodetic fusion, or truth input. Full source and at most one crop remain resident; runtime/model memory is separate. The arrival report declares frame admission and clock policy. Offline mode processes every declared frame; timed modes use actual elapsed computation time and retain original capture timestamps. Neither mode establishes live-camera deadline feasibility."}),
    )?;
    tracing::info!(path=%output.display(),"Saved resident acquisition and tracking sequence");
    Ok(())
}

fn cache_policy(job: &SurfaceJob, reference_cache_mib: usize) -> acquire::engine::CachePolicy {
    let source = acquire::store::source_namespace(&job.reference);
    acquire::engine::CachePolicy {
        bytes: reference_cache_mib * 1024 * 1024,
        source,
    }
}
