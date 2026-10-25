//! Test pose alternatives with subsequent images and unchanged geometry gates.
mod report;
mod tracking;
use super::{
    acquire,
    job::SurfaceJob,
    sequence::input::{Manifest, Sample},
};
use crate::{
    Error, assets,
    config::Config,
    evaluate::tracker::{Tracker, TrackerBackend},
};
use navigate_visual::{
    Frame, FrameStamp, LocalizerConfig, OrthographicReference, PosePrior, PoseVerifier,
};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};
use tracking::Hypothesis;

struct Runner {
    reference: OrthographicReference,
    prior: PosePrior,
    verifier: PoseVerifier,
    tracker: Tracker,
    hypotheses: Vec<Hypothesis>,
    previous: Frame,
    reference_limit: usize,
}

pub fn run_blocking(
    config: &Config,
    root: &Path,
    manifest: &Path,
    output: &Path,
    limit: usize,
) -> Result<(), Error> {
    let total = Instant::now();
    let spec: Manifest = assets::read_json_blocking(manifest)?;
    spec.validate()?;
    if !(1..=4).contains(&limit) || spec.keyframe_interval.is_some() {
        return Err(Error::Input(
            "hypothesis probe needs 1 through 4 hypotheses and no keyframe correction".into(),
        ));
    }
    let manifest_root = manifest.parent().unwrap_or(Path::new("."));
    let job_path = manifest_root.join(&spec.job);
    let job: SurfaceJob = assets::read_json_blocking(&job_path)?;
    acquire::validate(config, &job)?;
    let (initial, reference, prior) =
        job.load_blocking(job_path.parent().unwrap_or(manifest_root), config)?;
    let camera = initial.camera;
    drop(initial);
    let sample = spec
        .frames
        .first()
        .ok_or_else(|| Error::Input("missing initial frame".into()))?;
    let first = load_frame_blocking(sample, camera, manifest_root, config)?;
    let source_loading_ms = total.elapsed().as_secs_f64() * 1000.0;
    let (candidates, acquisition) =
        acquire_blocking(&first, &reference, &prior, config, root, limit)?;
    let mut runner = Runner {
        reference,
        prior,
        verifier: PoseVerifier::new(LocalizerConfig::default())?,
        tracker: Tracker::new_blocking(TrackerBackend::Wgpu)?,
        hypotheses: Vec::new(),
        previous: first,
        reference_limit: config.limits.reference_bytes,
    };
    let initialized = runner.initialize(candidates)?;
    let setup_ms = total.elapsed().as_secs_f64() * 1000.0;
    let mut frames = vec![runner.record(&runner.previous, initialized)];
    for sample in spec.frames.iter().skip(1) {
        let started = Instant::now();
        let frame = load_frame_blocking(sample, camera, manifest_root, config)?;
        let loading_ms = started.elapsed().as_secs_f64() * 1000.0;
        let rows = runner.advance_blocking(&frame)?;
        let mut row = runner.record(&frame, rows);
        row["frame_loading_ms"] = json!(loading_ms);
        row["pipeline_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
        frames.push(row);
        runner.previous = frame;
    }
    assets::write_json_blocking(
        output,
        &json!({"hypothesis_limit":limit,"job":job,"tracker":runner.tracker.identity,
        "acquisition":acquisition,"source_loading_ms":source_loading_ms,"setup_ms":setup_ms,"frames":frames,
        "total_ms":total.elapsed().as_secs_f64()*1000.0,
        "scope":"Keep the first bounded set of actual geometry passes in search order. Track their map points in one batch. Verify each with the same blind prior and geometry gates. Do not rank with truth, merge evidence, or recover a lost hypothesis. A pass is not an accurate or unique fix. Release inference models after acquisition. Keep one full source map and at most one temporary crop. Memory counters exclude models, driver buffers, decode peaks, allocator overhead, and stored JSON records. Covariance includes pixel noise only; it excludes map, calibration, and association bias. Repeated or shared observations are not independent measurements."}),
    )?;
    tracing::info!(path=%output.display(), "Saved pose hypothesis diagnostic");
    Ok(())
}

fn acquire_blocking(
    frame: &Frame,
    reference: &OrthographicReference,
    prior: &PosePrior,
    config: &Config,
    root: &Path,
    limit: usize,
) -> Result<(Vec<acquire::Acquired>, Value), Error> {
    let start = Instant::now();
    let mut engine = acquire::engine::Engine::new(config.load_blocking(root)?)?;
    let model_load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    engine.query_blocking(&frame.image, config)?;
    let query_ms = start.elapsed().as_secs_f64() * 1000.0;
    let result = acquire::search_hypotheses_blocking(
        frame,
        reference,
        prior,
        &mut engine,
        config,
        true,
        limit,
    )?;
    let candidates = result
        .selected
        .into_iter()
        .chain(result.alternatives)
        .collect();
    Ok((
        candidates,
        json!({"search":result.record,"model_load_ms":model_load_ms,"query_extraction_ms":query_ms,
        "reference_encodings":engine.reference_encodings,"peak_reference_accounted_bytes":engine.peak_reference_bytes}),
    ))
}

fn load_frame_blocking(
    sample: &Sample,
    camera: navigate_visual::CameraModel,
    root: &Path,
    config: &Config,
) -> Result<Frame, Error> {
    let image = assets::image_verified_blocking(
        &root.join(&sample.image.path),
        config.limits.image_pixels,
        Some(&sample.image.sha256),
    )?;
    if image.dimensions() != (camera.width, camera.height) {
        return Err(Error::Input(
            "hypothesis image does not match calibration".into(),
        ));
    }
    Ok(Frame {
        camera,
        image,
        stamp: FrameStamp {
            sequence: sample.frame_id,
            capture_time_ns: sample.capture_time_ns,
        },
    })
}

#[cfg(test)]
mod tests;
