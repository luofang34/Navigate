//! Compare map representations on windows chosen without query pose information.
mod cache;
mod evaluate;
mod geometry;
use super::job::SurfaceJob;
use crate::{
    Error, assets,
    config::{Backend, Config, Graph},
};
use navigate_visual::{LocalizerConfig, PoseVerifier};
use navigate_visual_workflow::FeatureBackend;
use serde_json::json;
use std::{path::Path, time::Instant};

pub fn run_blocking(
    config: &Config,
    root: &Path,
    input: &Path,
    output: &Path,
    iterations: usize,
) -> Result<(), Error> {
    compare_blocking(config, root, input, output, iterations, false)
}

pub fn run_scored_blocking(
    config: &Config,
    root: &Path,
    input: &Path,
    output: &Path,
    iterations: usize,
) -> Result<(), Error> {
    compare_blocking(config, root, input, output, iterations, true)
}

fn compare_blocking(
    config: &Config,
    root: &Path,
    input: &Path,
    output: &Path,
    iterations: usize,
    record_scores: bool,
) -> Result<(), Error> {
    let job: SurfaceJob = assets::read_json_blocking(input)?;
    validate(config, &job, iterations)?;
    if record_scores && !job.record_matches {
        return Err(Error::Input(
            "score diagnostics require recorded correspondences".into(),
        ));
    }
    let (frame, reference, prior) =
        job.load_blocking(input.parent().unwrap_or(Path::new(".")), config)?;
    let windows = geometry::windows(reference.image.dimensions())?;
    let start = Instant::now();
    let Backend::Features(mut backend) = config.load_blocking(root)? else {
        return Err(Error::Input("window probe needs a feature backend".into()));
    };
    let model_load_ms = start.elapsed().as_secs_f64() * 1000.0;
    let raster_bytes = reference.image.as_raw().len() * 5;
    let preparation_start = Instant::now();
    let caches = cache::prepare_blocking(
        &mut backend,
        &reference.image,
        &windows,
        &job.reference_quarter_turns,
        config,
        raster_bytes,
    )?;
    let preparation_ms = preparation_start.elapsed().as_secs_f64() * 1000.0;
    let verifier = PoseVerifier::new(LocalizerConfig::default())?;
    let context = evaluate::Context {
        frame: &frame,
        reference: &reference,
        prior: &prior,
        verifier: &verifier,
        record_matches: job.record_matches,
        record_scores,
    };
    let runs = iterations_blocking(&mut backend, &caches, &context, config, iterations)?;
    assets::write_json_blocking(
        output,
        &json!({"backend":backend.identity(),"model_load_ms":model_load_ms,
        "assignment_score_scope": "Optional raw compact matcher scores follow correspondence order. They are not calibrated probabilities, truth labels or position uncertainties.",
        "job":job,"feature_limit_per_encoder_input":config.keypoints,"windows":windows,"rotations":job.reference_quarter_turns,"preparation":caches.report,"preparation_ms":preparation_ms,
        "raster_bytes":raster_bytes,"iterations":runs,"observation_sha256":frame.evidence_sha256(),
        "scope":"Three overlapping windows are one map region, not independent geographic modes. Crops use their own letterbox, normalization and feature top-K. Shared subsets retain parent coordinates, encoder context, normalization and source digest. Thus the arms compare representations, not exact cache replacements. Query encoding is shared existing behavior. Resident feature bytes exclude models, runtime tensors and driver overhead. Packet roundtrip validates serialization in memory; it does not measure file loading. Full-block matching changes matcher context and is not equivalent to matching the subsets. No independent real-world accuracy is asserted."}),
    )?;
    tracing::info!(path=%output.display(), "Saved map feature window comparison");
    Ok(())
}

fn validate(config: &Config, job: &SurfaceJob, iterations: usize) -> Result<(), Error> {
    if !(1..=20).contains(&iterations) {
        return Err(Error::Input(
            "window iterations must be 1 through 20".into(),
        ));
    }
    if !matches!(config.graph, Graph::DenseImage) {
        return Err(Error::Input(
            "window probe requires the dense-image XFeat coordinate contract".into(),
        ));
    }
    geometry::validate_rotations(&job.reference_quarter_turns, config.limits.candidates)
}

fn iterations_blocking(
    backend: &mut navigate_visual_onnx::XFeatStages,
    caches: &cache::Cache,
    context: &evaluate::Context<'_>,
    config: &Config,
    iterations: usize,
) -> Result<Vec<serde_json::Value>, Error> {
    let mut runs = Vec::with_capacity(iterations);
    for iteration in 0..iterations {
        let started = Instant::now();
        let query = backend.extract_features_blocking(&context.frame.image)?;
        let metadata = backend.validate_features(&query, &context.frame.image)?;
        if metadata.count > config.limits.features_per_image
            || metadata.resident_bytes > config.limits.query_feature_bytes
        {
            return Err(Error::Input(
                "query features exceed window probe budget".into(),
            ));
        }
        let query_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut arms = Vec::new();
        // Rotate execution order so one representation does not always run on a cold matcher.
        for offset in 0..3 {
            arms.push(evaluate::arm_blocking(
                backend,
                &query,
                caches,
                context,
                (iteration + offset) % 3,
            )?);
        }
        runs.push(json!({"query_extraction_ms":query_ms,"query_features":query.len(),"arms":arms}));
    }
    Ok(runs)
}
