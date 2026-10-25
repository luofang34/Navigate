//! Run cross-image models with the same reference evidence.
use crate::{Error, assets, config::Config, job::Job, run};
use navigate_visual::{ImageMatcher, LocalizerConfig};
use navigate_visual_onnx::LoFtrMatcher;
use navigate_visual_workflow::PairLocalizer;
use serde_json::json;
use std::{path::Path, time::Instant};

pub fn locate_blocking(
    config: &Config,
    job: &Job,
    root: &Path,
    output: &Path,
    iterations: usize,
    matcher: LoFtrMatcher,
    load_ms: f64,
) -> Result<(), Error> {
    let identity = matcher.identity().to_owned();
    let frame = job.query_blocking(root, config.limits.image_pixels)?;
    let prior = job.prior()?;
    let started = Instant::now();
    let mut references = Vec::new();
    let mut resident = 0_usize;
    let mut source = job.references_blocking(root, config.limits.image_pixels)?;
    for entry in &job.references {
        if entry.features.is_some() {
            return Err(Error::Input(
                "image-pair matching cannot consume XFeat packets".into(),
            ));
        }
        let view = job.reference_blocking(entry, root, config.limits.image_pixels, &mut source)?;
        resident = resident
            .saturating_add(view.image.as_raw().len())
            .saturating_add(view.depth_m.len().saturating_mul(4));
        if resident > config.limits.reference_bytes {
            return Err(Error::Input("reference memory limit exceeded".into()));
        }
        references.push(view);
    }
    let cameras: Vec<_> = job
        .references
        .iter()
        .map(|r| r.camera.as_ref().unwrap_or(&job.camera).model())
        .collect();
    let calibrated: Vec<_> = references.iter().zip(&cameras).collect();
    let prepare_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut workflow =
        PairLocalizer::new(matcher, LocalizerConfig::default(), config.limits.model())?;
    let mut reports = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        reports.push(run::report(workflow.estimate_calibrated_blocking(
            &frame,
            &prior,
            &calibrated,
        )?));
    }
    assets::write_json_blocking(
        output,
        &json!({"backend":identity,"model_load_ms":load_ms,
        "reference_prepare_ms":prepare_ms,"reference_resident_bytes":resident,"render":run::render_report(&source),
        "reference_ids":job.references.iter().map(|r|r.id.as_str()).collect::<Vec<_>>(),"observation_sha256":frame.evidence_sha256(),
        "scope":"Image-pair matching and shared pose verification. No independent feature extraction stage.","iterations":reports}),
    )?;
    tracing::info!(path=%output.display(),iterations,"Saved image-pair localization results");
    Ok(())
}
