//! Execute prepared inputs without intermediate image files.
use crate::{
    Error, assets,
    config::{Backend, Config},
    job::Job,
};
use navigate_recording::Pose;
use navigate_visual::LocalizerConfig;
use navigate_visual_onnx::XFeatFeatures;
use navigate_visual_workflow::{FeatureBackend, FeatureLocalizer, FrameReport};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};

pub fn pack_blocking(
    config: &Config,
    root: &Path,
    image: &Path,
    output: &Path,
) -> Result<(), Error> {
    let Backend::Features(mut backend) = config.load_blocking(root)? else {
        return Err(Error::Input(
            "this model does not expose an offline feature encoder".into(),
        ));
    };
    let image = assets::image_blocking(image, config.limits.image_pixels)?;
    let features = backend.extract_features_blocking(&image)?;
    backend.validate_features(&features, &image)?;
    let bytes = features.encode();
    assets::write_blocking(output, &bytes)?;
    tracing::info!(path=%output.display(),features=features.len(),bytes=bytes.len(),"Saved reference features");
    Ok(())
}
pub fn locate_blocking(
    config: &Config,
    root: &Path,
    job_path: &Path,
    output: &Path,
    iterations: usize,
) -> Result<(), Error> {
    if !(1..=1000).contains(&iterations) {
        return Err(Error::Input("iterations must be 1 through 1000".into()));
    }
    let job: Job = assets::read_json_blocking(job_path)?;
    if job.references.len() > config.limits.candidates {
        return Err(Error::Input(
            "candidate count exceeds the configured limit".into(),
        ));
    }
    let job_root = job_path
        .parent()
        .ok_or_else(|| Error::Input("job path has no parent".into()))?;
    let started = Instant::now();
    let backend = config.load_blocking(root)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let backend = match backend {
        Backend::Pair(matcher) => {
            return super::pair::locate_blocking(
                config, &job, job_root, output, iterations, *matcher, load_ms,
            );
        }
        Backend::Features(backend) => *backend,
    };
    let identity = backend.identity().to_owned();
    let mut workflow =
        FeatureLocalizer::new(backend, LocalizerConfig::default(), config.limits.model())?;
    let frame = job.query_blocking(job_root, config.limits.image_pixels)?;
    let prior = job.prior()?;
    let started = Instant::now();
    let mut references = Vec::new();
    let mut resident = 0_usize;
    let mut source = job.references_blocking(job_root, config.limits.image_pixels)?;
    for entry in &job.references {
        let view =
            job.reference_blocking(entry, job_root, config.limits.image_pixels, &mut source)?;
        let reference = if let Some(path) = &entry.features {
            let bytes = assets::read_blocking(&job_root.join(path), 2 * 1024 * 1024)?;
            workflow.import_reference(view, XFeatFeatures::decode(&bytes, config.keypoints)?)?
        } else {
            workflow.prepare_reference_blocking(view)?
        };
        resident = resident.saturating_add(reference.resident_bytes());
        if resident > config.limits.reference_bytes {
            return Err(Error::Input(
                "resident references exceed the memory limit".into(),
            ));
        }
        references.push(reference);
    }
    let prepare_ms = started.elapsed().as_secs_f64() * 1000.0;
    let mut reports = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        reports.push(report(workflow.estimate_blocking(
            &frame,
            &prior,
            &references,
        )?));
    }
    let result = json!({"backend":identity,"model_load_ms":load_ms,"reference_prepare_ms":prepare_ms,"reference_resident_bytes":resident,
        "render":render_report(&source),"reference_ids":job.references.iter().map(|r|r.id.as_str()).collect::<Vec<_>>(),"observation_sha256":frame.evidence_sha256(),
        "scope":"Each iteration includes one query extraction, candidate matching, and pose verification. Model load, image decode, and reference preparation are separate.","iterations":reports,"adapter_tensor_io":workflow.backend().tensor_statistics()});
    assets::write_json_blocking(output, &result)?;
    tracing::info!(path=%output.display(),iterations,"Saved localization results");
    Ok(())
}
pub(crate) fn render_report(source: &crate::job::References) -> Value {
    let identity = source.identity();
    json!({"backend":identity.as_ref().map(|i|&i.revision),"style_sha256":identity.as_ref().map(|i|&i.style_sha256),
        "resident_texture_bytes":source.texture_bytes(),"render_ms":source.render_ms,
        "readback_bytes_per_view":source.readback_bytes(),"gpu_output_and_readback_bytes":source.readback_bytes()*2,
        "scope":"Render timings include dispatch, image/depth readback and host materialization; exclude scene load. Texture memory excludes host atlas staging and driver overhead."})
}
pub(crate) fn report(report: FrameReport) -> Value {
    let candidates:Vec<_>=report.candidates.into_iter().map(|candidate| {
        let mut value=json!({"reference_index":candidate.reference_index,"matches":candidate.matches,
            "matching_ms":candidate.times.matching.as_secs_f64()*1000.0,"geometry_ms":candidate.times.geometry.as_secs_f64()*1000.0,
            "accepted":candidate.decision.is_ok()});
        match candidate.decision {
            Ok(estimate)=>{value["pose"]=json!(Pose::from(estimate.pose));value["geodetic"]=json!(estimate.frame.geodetic(estimate.pose.position));
                value["quality"]=json!({"inliers":estimate.quality.inliers,"spatial_support":estimate.quality.spatial_support,
                    "occupied_cells":estimate.quality.occupied_cells,"rms_px":estimate.quality.reprojection_rms_px});},
            Err(error)=>value["rejection"]=json!(error.to_string()),
        }
        value
    }).collect();
    json!({"query_extraction_ms":report.extraction.as_secs_f64()*1000.0,"total_ms":report.total.as_secs_f64()*1000.0,"candidates":candidates})
}
