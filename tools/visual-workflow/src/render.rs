//! Export verified GPU references for an optional offline preparation stage.
use crate::{Error, assets, job::Job, run};
use navigate_visual::ReferenceView;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{io::Cursor, path::Path, time::Instant};

pub fn run_blocking(
    job_path: &Path,
    output: &Path,
    iterations: usize,
    max_pixels: usize,
) -> Result<(), Error> {
    if !(1..=1000).contains(&iterations) || !(1024..=16_777_216).contains(&max_pixels) {
        return Err(Error::Input("render iterations or pixel limit".into()));
    }
    let mut wire: Value = assets::read_json_blocking(job_path)?;
    let job: Job = serde_json::from_value(wire.clone()).map_err(|source| Error::Json {
        path: job_path.into(),
        source,
    })?;
    if job.references.is_empty() || job.references.len() > 128 {
        return Err(Error::Input(
            "render requires 1 through 128 references".into(),
        ));
    }
    let absolute = std::fs::canonicalize(job_path).map_err(|source| Error::Io {
        operation: "resolve job",
        path: job_path.into(),
        source,
    })?;
    let root = absolute
        .parent()
        .ok_or_else(|| Error::Input("job parent".into()))?;
    let started = Instant::now();
    let mut source = job.references_blocking(root, max_pixels)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    std::fs::create_dir(output).map_err(|source| Error::Io {
        operation: "create render output",
        path: output.into(),
        source,
    })?;
    let mut references = Vec::new();
    for (index, entry) in job.references.iter().enumerate() {
        let view = job.reference_blocking(entry, root, max_pixels, &mut source)?;
        let mut saved = save_blocking(output, index, &view)?;
        saved["id"] = json!(entry.id);
        saved["camera"] = json!(entry.camera.as_ref().unwrap_or(&job.camera));
        saved["pose"] = json!(navigate_recording::Pose::from(view.pose));
        references.push(saved);
    }
    for _ in 1..iterations {
        for entry in &job.references {
            job.reference_blocking(entry, root, max_pixels, &mut source)?;
        }
    }
    wire["references"] = json!(references);
    wire["query"]["path"] = json!(root.join(&job.query.path));
    wire["render_scene"] = Value::Null;
    assets::write_json_blocking(&output.join("job.json"), &wire)?;
    assets::write_json_blocking(
        &output.join("render.json"),
        &json!({"scene_load_ms":load_ms,"render":run::render_report(&source)}),
    )?;
    tracing::info!(path=%output.display(),iterations,"Saved rendered reference job");
    Ok(())
}

fn save_blocking(output: &Path, index: usize, view: &ReferenceView) -> Result<Value, Error> {
    let name = format!("reference-{index}.png");
    let mut encoded = Cursor::new(Vec::new());
    view.image
        .write_to(&mut encoded, image::ImageFormat::Png)
        .map_err(|source| Error::Image {
            path: output.join(&name),
            source,
        })?;
    let bytes = encoded.into_inner();
    assets::write_blocking(&output.join(&name), &bytes)?;
    let depth_name = format!("reference-{index}.depth");
    let depth: Vec<u8> = view.depth_m.iter().flat_map(|d| d.to_le_bytes()).collect();
    assets::write_blocking(&output.join(&depth_name), &depth)?;
    Ok(
        json!({"image":{"path":name,"sha256":hex::encode(Sha256::digest(&bytes))},
        "depth":depth_name,"depth_sha256":hex::encode(Sha256::digest(&depth))}),
    )
}
