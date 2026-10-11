//! Image matching and verified geometry on a fixed orthographic surface.
pub mod cascade;
mod job;
mod matching;
pub mod prepare;
pub mod probe;
mod report;
pub mod windows;

use crate::{Error, assets, config::Config};
use navigate_visual::{LocalizerConfig, PoseVerifier};
use serde_json::json;
use std::{path::Path, time::Instant};

pub fn run_blocking(
    config: &Config,
    root: &Path,
    input: &Path,
    output: &Path,
    iterations: usize,
) -> Result<(), Error> {
    if !(1..=1000).contains(&iterations) {
        return Err(Error::Input("iterations must be 1 through 1000".into()));
    }
    let settings: job::SurfaceJob = assets::read_json_blocking(input)?;
    let input_root = input.parent().unwrap_or(Path::new("."));
    let (frame, reference, prior) = settings.load_blocking(input_root, config)?;
    let start = Instant::now();
    let raster_bytes = reference.image.as_raw().len() + reference.heights_m.len() * 4;
    let mut matcher = matching::Matcher::new_blocking(
        config,
        root,
        &reference.image,
        &settings.reference_quarter_turns,
        raster_bytes,
    )?;
    let setup_ms = start.elapsed().as_secs_f64() * 1000.0;
    let verifier = PoseVerifier::new(LocalizerConfig::default())?;
    let mut reports = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        let (extraction_ms, candidates) = matcher.matches_blocking(&frame.image, config)?;
        let mut records = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            records.push(report::verify(
                &verifier,
                &frame,
                &reference,
                &prior,
                candidate,
                matcher.identity(),
                settings.record_matches,
            ));
        }
        reports.push(
            json!({"query_extraction_ms":extraction_ms,"candidates":records,
            "total_ms":start.elapsed().as_secs_f64()*1000.0}),
        );
    }
    assets::write_json_blocking(
        output,
        &json!({"backend":matcher.identity(),
        "setup_ms":setup_ms,"iterations":reports,"observation_sha256":frame.evidence_sha256(),
        "reference_raster_bytes":raster_bytes,"reference_resident_bytes":matcher.resident_bytes,
        "scope":"Local metric pose conditional on the supplied orthographic map and calibration. No geographic registration or independent accuracy is asserted. Timings exclude file loading and model/reference setup."}),
    )?;
    tracing::info!(path=%output.display(), "Saved orthographic localization results");
    Ok(())
}
