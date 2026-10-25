//! Bounded image-only propagation of map pairs before pose initialization.
mod input;
mod pool;
use super::{matching::Matches, report};
use crate::{
    Error, assets,
    config::Config,
    evaluate::tracker::{Tracker, TrackerBackend},
};
use navigate_visual::{Frame, LocalizerConfig, OrthographicReference, PosePrior, PoseVerifier};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

pub fn run_blocking(
    config: &Config,
    input: &Path,
    output: &Path,
    history: usize,
) -> Result<(), Error> {
    if !(1..=8).contains(&history) {
        return Err(Error::Input(
            "accumulation history must be 1 through 8 frames".into(),
        ));
    }
    let started = Instant::now();
    let bytes = assets::read_blocking(input, 128 * 1024 * 1024)?;
    let spec: input::Frozen = serde_json::from_slice(&bytes).map_err(|source| Error::Json {
        path: input.into(),
        source,
    })?;
    spec.validate()?;
    let root = input.parent().unwrap_or(Path::new("."));
    let (_, reference, prior) = spec.job.load_blocking(root, config)?;
    let mut tracker = Tracker::new_blocking(TrackerBackend::Wgpu)?;
    let verifier = PoseVerifier::new(LocalizerConfig::default())?;
    let mut previous: Option<Frame> = None;
    let mut evidence = Vec::new();
    let mut records = Vec::new();
    for sample in &spec.frames {
        let frame = spec.frame_blocking(sample, root, config)?;
        let current = sample.evidence(reference.image.dimensions(), frame.image.dimensions())?;
        let start = Instant::now();
        let older = if let Some(previous) = previous.as_ref() {
            pool::advance_blocking(
                evidence,
                previous,
                &frame,
                history,
                tracker.backend.as_mut(),
            )?
        } else {
            pool::Propagation::default()
        };
        let tracking_ms = start.elapsed().as_secs_f64() * 1000.0;
        let propagated = older.pairs.len();
        let before_merge = current.len() + propagated;
        evidence = pool::merge(current, older.pairs);
        let candidates = score(
            &verifier,
            &frame,
            &reference,
            &prior,
            &evidence,
            &spec.backend,
        );
        records.push(json!({"frame_id":sample.frame_id,"observation_sha256":sample.observation_sha256,"propagated_pairs":propagated,"tracking_attempted":older.attempted,"expired_or_reset":older.expired_or_reset,"tracking_rejected":older.tracking_rejected,"duplicates_or_capacity_removed":before_merge-evidence.len(),"retained_pairs":evidence.len(),"tracking_ms":tracking_ms,"candidates":candidates}));
        previous = Some(frame);
    }
    assets::write_json_blocking(
        output,
        &json!({"input_sha256":hex::encode(Sha256::digest(&bytes)),"history":history,"tracker":tracker.identity,"frames":records,"total_ms":started.elapsed().as_secs_f64()*1000.0,
        "scope":"Frozen map-only correspondences, one known map region. Propagate all retained query points with forward-backward optical flow before pose initialization. Keep newest evidence first, separated by 12 pixels in both images per rotation, at most 4096 pairs per rotation and 8 frames; reset on gaps over 200 ms. No truth, pose-based track filtering, acceptance relaxation or navigation update. Repeated tracks and reference errors remain correlated. History 1 isolates spatial filtering. Timing excludes original feature inference; this is not end-to-end navigation."}),
    )?;
    tracing::info!(path=%output.display(),history,"Saved map-pair accumulation probe");
    Ok(())
}

fn score(
    verifier: &PoseVerifier,
    frame: &Frame,
    reference: &OrthographicReference,
    prior: &PosePrior,
    evidence: &[pool::Evidence],
    backend: &str,
) -> Vec<Value> {
    let mut candidates = Vec::new();
    for rotation in 0..4 {
        let selected: Vec<_> = evidence.iter().filter(|p| p.rotation == rotation).collect();
        let mut record = report::verify(
            verifier,
            frame,
            reference,
            prior,
            Matches {
                rotation,
                pairs: selected.iter().map(|p| p.pair).collect(),
                matching_ms: 0.0,
            },
            backend,
            true,
        );
        record["origins"] = json!(selected.iter().map(|p| p.origin).collect::<Vec<_>>());
        record["source_pair_indices"] = json!(
            selected
                .iter()
                .map(|p| p.source_pair_index)
                .collect::<Vec<_>>()
        );
        record["ages"] = json!(selected.iter().map(|p| p.age).collect::<Vec<_>>());
        candidates.push(record);
    }
    candidates
}
