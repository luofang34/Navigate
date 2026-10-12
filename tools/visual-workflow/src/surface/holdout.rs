//! Replay frozen geometry before testing fixed spatial holdouts.
mod evidence;
mod scoring;
use super::{
    job::{SurfaceJob, window},
    matching::Matches,
    report,
    sequence::input::Manifest,
};
use crate::{Error, assets, config::Config, job::Image};
use navigate_recording::Pose;
use navigate_visual::{
    Frame, FrameStamp, LocalizerConfig, OrthographicReference, PosePrior, PoseVerifier,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{path::Path, time::Instant};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    job: Image,
    match_result: Image,
    sequence_manifest: Image,
    frame_id: u64,
}
struct Candidate {
    stage: &'static str,
    window: Option<[u32; 4]>,
    original: Value,
    matches: Matches,
}
struct Replay {
    frame: Frame,
    reference: OrthographicReference,
    prior: PosePrior,
    verifier: PoseVerifier,
    limit: usize,
}

pub fn run_blocking(config: &Config, input: &Path, output: &Path) -> Result<(), Error> {
    let total = Instant::now();
    let spec: Input = assets::read_json_blocking(input)?;
    let root = input.parent().unwrap_or(Path::new("."));
    let (runner, candidates, source_observation) = load_blocking(root, &spec, config)?;
    let loading_ms = total.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let mut controls = Vec::new();
    for candidate in &candidates {
        controls.push(runner.control(candidate)?);
    }
    let control_ms = start.elapsed().as_secs_f64() * 1000.0;
    let exact = controls
        .iter()
        .all(|c| c["exact_original_decision"] == true);
    let start = Instant::now();
    let mut rows = Vec::new();
    if exact {
        for candidate in &candidates {
            rows.push(runner.folds(candidate)?);
        }
    }
    let folds_ms = start.elapsed().as_secs_f64() * 1000.0;
    assets::write_json_blocking(
        output,
        &json!({"control_passed":exact,"input_job":spec.job,"input_result":spec.match_result,
        "sequence_manifest":spec.sequence_manifest,"frame_id":runner.frame.stamp.sequence,"capture_time_ns":runner.frame.stamp.capture_time_ns,
        "source_static_observation_sha256":source_observation,"observation_sha256":runner.frame.evidence_sha256(),
        "loading_ms":loading_ms,"control_ms":control_ms,"folds_ms":folds_ms,"total_ms":total.elapsed().as_secs_f64()*1000.0,
        "controls":controls,"candidates":rows,
        "scope":"Replay all recorded decisions before any fold fit. Keep crop-local fitting and original geometric gates. Partition full-map reference pixels into64px patches, with4px guards and five fixed folds. Fit training correspondences only. Score all valid heldout observations without residual filtering. Digests preserve ordered crop-local pairs. This measures agreement with heldout matcher output, not independent accuracy. Map, calibration, and repeated-feature errors can be shared. The restored capture stamp identifies the original observation; replay is not another measurement."}),
    )?;
    if !exact {
        return Err(Error::Input(
            "frozen geometry replay differs; saved controls without fold fits".into(),
        ));
    }
    tracing::info!(path=%output.display(), "Saved orthographic spatial holdout results");
    Ok(())
}
fn load_blocking(
    root: &Path,
    spec: &Input,
    config: &Config,
) -> Result<(Replay, Vec<Candidate>, String), Error> {
    let (job, job_path): (SurfaceJob, _) = evidence::verified_blocking(root, &spec.job)?;
    let (source, source_path): (Value, _) = evidence::verified_blocking(root, &spec.match_result)?;
    let (sequence, _): (Manifest, _) = evidence::verified_blocking(root, &spec.sequence_manifest)?;
    sequence.validate()?;
    let expected = serde_json::to_value(&job).map_err(|source| Error::Json {
        path: job_path.clone(),
        source,
    })?;
    if source["job"] != expected || job.reference.window.is_some() {
        return Err(Error::Input(
            "frozen acquisition job or full-map identity differs".into(),
        ));
    }
    let (mut frame, reference, prior) =
        job.load_blocking(job_path.parent().unwrap_or(root), config)?;
    let source_observation = frame.evidence_sha256();
    if source["observation_sha256"].as_str() != Some(&source_observation) {
        return Err(Error::Input(
            "frozen query pixels or calibration differ".into(),
        ));
    }
    let sample = sequence
        .frames
        .iter()
        .find(|s| s.frame_id == spec.frame_id)
        .ok_or_else(|| Error::Input("capture not found in sequence manifest".into()))?;
    if sample.image.sha256 != job.query.sha256 {
        return Err(Error::Input(
            "capture image differs from frozen query".into(),
        ));
    }
    frame.stamp = FrameStamp {
        sequence: sample.frame_id,
        capture_time_ns: sample.capture_time_ns,
    };
    let candidates = evidence::candidates(&source, &source_path, config.limits.candidates)?;
    Ok((
        Replay {
            frame,
            reference,
            prior,
            verifier: PoseVerifier::new(LocalizerConfig::default())?,
            limit: config.limits.reference_bytes,
        },
        candidates,
        source_observation,
    ))
}
impl Replay {
    fn region(&self, c: &Candidate) -> Result<Option<OrthographicReference>, Error> {
        c.window
            .map(|w| window::crop_borrowed(&self.reference, w, self.limit))
            .transpose()
    }
    fn control(&self, c: &Candidate) -> Result<Value, Error> {
        let started = Instant::now();
        let region = self.region(c)?;
        let crop_ms = started.elapsed().as_secs_f64() * 1000.0;
        let (decision, _) = report::verify_with_estimate(
            &self.verifier,
            &self.frame,
            region.as_ref().unwrap_or(&self.reference),
            &self.prior,
            &c.matches,
            "frozen-holdout-control",
            false,
        );
        let differences = evidence::differences(&decision, &c.original);
        Ok(
            json!({"stage":c.stage,"window":c.window,"rotation":c.matches.rotation,"original":evidence::decision(&c.original),
            "replayed":evidence::decision(&decision),"exact_original_decision":differences.is_empty(),"differences":differences,
            "geometry_ms":decision["geometry_ms"],"crop_ms":crop_ms,"raw_matches":c.matches.pairs.len(),"raw_pairs_sha256":scoring::digest(&c.matches.pairs)}),
        )
    }
    fn folds(&self, c: &Candidate) -> Result<Value, Error> {
        let started = Instant::now();
        let region = self.region(c)?;
        let crop_ms = started.elapsed().as_secs_f64() * 1000.0;
        let reference = region.as_ref().unwrap_or(&self.reference);
        let mut rows = Vec::new();
        let mut poses = Vec::new();
        for fold in 0..5 {
            let split = scoring::split(&c.matches.pairs, c.window, fold);
            let start = Instant::now();
            let matches = Matches {
                rotation: c.matches.rotation,
                pairs: split.training,
                matching_ms: 0.0,
            };
            let (mut decision, estimate) = report::verify_with_estimate(
                &self.verifier,
                &self.frame,
                reference,
                &self.prior,
                &matches,
                "frozen-spatial-holdout",
                false,
            );
            if let Ok(e) = estimate {
                let score_start = Instant::now();
                decision["heldout"] =
                    scoring::score(&split.heldout, &self.frame, reference, &e.pose);
                decision["heldout_scoring_ms"] =
                    json!(score_start.elapsed().as_secs_f64() * 1000.0);
                decision["condition_number"] = json!(e.quality.condition_number);
                decision["depth_matches"] = json!(e.quality.depth_matches);
                poses.push((fold, e.pose));
            } else {
                decision["heldout"] = Value::Null;
            }
            rows.push(json!({"fold":fold,"training_count":matches.pairs.len(),"heldout_count":split.heldout.len(),
                "boundary_count":split.boundary,"invalid_reference_count":split.invalid,"training_sha256":scoring::digest(&matches.pairs),
                "heldout_sha256":scoring::digest(&split.heldout),"decision":decision,"fold_ms":start.elapsed().as_secs_f64()*1000.0}));
        }
        let mut spread = Vec::new();
        for (i, (a, pa)) in poses.iter().enumerate() {
            for (b, pb) in poses.iter().skip(i + 1) {
                spread.push(json!({"a":a,"b":b,"position_m":(pa.position-pb.position).norm(),"angle_rad":pa.orientation.angle_to(&pb.orientation)}));
            }
        }
        Ok(
            json!({"stage":c.stage,"window":c.window,"rotation":c.matches.rotation,"original_accepted":c.original["accepted"],
            "raw_matches":c.matches.pairs.len(),"crop_ms":crop_ms,"folds":rows,"pose_disagreement":spread,
            "accepted_fold_poses":poses.iter().map(|(fold,p)|json!({"fold":fold,"pose":Pose::from(*p)})).collect::<Vec<_>>()}),
        )
    }
}
#[cfg(test)]
mod tests;
