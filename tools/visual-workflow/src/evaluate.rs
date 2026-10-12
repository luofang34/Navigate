//! Compare map matching with bounded temporal reuse on spatially withheld evidence.
mod backend;
mod focal;
mod initialization;
mod metrics;
mod navigation;
mod position;
mod temporal;
mod tracker;

use crate::{
    Error, assets,
    config::Config,
    job::{Image, Job},
};
use backend::Matcher;
use navigate_recording::Pose;
use navigate_visual::{
    CameraModel, Estimate, Frame, LocalizerConfig, PixelMatch, PoseVerifier, ReferenceView,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{path::Path, time::Instant};
use temporal::Tracks;
use tracker::{Tracker, TrackerBackend};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Sample {
    image: Image,
    capture_time_ns: u64,
    #[serde(default)]
    controls: Option<Vec<[f64; 4]>>,
    position_check: Option<position::PositionCheck>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Experiment {
    job: std::path::PathBuf,
    frames: Vec<Sample>,
    fold: Option<u32>,
    refresh_interval: usize,
    query_focal_scale: f64,
    query_transform: Option<metrics::ImageTransform>,
    navigation: Option<navigation::Settings>,
    focal_probe_scales: Option<Vec<f64>>,
    #[serde(default)]
    retain_valid_track: bool,
    #[serde(default)]
    tracking_backend: TrackerBackend,
    #[serde(default)]
    max_map_track_disagreement_m: Option<f64>,
    #[serde(default)]
    initial_match_order: initialization::MatchOrder,
}
#[derive(Default, serde::Serialize)]
struct StageTimings {
    tracking_ms: f64,
    geometry_ms: f64,
    map_matching_ms: f64,
    navigation_ms: f64,
}
struct Outcome {
    evidence: navigation::Evidence,
    timings: StageTimings,
    decision: Result<Estimate, Error>,
    pairs: Vec<PixelMatch>,
    matches: Option<Vec<PixelMatch>>,
    tracking_attempted: bool,
    map_attempted: bool,
    refreshed: bool,
    map_rejection: Option<String>,
    disagreement_m: Option<f64>,
}
struct Runner {
    navigation: Option<navigation::Navigation>,
    matcher: Matcher,
    tracker: Tracker,
    verifier: PoseVerifier,
    tracks: Option<Tracks>,
    reference: ReferenceView,
    reference_camera: CameraModel,
    query_camera: CameraModel,
    job: Job,
    experiment: Experiment,
}

pub fn run_blocking(
    config: &Config,
    root: &Path,
    manifest: &Path,
    output: &Path,
) -> Result<(), Error> {
    let experiment: Experiment = assets::read_json_blocking(manifest)?;
    validate(&experiment)?;
    let manifest_root = manifest.parent().unwrap_or(Path::new("."));
    let job_path = manifest_root.join(&experiment.job);
    let job: Job = assets::read_json_blocking(&job_path)?;
    if job.references.len() != 1 {
        return Err(Error::Input(
            "evaluation requires exactly one fixed reference".into(),
        ));
    }
    let job_root = job_path.parent().unwrap_or(Path::new("."));
    let query_camera = metrics::prepare_query_camera(
        job.camera.model(),
        experiment.query_transform,
        experiment.query_focal_scale,
    )?;
    let start = Instant::now();
    let mut source = job.references_blocking(job_root, config.limits.image_pixels)?;
    let reference = job.reference_blocking(
        &job.references[0],
        job_root,
        config.limits.image_pixels,
        &mut source,
    )?;
    let matcher = Matcher::new_blocking(config.load_blocking(root)?, &reference.image)?;
    let tracker = Tracker::new_blocking(experiment.tracking_backend)?;
    let navigation = experiment
        .navigation
        .map(|s| navigation::Navigation::new(query_camera, s))
        .transpose()?;
    let setup_ms = start.elapsed().as_secs_f64() * 1000.0;
    let mut runner = Runner {
        navigation,
        matcher,
        tracker,
        verifier: PoseVerifier::new(LocalizerConfig::default())?,
        tracks: None,
        query_camera,
        reference,
        reference_camera: job.references[0]
            .camera
            .as_ref()
            .unwrap_or(&job.camera)
            .model(),
        job,
        experiment,
    };
    let mut records = Vec::new();
    for index in 0..runner.experiment.frames.len() {
        records.push(runner.frame_blocking(index, manifest_root, config.limits.image_pixels)?);
    }
    let mut report = json!({"backend":runner.matcher.identity(),
        "setup_ms":setup_ms,"fold":runner.experiment.fold,
        "holdout_enabled":runner.experiment.fold.is_some(),
        "refresh_interval":runner.experiment.refresh_interval,
        "query_focal_scale":runner.experiment.query_focal_scale,"frames":records,
        "query_transform":runner.experiment.query_transform,
        "navigation":runner.experiment.navigation,
        "focal_probe_scales":runner.experiment.focal_probe_scales,
        "query_camera":{"size":[runner.query_camera.width,runner.query_camera.height],
        "focal":[runner.query_camera.fx,runner.query_camera.fy],
        "principal":[runner.query_camera.cx,runner.query_camera.cy]},
        "retain_valid_track":runner.experiment.retain_valid_track,
        "max_map_track_disagreement_m":runner.experiment.max_map_track_disagreement_m,
        "scope":"Conditional local navigation with an existing map hypothesis. When enabled, spatial holdout is fixed across the sequence. Validation matching is excluded from pipeline_ms. No geographic ground truth or independent confidence is asserted."});
    if runner.experiment.initial_match_order != initialization::MatchOrder::Original {
        report["initial_match_order"] = json!(runner.experiment.initial_match_order);
    }
    assets::write_json_blocking(output, &report)?;
    tracing::info!(path=%output.display(),frames=runner.experiment.frames.len(),"Saved pipeline evaluation");
    Ok(())
}
fn validate(experiment: &Experiment) -> Result<(), Error> {
    let invalid_controls = experiment.frames.iter().any(|s| {
        s.controls.as_ref().is_some_and(|c| {
            experiment.fold.is_none()
                || c.len() > 16384
                || c.iter()
                    .flatten()
                    .any(|v| !v.is_finite() || !(0.0..8192.0).contains(v))
        })
    });
    if experiment
        .max_map_track_disagreement_m
        .is_some_and(|v| !v.is_finite() || v <= 0.0)
        || experiment.navigation.is_some_and(|s| !s.valid())
        || experiment
            .focal_probe_scales
            .as_ref()
            .is_some_and(|s| !focal::valid_scales(s))
        || invalid_controls
        || experiment
            .frames
            .iter()
            .any(|s| s.position_check.as_ref().is_some_and(|p| !p.valid()))
        || experiment.frames.is_empty()
        || experiment.frames.len() > 1000
        || experiment.fold.is_some_and(|fold| fold >= 5)
        || !(1..=1000).contains(&experiment.refresh_interval)
        || !experiment.query_focal_scale.is_finite()
        || !(0.8..=1.2).contains(&experiment.query_focal_scale)
        || experiment
            .frames
            .windows(2)
            .any(|p| p[1].capture_time_ns <= p[0].capture_time_ns)
    {
        return Err(Error::Input(
            "invalid evaluation bounds or frame timestamps".into(),
        ));
    }
    Ok(())
}
impl Runner {
    fn fit(&self, frame: &Frame, pairs: &[PixelMatch]) -> Result<Estimate, Error> {
        Ok(self.verifier.verify_calibrated_reference(
            frame,
            &self.reference,
            &self.reference_camera,
            &self.job.prior()?,
            pairs,
            self.matcher.identity(),
        )?)
    }
    fn training(&self, pairs: &[PixelMatch]) -> Vec<PixelMatch> {
        pairs
            .iter()
            .filter(|p| {
                self.experiment
                    .fold
                    .is_none_or(|fold| metrics::partition(p, fold) == Some(false))
            })
            .copied()
            .collect()
    }
    fn load_frame_blocking(
        &self,
        index: usize,
        root: &Path,
        pixels: usize,
    ) -> Result<Frame, Error> {
        let sample = &self.experiment.frames[index];
        let frame = Frame {
            camera: self.query_camera,
            stamp: navigate_visual::FrameStamp {
                sequence: index as u64,
                capture_time_ns: sample.capture_time_ns,
            },
            image: assets::image_verified_blocking(
                &root.join(&sample.image.path),
                pixels,
                Some(&sample.image.sha256),
            )?,
        };
        Ok(frame)
    }
    fn controls_blocking(
        &mut self,
        index: usize,
        frame: &Frame,
        all: Option<Vec<PixelMatch>>,
    ) -> Result<Vec<PixelMatch>, Error> {
        if self.experiment.fold.is_none() {
            return Ok(Vec::new());
        }
        let controls = self.experiment.frames[index].controls.as_ref();
        let matches = match (controls, all) {
            (Some(controls), _) => controls
                .iter()
                .map(|p| PixelMatch {
                    reference: nalgebra::Vector2::new(p[0], p[1]),
                    query: nalgebra::Vector2::new(p[2], p[3]),
                })
                .collect(),
            (None, Some(p)) => p,
            (None, None) => self
                .matcher
                .matches_blocking(&self.reference.image, &frame.image)?,
        };
        let withheld: Vec<_> = matches
            .into_iter()
            .filter(|p| {
                self.experiment
                    .fold
                    .is_some_and(|fold| metrics::partition(p, fold) == Some(true))
            })
            .collect();
        Ok(withheld)
    }
    fn process_blocking(&mut self, frame: &Frame) -> Result<Outcome, Error> {
        let mut result = Outcome {
            evidence: navigation::Evidence::default(),
            timings: StageTimings::default(),
            decision: Err(Error::Input("map initialization required".into())),
            pairs: Vec::new(),
            matches: None,
            tracking_attempted: false,
            map_attempted: false,
            refreshed: false,
            map_rejection: None,
            disagreement_m: None,
        };
        let interval = self.experiment.refresh_interval;
        let limit = if self.experiment.retain_valid_track {
            usize::MAX
        } else {
            interval
        };
        if let Some(tracks) = &self.tracks
            && tracks.eligible(frame, limit)
        {
            result.tracking_attempted = true;
            let start = Instant::now();
            result.pairs = tracks.propagate_blocking(frame, self.tracker.backend.as_mut())?;
            result.timings.tracking_ms = start.elapsed().as_secs_f64() * 1000.0;
            let start = Instant::now();
            result.decision = self.fit(frame, &result.pairs);
            if self.navigation.is_some() {
                result.evidence.tracked = result.decision.as_ref().ok().cloned();
            }
            result.timings.geometry_ms = start.elapsed().as_secs_f64() * 1000.0;
        }
        let scheduled = self
            .tracks
            .as_ref()
            .is_some_and(|t| t.refresh_due(interval));
        if result.decision.is_ok() && !scheduled {
            return Ok(result);
        }
        self.refresh_blocking(frame, &mut result)?;
        Ok(result)
    }
    fn refresh_blocking(&mut self, frame: &Frame, result: &mut Outcome) -> Result<(), Error> {
        result.map_attempted = true;
        let start = Instant::now();
        let mut matches = self
            .matcher
            .matches_blocking(&self.reference.image, &frame.image)?;
        result.timings.map_matching_ms = start.elapsed().as_secs_f64() * 1000.0;
        self.experiment
            .initial_match_order
            .apply(frame.stamp.sequence, &mut matches);
        let pairs = self.training(&matches);
        let start = Instant::now();
        let decision = self.fit(frame, &pairs);
        if self.navigation.is_some() {
            result.evidence.matched = decision.as_ref().ok().cloned();
        }
        result.timings.geometry_ms += start.elapsed().as_secs_f64() * 1000.0;
        let (replace, disagreement) = select_map(
            &result.decision,
            &decision,
            self.experiment.max_map_track_disagreement_m,
        );
        result.disagreement_m = disagreement;
        result.map_rejection = decision.as_ref().err().map(ToString::to_string);
        if replace {
            result.refreshed = decision.is_ok();
            result.decision = decision;
            result.pairs = pairs;
        }
        result.matches = Some(matches);
        Ok(())
    }
    fn position_checks(
        &self,
        index: usize,
        frame: &Frame,
        decision: &Result<Estimate, Error>,
        navigation: Option<navigation::Output>,
        record: &mut Value,
    ) {
        let check = self.experiment.frames[index].position_check.as_ref();
        if let Some(check) = check {
            record["position_check"] =
                check.compare(frame.stamp.capture_time_ns, decision.as_ref().ok());
        }
        if let Some(mut output) = navigation {
            if let Some(check) = check {
                output.record["position_check"] =
                    check.compare_position(frame.stamp.capture_time_ns, output.position);
            }
            record["navigation"] = output.record;
        }
    }
    fn advance_navigation(
        &mut self,
        frame: &Frame,
        outcome: &mut Outcome,
    ) -> Result<Option<navigation::Output>, Error> {
        let navigation_start = Instant::now();
        let navigation = self
            .navigation
            .as_mut()
            .map(|s| s.step(frame, &outcome.evidence, outcome.decision.as_ref().ok()))
            .transpose()?;
        if navigation.is_some() {
            outcome.timings.navigation_ms = navigation_start.elapsed().as_secs_f64() * 1000.0;
        }
        Ok(navigation)
    }
    fn frame_blocking(&mut self, index: usize, root: &Path, pixels: usize) -> Result<Value, Error> {
        let start = Instant::now();
        let frame = self.load_frame_blocking(index, root, pixels)?;
        self.reference.validate_camera(&self.reference_camera)?;
        let frame_loading_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut outcome = self.process_blocking(&frame)?;
        let navigation = self.advance_navigation(&frame, &mut outcome)?;
        let Outcome {
            evidence: _,
            timings,
            decision,
            pairs,
            matches,
            tracking_attempted,
            map_attempted,
            refreshed,
            map_rejection,
            disagreement_m,
        } = outcome;
        let pipeline_ms = start.elapsed().as_secs_f64() * 1000.0;
        let validation_start = Instant::now();
        let (withheld, probe) = self.validation_blocking(index, &frame, matches)?;
        let mut record = json!({"index":index,"capture_time_ns":frame.stamp.capture_time_ns,
            "observation_sha256":frame.evidence_sha256(),"pipeline_ms":pipeline_ms,
            "stage_ms":timings,"frame_loading_ms":frame_loading_ms,
            "tracking_attempted":tracking_attempted,"map_refreshed":refreshed,"map_attempted":map_attempted,
            "map_rejection":map_rejection,"map_track_disagreement_m":disagreement_m,
            "evidence_kind":if decision.is_err() {"rejected"} else if refreshed {"map_match"} else {"propagated_map_matches"},
            "tracking_backend":self.tracker.identity,
            "controls":withheld.iter().map(|p|[p.reference.x,p.reference.y,p.query.x,p.query.y]).collect::<Vec<_>>(),
            "frozen_controls":self.experiment.frames[index].controls.is_some(),
            "training_matches":pairs.len(),"heldout_matches":withheld.len(),"accepted":decision.is_ok()});
        record["focal_probe"] = json!(probe);
        self.position_checks(index, &frame, &decision, navigation, &mut record);
        match decision {
            Ok(estimate) => {
                record["pose"] = json!(Pose::from(estimate.pose));
                record["geodetic"] = json!(estimate.frame.geodetic(estimate.pose.position));
                record["support"] = json!(estimate.quality.spatial_support);
                record["heldout"] = metrics::score(
                    &withheld,
                    &frame.camera,
                    (&self.reference, &self.reference_camera),
                    &estimate.pose,
                );
                let update_start = Instant::now();
                self.tracks = Some(Tracks::update(
                    frame,
                    pairs,
                    (&self.reference, &self.reference_camera),
                    &estimate,
                    self.tracks.as_ref(),
                    refreshed,
                ));
                let update_ms = update_start.elapsed().as_secs_f64() * 1000.0;
                record["state_update_ms"] = json!(update_ms);
                record["pipeline_ms"] = json!(pipeline_ms + update_ms);
            }
            Err(error) => {
                record["rejection"] = json!(error.to_string());
                self.tracks = None;
            }
        }
        record["validation_ms"] = json!(
            (validation_start.elapsed().as_secs_f64() * 1000.0
                - record["state_update_ms"].as_f64().unwrap_or(0.0))
            .max(0.0)
        );
        Ok(record)
    }
}

fn select_map(
    track: &Result<Estimate, Error>,
    map: &Result<Estimate, Error>,
    limit: Option<f64>,
) -> (bool, Option<f64>) {
    let distance = match (track, map) {
        (Ok(track), Ok(map)) => Some((track.pose.position - map.pose.position).norm()),
        _ => None,
    };
    let conflict = distance
        .zip(limit)
        .is_some_and(|(distance, limit)| distance > limit);
    (track.is_err() || (map.is_ok() && !conflict), distance)
}
#[cfg(test)]
mod tests;
