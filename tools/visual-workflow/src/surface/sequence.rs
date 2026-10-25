//! Conditional temporal localization on one fixed orthographic map region.
pub(super) mod continuity;
pub(super) mod input;
mod keyframe;
pub(super) mod tracks;
use super::{job::SurfaceJob, matching::Matcher};
use crate::{
    Error, assets,
    config::Config,
    evaluate::tracker::{Tracker, TrackerBackend},
};
use navigate_recording::Pose;
use navigate_visual::{
    Frame, FrameStamp, LocalizerConfig, OrthographicEstimate, OrthographicReference, PixelMatch,
    PosePrior, PoseVerifier,
};
use serde_json::{Value, json};
use std::{path::Path, time::Instant};
use tracks::Tracks;

type Decision = (OrthographicEstimate, Vec<PixelMatch>);
struct Runner {
    anchoring: crate::anchoring::Anchoring,
    continuity: continuity::Continuity,
    matcher: Matcher,
    tracker: Tracker,
    verifier: PoseVerifier,
    reference: OrthographicReference,
    prior: PosePrior,
    tracks: Option<Tracks>,
    interval: usize,
    record_matches: bool,
}
pub fn run_blocking(
    config: &Config,
    root: &Path,
    manifest: &Path,
    output: &Path,
) -> Result<(), Error> {
    let total = Instant::now();
    let spec: input::Manifest = assets::read_json_blocking(manifest)?;
    spec.validate()?;
    let manifest_root = manifest.parent().unwrap_or(Path::new("."));
    let job_path = manifest_root.join(&spec.job);
    let job: SurfaceJob = assets::read_json_blocking(&job_path)?;
    let (initial, reference, prior) =
        job.load_blocking(job_path.parent().unwrap_or(manifest_root), config)?;
    let raster_bytes = reference.image.as_raw().len() * 5;
    let matcher = Matcher::new_blocking(
        config,
        root,
        &reference.image,
        &job.reference_quarter_turns,
        raster_bytes,
    )?;
    let mut runner = Runner {
        anchoring: crate::anchoring::Anchoring::new(spec.keyframe_interval),
        continuity: continuity::Continuity::new(initial.camera)?,
        matcher,
        tracker: Tracker::new_blocking(TrackerBackend::Wgpu)?,
        verifier: PoseVerifier::new(LocalizerConfig::default())?,
        reference,
        prior,
        tracks: None,
        interval: spec.refresh_interval,
        record_matches: job.record_matches,
    };
    let setup_ms = total.elapsed().as_secs_f64() * 1000.0;
    let mut records = Vec::with_capacity(spec.frames.len());
    for sample in &spec.frames {
        let started = Instant::now();
        let frame = Frame {
            camera: initial.camera,
            stamp: FrameStamp {
                sequence: sample.frame_id,
                capture_time_ns: sample.capture_time_ns,
            },
            image: assets::image_verified_blocking(
                &manifest_root.join(&sample.image.path),
                config.limits.image_pixels,
                Some(&sample.image.sha256),
            )?,
        };
        if frame.image.dimensions() != (frame.camera.width, frame.camera.height) {
            return Err(Error::Input(
                "surface sequence image does not match calibration".into(),
            ));
        }
        let loading_ms = started.elapsed().as_secs_f64() * 1000.0;
        let mut record = runner.process_blocking(frame, config)?;
        record["image"] = json!(sample.image);
        record["frame_loading_ms"] = json!(loading_ms);
        record["pipeline_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
        records.push(record);
    }
    assets::write_json_blocking(
        output,
        &json!({
            "backend":runner.matcher.identity(), "tracker":runner.tracker.identity,
            "setup_ms":setup_ms, "total_ms":total.elapsed().as_secs_f64()*1000.0,
            "reference_resident_bytes":runner.matcher.resident_bytes,
            "reference_window":job.reference.window,"job":job,"refresh_interval":spec.refresh_interval,"keyframe_interval":spec.keyframe_interval,"frames":records,
            "scope":"One supplied map region; no global retrieval or navigation fusion. Map pixels and DSM samples stay fixed between anchors. Tracking is conditional evidence. Wgpu LK is explicit; gaps over 200ms require map matching. Scheduled refresh skips tracking; a failed refresh clears the track. All rotations are checked and the accepted candidate with greatest spatial support wins, ties by input order. Setup includes source loading, models and reference encoding. Total includes setup, all frame loading and processing, excludes output writing. Independent truth must be scored separately."
        }),
    )?;
    tracing::info!(path=%output.display(), "Saved orthographic sequence evaluation");
    Ok(())
}
impl Runner {
    fn process_blocking(&mut self, frame: Frame, config: &Config) -> Result<Value, Error> {
        let observation = frame.evidence_sha256();
        let mut record = json!({"frame_id":frame.stamp.sequence,"capture_time_ns":frame.stamp.capture_time_ns,
            "observation_sha256":observation,"tracking_attempted":false,"map_attempted":false,
            "tracking_ms":0.0,"tracking_geometry_ms":0.0,"map_ms":0.0});
        let mut decision = None;
        if let Some(tracks) = &self.tracks
            && tracks.eligible(&frame, self.interval)
        {
            record["tracking_attempted"] = json!(true);
            let started = Instant::now();
            let pairs = tracks.propagate_blocking(&frame, self.tracker.backend.as_mut())?;
            record["tracking_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
            let (result, pairs) = self.fit_tracking_blocking(&frame, pairs, &mut record)?;
            match result {
                Ok(estimate) => decision = Some((estimate, pairs)),
                Err(error) => record["tracking_rejection"] = json!(error.to_string()),
            }
        }
        let refreshed = decision.is_none();
        if refreshed {
            record["map_attempted"] = json!(true);
            let started = Instant::now();
            let (selected, candidates, extraction_ms) = self.map_blocking(&frame, config)?;
            record["map_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
            record["map_query_extraction_ms"] = json!(extraction_ms);
            record["map_candidates"] = candidates;
            decision = selected;
        }
        record["accepted"] = json!(decision.is_some());
        record["source"] = json!(if decision.is_none() {
            "rejected"
        } else if refreshed {
            "map"
        } else {
            "track"
        });
        record["motion_continuity"] = self.continuity.step(
            &frame,
            &observation,
            decision.as_ref().map(|(estimate, _)| estimate),
            refreshed,
        )?;
        self.tracks = match decision {
            Some((estimate, pairs)) => {
                record["pose"] = json!(Pose::from(estimate.pose));
                record["quality"] = quality(&estimate);
                let tracks = Tracks::new(
                    frame,
                    &self.reference,
                    &estimate,
                    pairs,
                    self.tracks.as_ref(),
                    refreshed,
                );
                record["anchor_observation_sha256"] = json!(tracks.origin);
                self.anchoring
                    .observe(&tracks.previous, &tracks.pairs, refreshed);
                record["retained_tracks"] = json!(tracks.pairs.len());
                Some(tracks)
            }
            None => {
                self.anchoring.clear();
                None
            }
        };
        Ok(record)
    }
    fn map_blocking(
        &mut self,
        frame: &Frame,
        config: &Config,
    ) -> Result<(Option<Decision>, Value, f64), Error> {
        let (extraction_ms, candidates) = self.matcher.matches_blocking(&frame.image, config)?;
        let mut selected: Option<Decision> = None;
        let mut records = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let started = Instant::now();
            let fit = self.verifier.verify_orthographic(
                frame,
                &self.reference,
                &self.prior,
                &candidate.pairs,
                self.matcher.identity(),
            );
            let mut record = json!({"reference_quarter_turns":candidate.rotation,"matches":candidate.pairs.len(),
                "matching_ms":candidate.matching_ms,"geometry_ms":started.elapsed().as_secs_f64()*1000.0,"accepted":fit.is_ok()});
            if self.record_matches {
                record["correspondences"] = json!(
                    candidate
                        .pairs
                        .iter()
                        .map(|p| [p.reference.x, p.reference.y, p.query.x, p.query.y])
                        .collect::<Vec<_>>()
                );
            }
            match fit {
                Ok(estimate) => {
                    record["pose"] = json!(Pose::from(estimate.pose));
                    record["quality"] = quality(&estimate);
                    if selected.as_ref().is_none_or(|(old, _)| {
                        estimate.quality.spatial_support > old.quality.spatial_support
                    }) {
                        selected = Some((estimate, candidate.pairs));
                    }
                }
                Err(error) => record["rejection"] = json!(error.to_string()),
            }
            records.push(record);
        }
        Ok((selected, json!(records), extraction_ms))
    }
}
fn quality(estimate: &OrthographicEstimate) -> Value {
    json!({"inliers":estimate.quality.inliers,"spatial_support":estimate.quality.spatial_support,
        "occupied_cells":estimate.quality.occupied_cells,"rms_px":estimate.quality.reprojection_rms_px})
}
