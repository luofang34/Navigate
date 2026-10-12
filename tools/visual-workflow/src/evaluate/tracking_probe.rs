//! Compare trackers on identical temporal inputs without updating live state.
use super::{Runner, Tracker, TrackerBackend};
use crate::Error;
use navigate_recording::Pose;
use navigate_visual::{Estimate, Frame, PixelMatch, PointTracker};
use serde_json::{Value, json};
use std::time::Instant;

pub(super) fn valid_frames(indices: &[usize], count: usize) -> bool {
    indices.len() <= 16
        && indices.iter().all(|&i| i < count)
        && indices.windows(2).all(|p| p[0] < p[1])
}

impl Runner {
    pub(super) fn record_tracking_probe_blocking(
        &self,
        index: usize,
        frame: &Frame,
        record: &mut Value,
    ) -> Result<(), Error> {
        if !self.experiment.tracking_probe_frames.contains(&index) {
            return Ok(());
        }
        let Some(tracks) = &self.tracks else {
            record["tracking_probe"] = json!({"unavailable":"no preceding track state"});
            return Ok(());
        };
        let limit = if self.experiment.retain_valid_track {
            usize::MAX
        } else {
            self.experiment.refresh_interval
        };
        if !tracks.eligible(frame, limit) {
            record["tracking_probe"] = json!({"unavailable":"preceding track state is ineligible"});
            return Ok(());
        }
        let start = Instant::now();
        let (previous, pairs) = tracks.inputs();
        let mut arms = Vec::new();
        for backend in [TrackerBackend::Cpu, TrackerBackend::Wgpu] {
            let mut tracker = Tracker::new_blocking(backend)?;
            let (mut arm, propagated) =
                propagate_blocking(previous, frame, pairs, tracker.backend.as_mut())?;
            arm["tracker"] = json!(tracker.identity);
            let fit_start = Instant::now();
            arm["fit"] = decision(self.fit(frame, &propagated));
            arm["geometry_ms"] = json!(fit_start.elapsed().as_secs_f64() * 1000.0);
            arms.push(arm);
        }
        record["tracking_probe"] = json!({
            "previous_index":previous.stamp.sequence,
            "previous_capture_time_ns":previous.stamp.capture_time_ns,
            "previous_observation_sha256":previous.evidence_sha256(),
            "query_observation_sha256":frame.evidence_sha256(),
            "input_pairs":pairs.iter().enumerate().map(|(slot,p)|json!({"slot":slot,"map":[p.reference.x,p.reference.y],"previous":[p.query.x,p.query.y]})).collect::<Vec<_>>(),
            "arms":arms,"elapsed_ms":start.elapsed().as_secs_f64()*1000.0,
            "scope":"Both trackers use identical previous/query pixels, calibration, ordered map identities and previous query points. Slot identities are local to this probe; map pixels persist within an anchor. Both fits use the unchanged verifier, fixed reference pose and job admission prior. No arm updates live tracking or selects a pose. Time belongs to validation, excluded from pipeline_ms; fresh diagnostic devices are not a performance comparison. No independent accuracy is asserted."
        });
        Ok(())
    }
}

fn propagate_blocking(
    previous: &Frame,
    frame: &Frame,
    pairs: &[PixelMatch],
    tracker: &mut dyn PointTracker,
) -> Result<(Value, Vec<PixelMatch>), Error> {
    let points: Vec<_> = pairs.iter().map(|p| p.query).collect();
    let start = Instant::now();
    let tracked = tracker.track_points_blocking(&previous.image, &frame.image, &points)?;
    if tracked.len() != pairs.len() {
        return Err(Error::Input(
            "diagnostic tracker changed output slot count".into(),
        ));
    }
    let rows: Vec<_> = tracked
        .iter()
        .enumerate()
        .map(|(slot, q)| json!({"slot":slot,"query":q.map(|q|[q.x,q.y])}))
        .collect();
    let propagated: Vec<_> = pairs
        .iter()
        .zip(tracked)
        .filter_map(|(p, q)| {
            q.map(|query| PixelMatch {
                reference: p.reference,
                query,
            })
        })
        .collect();
    Ok((
        json!({"slots":rows,"survived":propagated.len(),"tracking_ms":start.elapsed().as_secs_f64()*1000.0}),
        propagated,
    ))
}

fn decision(result: Result<Estimate, Error>) -> Value {
    match result {
        Ok(e) => json!({"accepted":true,"pose":Pose::from(e.pose),"quality":{
            "inliers":e.quality.inliers,"support":e.quality.spatial_support,
            "occupied_cells":e.quality.occupied_cells,"rms_px":e.quality.reprojection_rms_px}}),
        Err(e) => json!({"accepted":false,"rejection":e.to_string()}),
    }
}

#[cfg(test)]
mod tests;
