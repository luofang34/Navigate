//! Share image tracking but preserve each candidate's map coordinates.
use super::Runner;
use crate::{
    Error,
    surface::{
        acquire::Acquired,
        job::window,
        sequence::tracks::{MAX_TRACK_GAP_NS, retain_inliers},
    },
};
use navigate_visual::{Frame, OrthographicEstimate, PixelMatch};
use serde_json::{Value, json};
use std::time::Instant;

pub(super) struct Hypothesis {
    pub id: usize,
    pub window: Option<[u32; 4]>,
    pub rotation: u8,
    pub pairs: Vec<PixelMatch>,
    pub estimate: Option<OrthographicEstimate>,
}
impl Runner {
    pub(super) fn initialize(&mut self, candidates: Vec<Acquired>) -> Result<Value, Error> {
        let started = Instant::now();
        for (id, candidate) in candidates.into_iter().enumerate() {
            let crop = candidate
                .window
                .map(|area| window::crop_borrowed(&self.reference, area, self.reference_limit))
                .transpose()?;
            let pairs = retain_inliers(
                &self.previous,
                crop.as_ref().unwrap_or(&self.reference),
                &candidate.estimate,
                candidate.pairs,
            );
            self.hypotheses.push(Hypothesis {
                id,
                window: candidate.window,
                rotation: candidate.rotation,
                pairs,
                estimate: Some(candidate.estimate),
            });
        }
        Ok(
            json!({"initialization_ms":started.elapsed().as_secs_f64()*1000.0,"hypotheses":self.hypotheses.iter().map(|h| super::report::hypothesis(h, Value::Null)).collect::<Vec<_>>()}),
        )
    }
    pub(super) fn advance_blocking(&mut self, frame: &Frame) -> Result<Value, Error> {
        let started = Instant::now();
        if frame.camera != self.previous.camera
            || frame.stamp.sequence <= self.previous.stamp.sequence
            || frame.stamp.capture_time_ns <= self.previous.stamp.capture_time_ns
            || frame.stamp.capture_time_ns - self.previous.stamp.capture_time_ns > MAX_TRACK_GAP_NS
        {
            let rows: Vec<_> = self
                .hypotheses
                .iter_mut()
                .map(|h| {
                    h.estimate = None;
                    h.pairs.clear();
                    super::report::hypothesis(h, json!({"rejection":"ineligible_frame_gap"}))
                })
                .collect();
            return Ok(json!({"hypotheses":rows,"tracking_ms":0.0,"tracker_calls":0}));
        }
        let points: Vec<_> = self
            .hypotheses
            .iter()
            .flat_map(|h| h.pairs.iter().map(|p| p.query))
            .collect();
        let tracked = if points.is_empty() {
            Vec::new()
        } else {
            self.tracker.backend.track_points_blocking(
                &self.previous.image,
                &frame.image,
                &points,
            )?
        };
        if tracked.len() != points.len() {
            return Err(Error::Input(
                "hypothesis tracker returned a different feature count".into(),
            ));
        }
        let tracking_ms = started.elapsed().as_secs_f64() * 1000.0;
        let scratch_bytes = points.capacity() * std::mem::size_of::<nalgebra::Vector2<f64>>()
            + tracked.capacity() * std::mem::size_of::<Option<nalgebra::Vector2<f64>>>();
        let pairs_before = self.pair_bytes();
        let mut tracked = tracked.into_iter();
        let mut rows = Vec::new();
        for h in &mut self.hypotheses {
            let old = std::mem::take(&mut h.pairs);
            h.pairs = old
                .into_iter()
                .filter_map(|pair| {
                    tracked.next().flatten().map(|query| PixelMatch {
                        query,
                        reference: pair.reference,
                    })
                })
                .collect();
            let row = if h.estimate.is_some() {
                verify(
                    h,
                    frame,
                    &self.reference,
                    &self.prior,
                    &self.verifier,
                    &self.tracker.identity,
                    self.reference_limit,
                )?
            } else {
                json!({"rejection":"hypothesis_already_lost"})
            };
            rows.push(super::report::hypothesis(h, row));
        }
        Ok(
            json!({"hypotheses":rows,"tracking_ms":tracking_ms,"tracker_calls":usize::from(!points.is_empty()),"tracked_input_points":points.len(),
            "host_tracking_scratch_bytes":scratch_bytes,"host_pair_copy_upper_bound_bytes":2*pairs_before,
            "processing_ms":started.elapsed().as_secs_f64()*1000.0}),
        )
    }
}
fn verify(
    h: &mut Hypothesis,
    frame: &Frame,
    source: &navigate_visual::OrthographicReference,
    prior: &navigate_visual::PosePrior,
    verifier: &navigate_visual::PoseVerifier,
    backend: &str,
    limit: usize,
) -> Result<Value, Error> {
    let start = Instant::now();
    let crop = h
        .window
        .map(|area| window::crop_borrowed(source, area, limit))
        .transpose()?;
    let crop_ms = start.elapsed().as_secs_f64() * 1000.0;
    let reference = crop.as_ref().unwrap_or(source);
    let start = Instant::now();
    let decision = verifier.verify_orthographic(frame, reference, prior, &h.pairs, backend);
    let geometry_ms = start.elapsed().as_secs_f64() * 1000.0;
    let count = h.pairs.len();
    let start = Instant::now();
    let rejection = match decision {
        Ok(estimate) => {
            h.pairs = retain_inliers(frame, reference, &estimate, std::mem::take(&mut h.pairs));
            h.estimate = Some(estimate);
            Value::Null
        }
        Err(error) => {
            h.estimate = None;
            h.pairs.clear();
            json!(error.to_string())
        }
    };
    Ok(
        json!({"crop_ms":crop_ms,"crop_copy_bytes":crop.as_ref().map_or(0, |r| r.image.as_raw().len()*5),
        "geometry_ms":geometry_ms,"inlier_filter_ms":start.elapsed().as_secs_f64()*1000.0,"matches":count,"rejection":rejection}),
    )
}
