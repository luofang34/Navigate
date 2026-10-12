//! Evaluate corrected geometry before paying for an unused prediction fit.
use super::Runner;
use crate::Error;
use navigate_visual::{Frame, OrthographicEstimate, PixelMatch, VisualError};
use serde_json::{Value, json};
use std::time::Instant;

impl Runner {
    pub(super) fn fit_tracking_blocking(
        &mut self,
        frame: &Frame,
        pairs: Vec<PixelMatch>,
        record: &mut Value,
    ) -> Result<(Result<OrthographicEstimate, VisualError>, Vec<PixelMatch>), Error> {
        let started = Instant::now();
        let correction =
            self.anchoring
                .correct_blocking(frame, &pairs, self.tracker.backend.as_mut())?;
        if let Some(mut correction) = correction {
            record["tracking_ms"] = json!(
                record["tracking_ms"].as_f64().unwrap_or(0.0)
                    + started.elapsed().as_secs_f64() * 1000.0
            );
            correction.report["selected"] = json!(false);
            correction.report["prediction_fit_evaluated"] = json!(false);
            if correction.report["corrected_points"].as_u64() != Some(0) {
                let decision = self.fit_track(frame, &correction.pairs, record);
                correction.report["selected"] = json!(decision.is_ok());
                correction.report["geometry_rejection"] =
                    json!(decision.as_ref().err().map(ToString::to_string));
                if decision.is_ok() {
                    record["keyframe_correction"] = correction.report;
                    return Ok((decision, correction.pairs));
                }
            }
            record["keyframe_correction"] = correction.report;
        }
        let decision = self.fit_track(frame, &pairs, record);
        if let Some(report) = record.get_mut("keyframe_correction") {
            report["prediction_fit_evaluated"] = json!(true);
            report["prediction_rejection"] =
                json!(decision.as_ref().err().map(ToString::to_string));
        }
        Ok((decision, pairs))
    }
    fn fit_track(
        &self,
        frame: &Frame,
        pairs: &[PixelMatch],
        record: &mut Value,
    ) -> Result<OrthographicEstimate, VisualError> {
        let started = Instant::now();
        let decision = self.verifier.verify_orthographic(
            frame,
            &self.reference,
            &self.prior,
            pairs,
            &self.tracker.identity,
        );
        record["tracking_geometry_ms"] = json!(
            record["tracking_geometry_ms"].as_f64().unwrap_or(0.0)
                + started.elapsed().as_secs_f64() * 1000.0
        );
        decision
    }
}
