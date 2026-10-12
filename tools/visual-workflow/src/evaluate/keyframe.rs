//! Evaluate corrected pixels first; only a failed proposal needs the prediction fit.
use super::{Outcome, Runner};
use crate::Error;
use navigate_visual::Frame;
use serde_json::{Value, json};
use std::time::Instant;

impl Runner {
    pub(super) fn fit_tracking_blocking(
        &mut self,
        frame: &Frame,
        result: &mut Outcome,
    ) -> Result<(), Error> {
        let started = Instant::now();
        let correction =
            self.anchoring
                .correct_blocking(frame, &result.pairs, self.tracker.backend.as_mut())?;
        if let Some(mut correction) = correction {
            result.timings.tracking_ms += started.elapsed().as_secs_f64() * 1000.0;
            correction.report["selected"] = json!(false);
            correction.report["prediction_fit_evaluated"] = json!(false);
            if correction.report["corrected_points"].as_u64() != Some(0) {
                let started = Instant::now();
                let decision = self.fit(frame, &correction.pairs);
                result.timings.geometry_ms += started.elapsed().as_secs_f64() * 1000.0;
                correction.report["selected"] = json!(decision.is_ok());
                correction.report["geometry_rejection"] =
                    json!(decision.as_ref().err().map(ToString::to_string));
                if decision.is_ok() {
                    result.pairs = correction.pairs;
                    result.decision = decision;
                    result.correction = Some(correction.report);
                    return Ok(());
                }
            }
            result.correction = Some(correction.report);
        }
        let started = Instant::now();
        result.decision = self.fit(frame, &result.pairs);
        result.timings.geometry_ms += started.elapsed().as_secs_f64() * 1000.0;
        if let Some(report) = &mut result.correction {
            report["prediction_fit_evaluated"] = json!(true);
            report["prediction_rejection"] =
                json!(result.decision.as_ref().err().map(ToString::to_string));
        }
        Ok(())
    }
}

pub(super) fn record_decisions(
    record: &mut Value,
    rejection: Option<String>,
    correction: Option<Value>,
) {
    if let Some(reason) = rejection {
        record["tracking_rejection"] = json!(reason);
    }
    if let Some(correction) = correction {
        record["keyframe_correction"] = correction;
    }
}
