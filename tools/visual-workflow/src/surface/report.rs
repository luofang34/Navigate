//! One geometry policy for exhaustive and routed image matching.
use super::matching::Matches;
use navigate_recording::Pose;
use navigate_visual::{
    Frame, OrthographicEstimate, OrthographicReference, PosePrior, PoseVerifier, VisualError,
};
use serde_json::{Value, json};
use std::time::Instant;

pub(super) fn verify(
    verifier: &PoseVerifier,
    frame: &Frame,
    reference: &OrthographicReference,
    prior: &PosePrior,
    candidate: Matches,
    identity: &str,
    record_matches: bool,
) -> Value {
    verify_with_estimate(
        verifier,
        frame,
        reference,
        prior,
        &candidate,
        identity,
        record_matches,
    )
    .0
}
pub(super) fn verify_with_estimate(
    verifier: &PoseVerifier,
    frame: &Frame,
    reference: &OrthographicReference,
    prior: &PosePrior,
    candidate: &Matches,
    identity: &str,
    record_matches: bool,
) -> (Value, Result<OrthographicEstimate, VisualError>) {
    Verification {
        verifier,
        rectification: None,
    }
    .run(frame, reference, prior, candidate, identity, record_matches)
}

pub(super) struct Verification<'a> {
    pub verifier: &'a PoseVerifier,
    pub rectification: Option<&'a navigate_visual::RectificationMap>,
}
impl Verification<'_> {
    pub fn run(
        &self,
        frame: &Frame,
        reference: &OrthographicReference,
        prior: &PosePrior,
        candidate: &Matches,
        identity: &str,
        record_matches: bool,
    ) -> (Value, Result<OrthographicEstimate, VisualError>) {
        let start = Instant::now();
        let mut decision =
            self.verifier
                .verify_orthographic(frame, reference, prior, &candidate.pairs, identity);
        let pinhole_accepted = decision.is_ok();
        let mut raw_report = Value::Null;
        if let Some(map) = self.rectification {
            raw_report = json!({"evaluated":false});
            if let Ok(estimate) = &mut decision {
                match self.verifier.check_rectified_orthographic(
                    frame,
                    reference,
                    prior,
                    &candidate.pairs,
                    map,
                    estimate,
                ) {
                    Ok(q) => {
                        raw_report = json!({"evaluated":true,"accepted":true,
                    "inliers":q.inliers,"spatial_support":q.spatial_support,
                    "query_cells":q.query_cells,"reference_cells":q.reference_cells,
                    "rms_px":q.reprojection_rms_px})
                    }
                    Err(error) => {
                        raw_report = json!({"evaluated":true,"accepted":false,"rejection":error.to_string()});
                        decision = Err(error);
                    }
                }
            }
        }
        let mut report = json!({"reference_quarter_turns":candidate.rotation,"matching_ms":candidate.matching_ms,"geometry_ms":start.elapsed().as_secs_f64()*1000.0,"matches":candidate.pairs.len(),"accepted":decision.is_ok()});
        if self.rectification.is_some() {
            report["pinhole_candidate_accepted"] = json!(pinhole_accepted);
            report["raw_pixel_gate"] = raw_report;
        }
        if record_matches {
            report["correspondences"] = json!(
                candidate
                    .pairs
                    .iter()
                    .map(|p| [p.reference.x, p.reference.y, p.query.x, p.query.y])
                    .collect::<Vec<_>>()
            );
        }
        match &decision {
            Ok(e) => {
                report["pose"] = json!(Pose::from(e.pose));
                report["quality"] = json!({"inliers":e.quality.inliers,"spatial_support":e.quality.spatial_support,"occupied_cells":e.quality.occupied_cells,"rms_px":e.quality.reprojection_rms_px});
            }
            Err(e) => report["rejection"] = Value::String(e.to_string()),
        }
        (report, decision)
    }
}

#[cfg(test)]
mod tests;
