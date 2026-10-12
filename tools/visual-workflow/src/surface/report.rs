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
    let start = Instant::now();
    let decision =
        verifier.verify_orthographic(frame, reference, prior, &candidate.pairs, identity);
    let mut report = json!({"reference_quarter_turns":candidate.rotation,"matching_ms":candidate.matching_ms,"geometry_ms":start.elapsed().as_secs_f64()*1000.0,"matches":candidate.pairs.len(),"accepted":decision.is_ok()});
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

#[cfg(test)]
mod tests;
