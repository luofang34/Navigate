//! Bounded map refinement retains each attempt on the same observation.
use super::{Timing, error_report, pose_report, provenance, render};
use crate::BenchError;
use navigate_visual::{
    CameraPose, Frame, ImageMatcher, PosePrior, PoseVerifier, ReferenceRenderer, ReferenceView,
};
use serde_json::{Value, json};
use std::time::Instant;

struct Attempt {
    report: Value,
    accepted: Option<CameraPose>,
    next: Option<CameraPose>,
}

pub(super) fn map_check(
    renderer: &mut dyn ReferenceRenderer<Error = BenchError>,
    matcher: &mut dyn ImageMatcher,
    verifier: &PoseVerifier,
    frame: &Frame,
    mut reference: ReferenceView,
    prior: &PosePrior,
    timing: &mut Timing,
) -> Result<(Value, Option<CameraPose>), BenchError> {
    let mut attempts = Vec::new();
    let observation = frame.evidence_sha256();
    let mut final_report = json!({"accepted":false});
    let mut accepted = None;
    for iteration in 0..3 {
        let matching_before = timing.matching_ms;
        let geometry_before = timing.geometry_ms;
        let mut attempt = evaluate(matcher, verifier, frame, &reference, prior, timing);
        provenance(&mut attempt.report, &reference);
        attempt.report["observation_sha256"] = observation.clone().into();
        attempt.report["matching_ms"] = (timing.matching_ms - matching_before).into();
        attempt.report["geometry_ms"] = (timing.geometry_ms - geometry_before).into();
        attempts.push(attempt.report.clone());
        final_report = attempt.report;
        accepted = attempt.accepted;
        let Some(next) = attempt.next else { break };
        // A render is useful only if another match will consume it.
        if iteration == 2
            || ((next.position - reference.pose.position).norm() < 0.1
                && next.orientation.angle_to(&reference.pose.orientation) < 0.001)
        {
            break;
        }
        reference = render(renderer, next, timing)?;
    }
    final_report["refinement_attempts"] = attempts.into();
    final_report["refinement_evidence"] =
        "same observation; attempts are not independent measurements".into();
    Ok((final_report, accepted))
}

fn evaluate(
    matcher: &mut dyn ImageMatcher,
    verifier: &PoseVerifier,
    frame: &Frame,
    reference: &ReferenceView,
    prior: &PosePrior,
    timing: &mut Timing,
) -> Attempt {
    let start = Instant::now();
    let pairs = matcher.match_images_blocking(&reference.image, &frame.image);
    timing.matching_ms += start.elapsed().as_secs_f64() * 1000.0;
    timing.dense_runs = timing.dense_runs.wrapping_add(1);
    let pairs = match pairs {
        Ok(pairs) => pairs,
        Err(error) => {
            return Attempt {
                report: json!({"accepted":false,"acceptance_stage":"image_matching_error",
                    "backend":matcher.identity(),"reason":error.to_string(),"error":error_report(&error)}),
                accepted: None,
                next: None,
            };
        }
    };
    let start = Instant::now();
    let evaluation = verifier.evaluate(frame, reference, prior, &pairs, matcher.identity());
    timing.geometry_ms += start.elapsed().as_secs_f64() * 1000.0;
    let mut report = json!({"accepted":false,"acceptance_stage":"map_geometry",
        "backend":matcher.identity()});
    let (accepted, next) = match evaluation.acceptance {
        Ok(estimate) => {
            report = pose_report(estimate.pose, estimate.frame, estimate.quality);
            report["accepted"] = true.into();
            report["tracking_supported"] = false.into();
            report["backend"] = estimate.backend.into();
            report["acceptance_stage"] = "map_geometry".into();
            (Some(estimate.pose), Some(estimate.pose))
        }
        Err(error) => {
            report["reason"] = error.to_string().into();
            if let Some(seed) = evaluation.refinement {
                report["refinement_pose"] = json!({
                    "position_enu_m":seed.pose.position.as_slice(),
                    "eye_to_enu_xyzw":seed.pose.orientation.coords.as_slice(),
                    "inliers":seed.inliers,"spatial_support":seed.spatial_support,
                    "query_cells":seed.query_cells,"reference_cells":seed.reference_cells,
                    "status":"render seed only; not an accepted measurement"});
            }
            (None, evaluation.refinement.map(|seed| seed.pose))
        }
    };
    report["image_correspondences"] = pairs.len().into();
    Attempt {
        report,
        accepted,
        next,
    }
}
