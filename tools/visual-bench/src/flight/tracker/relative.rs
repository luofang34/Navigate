//! Conditional tracking evaluates each correspondence set on the same evidence.
use super::{Timing, error_report, pose_report, provenance};
use crate::BenchError;
use navigate_visual::{
    CameraPose, Frame, ImageMatcher, PixelMatch, PosePrior, PoseVerifier, TrackingReference,
};
use serde_json::{Value, json};
use std::time::Instant;

pub(super) struct Check<'a> {
    pub verifier: &'a PoseVerifier,
    pub prior: &'a PosePrior,
    pub dense_only: bool,
    pub fixed_tilt: bool,
}

pub(super) fn relative(
    fast: &mut dyn ImageMatcher,
    dense: &mut dyn ImageMatcher,
    frame: &Frame,
    context: TrackingReference<'_>,
    check: Check<'_>,
    timing: &mut Timing,
) -> Result<(Value, Option<CameraPose>), BenchError> {
    let mut attempts = Vec::new();
    let mut errors = Vec::new();
    let mut result = (json!({"accepted":false,"tracking_supported":false}), None);
    let mut matchers: Vec<&mut dyn ImageMatcher> = Vec::new();
    if !check.dense_only {
        matchers.push(fast);
    }
    matchers.push(dense);
    'backends: for (backend, matcher) in matchers.into_iter().enumerate() {
        for attempt in 0..4 {
            let start = Instant::now();
            let pairs = if attempt == 0 {
                matcher
                    .match_images_blocking(&context.observation.image, &frame.image)
                    .map(Some)
            } else {
                matcher.match_alternative_blocking(
                    &context.observation.image,
                    &frame.image,
                    attempt - 1,
                )
            };
            timing.matching_ms += start.elapsed().as_secs_f64() * 1000.0;
            if matches!(pairs, Ok(None)) {
                break;
            }
            if backend == 0 && !check.dense_only {
                timing.fast_runs = timing.fast_runs.wrapping_add(1);
            } else {
                timing.dense_runs = timing.dense_runs.wrapping_add(1);
            }
            let pairs = match pairs {
                Ok(Some(pairs)) => pairs,
                Ok(None) => break,
                Err(error) => {
                    errors.push(json!({"backend":matcher.identity(),"attempt":attempt,"error":error_report(&error)}));
                    result.0["acceptance_stage"] = "image_matching_error".into();
                    break;
                }
            };
            result = evaluate(matcher.identity(), frame, &context, &check, &pairs, timing);
            result.0["matcher_attempt"] = attempt.into();
            attempts.push(result.0.clone());
            if result.1.is_some() {
                break 'backends;
            }
        }
    }
    if !errors.is_empty() {
        result.0["matching_errors"] = errors.into();
    }
    result.0["matching_attempts"] = attempts.into();
    result.0["matching_evidence"] =
        "same image pair; attempts are not independent measurements".into();
    provenance(&mut result.0, context.surface);
    Ok(result)
}

fn evaluate(
    identity: &str,
    frame: &Frame,
    context: &TrackingReference<'_>,
    check: &Check<'_>,
    pairs: &[PixelMatch],
    timing: &mut Timing,
) -> (Value, Option<CameraPose>) {
    let start = Instant::now();
    let evaluation = check.verifier.track_with_motion(
        frame,
        TrackingReference {
            observation: context.observation,
            surface: context.surface,
        },
        check.prior,
        pairs,
        identity,
        if check.fixed_tilt {
            navigate_visual::TrackingMotion::FixedTilt
        } else {
            navigate_visual::TrackingMotion::Free
        },
    );
    timing.geometry_ms += start.elapsed().as_secs_f64() * 1000.0;
    let (mut report, pose) = match evaluation {
        Ok(proposal) => {
            let mut report = pose_report(proposal.pose, proposal.frame, proposal.quality);
            report["reference_observation_sha256"] = proposal.reference_observation_sha256.into();
            (report, Some(proposal.pose))
        }
        Err(error) => (json!({"reason":error.to_string()}), None),
    };
    report["accepted"] = false.into();
    report["tracking_supported"] = pose.is_some().into();
    report["backend"] = identity.into();
    report["image_correspondences"] = pairs.len().into();
    report["acceptance_stage"] = "conditional_relative_geometry".into();
    report["motion_assumption"] = if check.fixed_tilt {
        "fixed reference tilt; unmeasured assumption"
    } else {
        "free pose"
    }
    .into();
    (report, pose)
}
