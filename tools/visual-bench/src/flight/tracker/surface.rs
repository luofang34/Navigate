//! Track immutable conditional surface points through consecutive camera images.
use super::*;
use navigate_visual::{PointTracker, SurfaceTracks};

pub(super) fn relative(
    tracks: &mut Option<SurfaceTracks>,
    matcher: &mut dyn PointTracker,
    verifier: &PoseVerifier,
    frame: &Frame,
    reference: TrackingReference<'_>,
    prior: &PosePrior,
    timing: &mut Timing,
) -> Result<(Value, Option<CameraPose>), BenchError> {
    let mut report = json!({"accepted":false,"tracking_supported":false,"motion_assumption":"free pose","backend":matcher.identity()});
    let started = Instant::now();
    let initialize = || -> Result<SurfaceTracks, navigate_visual::VisualError> {
        let points = matcher.features_blocking(&reference.observation.image)?;
        SurfaceTracks::new(reference.observation, reference.surface, &points)
    };
    if tracks.is_none() {
        let mut initialize = initialize;
        match initialize() {
            Ok(value) => *tracks = Some(value),
            Err(e) => {
                report["reason"] = e.to_string().into();
                return Ok((report, None));
            }
        }
    }
    let Some(tracks) = tracks else {
        return Ok((report, None));
    };
    if tracks.len() < 400 {
        let added = matcher
            .features_blocking(&reference.observation.image)
            .and_then(|points| tracks.replenish(reference.surface, &points));
        if let Err(error) = added {
            report["reason"] = error.to_string().into();
            return Ok((report, None));
        }
    }
    let points = tracks.pixels();
    let locations =
        matcher.track_points_blocking(&tracks.observation().image, &frame.image, &points);
    timing.matching_ms += started.elapsed().as_secs_f64() * 1000.0;
    timing.fast_runs = timing.fast_runs.wrapping_add(1);
    let locations = match locations {
        Ok(value) => value,
        Err(e) => {
            report["reason"] = e.to_string().into();
            return Ok((report, None));
        }
    };
    let started = Instant::now();
    let result = tracks.update(verifier, frame, prior, &locations, matcher.identity());
    timing.geometry_ms += started.elapsed().as_secs_f64() * 1000.0;
    let pose = match result {
        Ok(update) => {
            let proposal = update.proposal;
            report = pose_report(proposal.pose, proposal.frame, proposal.quality);
            report["accepted"] = false.into();
            report["tracking_supported"] = true.into();
            report["backend"] = proposal.backend.into();
            report["reference_observation_sha256"] = proposal.reference_observation_sha256.into();
            report["depth_observation_sha256s"] = json!(update.depth_observations);
            report["motion_assumption"] = "free pose".into();
            report["acceptance_stage"] = "conditional_persistent_surface_tracks".into();
            Some(proposal.pose)
        }
        Err(e) => {
            report["reason"] = e.to_string().into();
            None
        }
    };
    report["image_correspondences"] = locations.iter().filter(|p| p.is_some()).count().into();
    provenance(&mut report, reference.surface);
    Ok((report, pose))
}
