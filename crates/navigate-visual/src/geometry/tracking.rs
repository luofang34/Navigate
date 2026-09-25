//! Relative image tracking, conditional on an estimated surface pose.
use super::PoseVerifier;
use crate::{
    CameraPose, EstimateQuality, Frame, LocalFrame, MapRevision, PixelMatch, PosePrior,
    ReferenceView, VisualError,
};

/// A previous observation and depth rendered at its estimated pose.
///
/// The surface image is not the previous camera image. Correspondences refer
/// to `observation.image`. Both images must use the current camera calibration.
/// Rendered depth and the estimated pose remain uncertain and correlated.
pub struct TrackingReference<'a> {
    /// Previous camera observation used for image matching.
    pub observation: &'a Frame,
    /// Surface depth rendered at the previous estimated camera pose.
    pub surface: &'a ReferenceView,
}

/// Motion assumptions for conditional camera-to-camera geometry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TrackingMotion {
    /// Estimate translation and all three rotation components.
    #[default]
    Free,
    /// Hold reference tilt fixed. Estimate translation and rotation about ENU up.
    ///
    /// The host must justify this assumption. It is not a measured attitude,
    /// and it does not set tilt uncertainty to zero. Other scenes must use `Free`.
    FixedTilt,
}

/// A relative tracking result, conditional on a previous pose and surface depth.
///
/// This is not an independent geographic measurement. It has no absolute
/// covariance. The host must retain the initial map hypothesis and the chain
/// of observation identities. Repeated evaluation adds no confidence.
pub struct TrackingProposal {
    /// Current pose in the surface frame, conditional on the reference pose.
    pub pose: CameraPose,
    /// Geometric support for this pair of camera observations.
    pub quality: EstimateQuality,
    /// Current observation identity.
    pub observation_sha256: String,
    /// Previous observation identity.
    pub reference_observation_sha256: String,
    /// Reference map revision.
    pub map: MapRevision,
    /// Local frame of the rendered surface.
    pub frame: LocalFrame,
    /// Image matcher identity.
    pub backend: String,
    /// Motion assumption used for this proposal. It does not assert a measured attitude.
    pub motion: TrackingMotion,
}

impl PoseVerifier {
    /// Check camera-to-camera matches with conditional rendered surface depth.
    ///
    /// This retains the geometric thresholds used for map matching. It does
    /// not assert that the current image matches the map imagery.
    ///
    /// # Errors
    /// Rejects reused observations, different calibrations, invalid depth,
    /// weak geometry, and poses outside the navigation prior.
    pub fn track(
        &self,
        frame: &Frame,
        reference: TrackingReference<'_>,
        prior: &PosePrior,
        matches: &[PixelMatch],
        matcher_identity: &str,
    ) -> Result<TrackingProposal, VisualError> {
        self.track_with_motion(
            frame,
            reference,
            prior,
            matches,
            matcher_identity,
            TrackingMotion::Free,
        )
    }

    /// Check relative geometry with an explicit motion assumption.
    ///
    /// Fixed tilt retains terrain depth and all support and prior checks.
    /// It returns no covariance for the constrained attitude components.
    ///
    /// # Errors
    /// Rejects invalid observations, missing depth, weak support, or prior violations.
    pub fn track_with_motion(
        &self,
        frame: &Frame,
        reference: TrackingReference<'_>,
        prior: &PosePrior,
        matches: &[PixelMatch],
        matcher_identity: &str,
        motion: TrackingMotion,
    ) -> Result<TrackingProposal, VisualError> {
        let (observation_sha256, reference_observation_sha256) =
            validate_tracking(frame, &reference, matcher_identity)?;
        let solver_motion = match motion {
            TrackingMotion::Free => crate::pose_solver::Motion::Free,
            TrackingMotion::FixedTilt => crate::pose_solver::Motion::FixedTilt,
        };
        let (pose, inliers, depth_matches) =
            self.fit_candidate(frame, reference.surface, prior, matches, solver_motion)?;
        let (quality, _) = self.assess(frame, &pose, &inliers, depth_matches)?;
        Ok(TrackingProposal {
            pose,
            quality,
            observation_sha256,
            reference_observation_sha256,
            map: reference.surface.map.clone(),
            frame: reference.surface.frame,
            backend: matcher_identity.to_owned(),
            motion,
        })
    }
    /// Check a supplied pose against camera pairs without fitting or moving that pose.
    ///
    /// Support is conditional on the reference pose and rendered surface depth.
    /// It does not establish an independent map match or geographic accuracy.
    /// Repeated checks of the same pairs do not add evidence.
    ///
    /// # Errors
    /// Rejects invalid observations, calibration changes, missing depth, insufficient
    /// support at the supplied pose, and violations of the navigation prior.
    pub fn check_tracking_pose(
        &self,
        frame: &Frame,
        reference: TrackingReference<'_>,
        prior: &PosePrior,
        pose: &CameraPose,
        matches: &[PixelMatch],
        matcher_identity: &str,
    ) -> Result<EstimateQuality, VisualError> {
        validate_tracking(frame, &reference, matcher_identity)?;
        prior.validate()?;
        pose.validate()?;
        super::check_bounds(pose, prior)?;
        let points = super::depth_correspondences(frame, reference.surface, matches);
        let count = points.len();
        let inliers: Vec<_> = points
            .into_iter()
            .filter(|point| {
                crate::pose_solver::residual(&frame.camera, pose, point)
                    <= self.config.inlier_threshold_px
            })
            .collect();
        self.assess(frame, pose, &inliers, count)
            .map(|(quality, _)| quality)
    }
}

fn validate_tracking(
    frame: &Frame,
    reference: &TrackingReference<'_>,
    matcher_identity: &str,
) -> Result<(String, String), VisualError> {
    if matcher_identity.is_empty() {
        return Err(VisualError::Invalid {
            field: "matcher identity",
        });
    }
    reference.surface.validate(reference.observation)?;
    let a = &frame.camera;
    let b = &reference.observation.camera;
    if [a.width, a.height] != [b.width, b.height]
        || [a.fx, a.fy, a.cx, a.cy] != [b.fx, b.fy, b.cx, b.cy]
    {
        return Err(VisualError::Invalid {
            field: "tracking camera calibration",
        });
    }
    let observation_sha256 = frame.evidence_sha256();
    let reference_observation_sha256 = reference.observation.evidence_sha256();
    if observation_sha256 == reference_observation_sha256 {
        return Err(VisualError::Invalid {
            field: "tracking requires different observations",
        });
    }
    reference.surface.validate(frame)?;
    Ok((observation_sha256, reference_observation_sha256))
}

#[cfg(test)]
mod tests;
