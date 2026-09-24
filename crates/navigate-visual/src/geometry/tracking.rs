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
        let (pose, inliers, depth_matches) =
            self.fit_candidate(frame, reference.surface, prior, matches, matcher_identity)?;
        let (quality, _) = self.assess(frame, &pose, &inliers, depth_matches)?;
        Ok(TrackingProposal {
            pose,
            quality,
            observation_sha256,
            reference_observation_sha256,
            map: reference.surface.map.clone(),
            frame: reference.surface.frame,
            backend: matcher_identity.to_owned(),
        })
    }
}

#[cfg(test)]
mod tests;
