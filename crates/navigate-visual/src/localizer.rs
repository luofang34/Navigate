//! Frame admission and geometric pose acceptance.

use crate::{
    CameraPose, Frame, FrameStamp, ImageMatcher, LocalFrame, MapRevision, PixelMatch, PosePrior,
    ReferenceView, TrackingMotion, VisualError,
};
use nalgebra::SMatrix;

/// Acceptance thresholds for one visual observation.
#[derive(Clone, Copy, Debug)]
pub struct LocalizerConfig {
    /// Minimum count of spatially separated geometric inliers.
    pub min_inliers: usize,
    /// Minimum separation of counted support in both images, in pixels.
    /// This limits dense repetitions. It does not establish statistical independence.
    pub support_separation_px: f64,
    /// Maximum reprojection error for an inlier, in pixels.
    pub inlier_threshold_px: f64,
    /// Minimum occupied cells in a four-column, three-row image grid.
    pub min_occupied_cells: usize,
    /// Assumed pixel noise floor for the local geometry covariance.
    pub pixel_noise_floor: f64,
    /// Residual at which the local geometry covariance doubles, in pixels.
    ///
    /// The covariance takes the reprojection variance of the support as
    /// independent pixel noise. A residual above the precision of the
    /// matcher comes from correspondences whose errors are shared, such as
    /// a repeated texture, a changed appearance, or a resampled image, and
    /// those errors do not average down over the support. The variance is
    /// multiplied by `1 + (rms / this)²`, with `rms` the reprojection RMS.
    /// Infinity takes the residual as independent noise.
    pub residual_correlation_px: f64,
}

impl Default for LocalizerConfig {
    fn default() -> Self {
        Self {
            min_inliers: 20,
            support_separation_px: 12.0,
            inlier_threshold_px: 2.5,
            min_occupied_cells: 4,
            pixel_noise_floor: 1.0,
            residual_correlation_px: f64::INFINITY,
        }
    }
}

/// Measured evidence for an accepted pose. These values are not an integrity guarantee.
#[derive(Clone, Copy, Debug)]
pub struct EstimateQuality {
    /// Number of matches with usable reference depth.
    pub depth_matches: usize,
    /// Number of final geometric inliers.
    pub inliers: usize,
    /// Spatially separated inliers used for support checks and local covariance.
    /// Remaining feature, map, and calibration correlations are not quantified.
    pub spatial_support: usize,
    /// Root mean square reprojection error, in pixels.
    pub reprojection_rms_px: f64,
    /// Number of occupied cells in the query image.
    pub occupied_cells: usize,
    /// Condition number of the scaled pose normal matrix.
    pub condition_number: f64,
}

/// A map-relative camera pose from one frame.
#[derive(Clone)]
pub struct Estimate {
    /// Acquisition stamp of the query image.
    pub stamp: FrameStamp,
    /// Digest of the query pixels, calibration and capture stamp.
    /// Identical digests identify repeated processing, not independent evidence.
    pub observation_sha256: String,
    /// Selected map release.
    pub map: MapRevision,
    /// Frame of `pose` and of the position axes of `geometry_covariance`.
    pub frame: LocalFrame,
    /// Camera pose in `frame`.
    pub pose: CameraPose,
    /// Acceptance evidence.
    pub quality: EstimateQuality,
    /// Local linearized covariance from image residuals only.
    ///
    /// First three axes are position along the `frame` axes (east, north, up) in metres. Last three are local camera
    /// rotation in radians. Map, calibration, and association errors are excluded.
    /// A fusion adapter must account for these errors before admission.
    /// Unknown map error and shared-evidence correlation are not zero.
    /// Re-evaluating an observation does not create a second measurement.
    pub geometry_covariance: SMatrix<f64, 6, 6>,
    /// Matcher and model identity.
    pub backend: String,
}

/// The matches of one frame and which of them the geometric fit kept, for
/// a host that shows the match. Pixels are centres in the images that
/// were matched: the reference view and the query frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MatchEvidence {
    /// The matches that the matcher gave, in its order.
    pub matches: Vec<PixelMatch>,
    /// For each match, whether the final fit kept it as an inlier. Empty
    /// when the frame was refused before a fit: invalid input, a backend
    /// failure, or too few matches with reference depth.
    pub inliers: Vec<bool>,
}

impl MatchEvidence {
    /// The evidence of `matches`, with the query pixels `kept` of the final
    /// inliers. Each kept pixel is that of one correspondence with depth,
    /// in the order of the matches, and no two correspondences share a
    /// query pixel, so the walk below finds each one exactly.
    fn of(
        frame: &Frame,
        reference: &ReferenceView,
        matches: Vec<PixelMatch>,
        kept: &[nalgebra::Vector2<f64>],
    ) -> Self {
        if kept.is_empty() {
            return Self {
                matches,
                inliers: Vec::new(),
            };
        }
        let mut inliers = vec![false; matches.len()];
        let mut next = kept.iter().peekable();
        for (index, point) in
            crate::geometry::depth_correspondences_indexed(frame, reference, &matches)
        {
            if next.peek().is_some_and(|pixel| **pixel == point.pixel) {
                next.next();
                if let Some(flag) = inliers.get_mut(index) {
                    *flag = true;
                }
            }
        }
        Self { matches, inliers }
    }
}

/// Ordered visual map observations with no assumed statistical independence.
///
/// The caller supplies each frame's prior and reference. This component does not
/// feed its own result back as an independent measurement or fuse IMU data.
///
/// A clone continues the same capture stream: it holds the last admitted
/// stamp, so a host can copy a localizer with the state of its navigator.
#[derive(Clone, Debug)]
pub struct Localizer<M> {
    matcher: M,
    verifier: crate::PoseVerifier,
    last_stamp: Option<FrameStamp>,
}

impl<M: ImageMatcher> Localizer<M> {
    /// Matcher identity and input verification remain available to the host.
    pub fn matcher(&self) -> &M {
        &self.matcher
    }
    /// The matcher, to tune it for the next frame, as a bound of its search
    /// from the prior of that frame.
    pub fn matcher_mut(&mut self) -> &mut M {
        &mut self.matcher
    }
    /// Build a localizer with explicit acceptance thresholds.
    ///
    /// # Errors
    ///
    /// Returns [`VisualError::Invalid`] for invalid acceptance thresholds.
    pub fn new(matcher: M, config: LocalizerConfig) -> Result<Self, VisualError> {
        Ok(Self {
            matcher,
            verifier: crate::PoseVerifier::new(config)?,
            last_stamp: None,
        })
    }

    /// Refine the reference pose from image evidence and enforce the prior bounds.
    ///
    /// A retrieval candidate may differ from the navigation prior. It initializes
    /// optimization but cannot replace the prior's admission bounds.
    ///
    /// A valid input consumes its stamp before matching starts. A visual rejection
    /// therefore consumes the stamp. Use a new localizer for a new capture stream.
    /// Use [`crate::PoseVerifier`] for multiple candidates or repeated refinement
    /// within one observation. This call waits for the matcher and GPU readback.
    /// Call it on a worker that may block.
    ///
    /// # Errors
    ///
    /// Rejects invalid inputs, repeated or out-of-order stamps, backend failures,
    /// insufficient unique matches, weak geometry, and corrections outside the
    /// prior bounds. A rejection contains no substitute pose.
    pub fn estimate_blocking(
        &mut self,
        frame: &Frame,
        reference: &ReferenceView,
        prior: &PosePrior,
    ) -> Result<Estimate, VisualError> {
        self.estimate_and_match(frame, reference, prior).0
    }

    /// [`Self::estimate_blocking`], and the matches that it made with the
    /// inliers of its fit. The estimate and the stream state are those of
    /// `estimate_blocking`; the evidence costs one more pass over the
    /// matches.
    pub fn estimate_with_evidence_blocking(
        &mut self,
        frame: &Frame,
        reference: &ReferenceView,
        prior: &PosePrior,
    ) -> (Result<Estimate, VisualError>, MatchEvidence) {
        let (estimate, matches, kept) = self.estimate_and_match(frame, reference, prior);
        let evidence = MatchEvidence::of(frame, reference, matches, &kept);
        (estimate, evidence)
    }

    /// The estimate, the matches, and the query pixels of the final
    /// inliers of the fit.
    fn estimate_and_match(
        &mut self,
        frame: &Frame,
        reference: &ReferenceView,
        prior: &PosePrior,
    ) -> (
        Result<Estimate, VisualError>,
        Vec<PixelMatch>,
        Vec<nalgebra::Vector2<f64>>,
    ) {
        let admitted = reference
            .validate(frame)
            .and_then(|()| prior.validate())
            .and_then(|()| self.admit_stamp(frame.stamp));
        if let Err(error) = admitted {
            return (Err(error), Vec::new(), Vec::new());
        }
        let matches = match self
            .matcher
            .match_images_blocking(&reference.image, &frame.image)
        {
            Ok(matches) => matches,
            Err(error) => return (Err(error), Vec::new(), Vec::new()),
        };
        let (evaluation, kept) = self.verifier.evaluate_keeping_inliers(
            frame,
            reference,
            prior,
            &matches,
            self.matcher.identity(),
            TrackingMotion::Free,
        );
        (evaluation.acceptance, matches, kept)
    }

    fn admit_stamp(&mut self, stamp: FrameStamp) -> Result<(), VisualError> {
        if let Some(previous) = self.last_stamp {
            if stamp.capture_time_ns <= previous.capture_time_ns {
                return Err(VisualError::FrameOrder {
                    previous_ns: previous.capture_time_ns,
                    received_ns: stamp.capture_time_ns,
                });
            }
            let advance = stamp.sequence.wrapping_sub(previous.sequence);
            if advance == 0 || advance >= (1_u64 << 63) {
                return Err(VisualError::Invalid {
                    field: "frame sequence",
                });
            }
        }
        self.last_stamp = Some(stamp);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
