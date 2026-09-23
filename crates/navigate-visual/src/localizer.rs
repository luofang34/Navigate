//! Frame admission and geometric pose acceptance.

use crate::{
    CameraPose, Frame, FrameStamp, ImageMatcher, LocalFrame, MapRevision, PosePrior, ReferenceView,
    VisualError,
};
use nalgebra::SMatrix;

/// Acceptance thresholds for one visual observation.
#[derive(Clone, Copy, Debug)]
pub struct LocalizerConfig {
    /// Minimum count of geometric inliers.
    pub min_inliers: usize,
    /// Maximum reprojection error for an inlier, in pixels.
    pub inlier_threshold_px: f64,
    /// Minimum occupied cells in a four-column, three-row image grid.
    pub min_occupied_cells: usize,
    /// Assumed pixel noise floor for the local geometry covariance.
    pub pixel_noise_floor: f64,
}

impl Default for LocalizerConfig {
    fn default() -> Self {
        Self {
            min_inliers: 20,
            inlier_threshold_px: 2.5,
            min_occupied_cells: 4,
            pixel_noise_floor: 1.0,
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

/// Ordered visual map observations with no assumed statistical independence.
///
/// The caller supplies each frame's prior and reference. This component does not
/// feed its own result back as an independent measurement or fuse IMU data.
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
        reference.validate(frame)?;
        prior.validate()?;
        self.admit_stamp(frame.stamp)?;
        let matches = self
            .matcher
            .match_images_blocking(&reference.image, &frame.image)?;
        self.verifier
            .verify(frame, reference, prior, &matches, self.matcher.identity())
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
