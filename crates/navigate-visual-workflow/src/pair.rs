//! Image-pair models use the same bounds and geometric acceptance policy.
use crate::{CandidateReport, FrameReport, ResourceLimits, StageTimes, WorkflowError};
use navigate_visual::{
    AsyncImageMatcher, CameraModel, Frame, ImageMatcher, LocalizerConfig, PixelMatch, PosePrior,
    PoseVerifier, ReferenceView, VisualError,
};
use std::time::Duration;
use web_time::Instant;

/// Bounded candidate evaluation for models with cross-image computation.
pub struct PairLocalizer<M> {
    matcher: M,
    verifier: PoseVerifier,
    limits: ResourceLimits,
}
impl<M> PairLocalizer<M> {
    /// Construct an image-pair workflow without an offline feature requirement.
    ///
    /// # Errors
    /// Rejects invalid geometry and resource settings.
    pub fn new(
        matcher: M,
        geometry: LocalizerConfig,
        limits: ResourceLimits,
    ) -> Result<Self, WorkflowError> {
        Ok(Self {
            matcher,
            verifier: PoseVerifier::new(geometry)?,
            limits: limits.validate()?,
        })
    }

    /// Verify one candidate's correspondences with the shared pose policy.
    #[allow(clippy::too_many_arguments)]
    fn candidate(
        &self,
        index: usize,
        (reference, camera): (&ReferenceView, &CameraModel),
        (frame, prior): (&Frame, &PosePrior),
        matches: &[PixelMatch],
        identity: &str,
        matching: Duration,
    ) -> CandidateReport {
        let start = Instant::now();
        let decision = self
            .verifier
            .verify_calibrated_reference(frame, reference, camera, prior, matches, identity);
        CandidateReport {
            matching_reused_from: None,
            reference_index: index,
            matches: matches.len(),
            decision,
            times: StageTimes {
                matching,
                geometry: start.elapsed(),
            },
        }
    }

    fn validate(
        &self,
        frame: &Frame,
        prior: &PosePrior,
        references: &[(&ReferenceView, &CameraModel)],
    ) -> Result<(), WorkflowError> {
        frame.camera.validate()?;
        prior.validate()?;
        if frame.image.dimensions() != (frame.camera.width, frame.camera.height) {
            return Err(VisualError::Invalid {
                field: "query image calibration",
            }
            .into());
        }
        super::localizer::check(
            "image pixels",
            frame.image.as_raw().len(),
            self.limits.image_pixels,
        )?;
        super::localizer::check(
            "reference candidates",
            references.len(),
            self.limits.candidates,
        )?;
        let mut bytes = 0_usize;
        for (view, camera) in references {
            view.validate_camera(camera)?;
            super::localizer::check(
                "image pixels",
                view.image.as_raw().len(),
                self.limits.image_pixels,
            )?;
            bytes = bytes
                .saturating_add(view.image.as_raw().len())
                .saturating_add(view.depth_m.len().saturating_mul(4));
        }
        super::localizer::check("reference bytes", bytes, self.limits.reference_bytes)?;
        Ok(())
    }
}

impl<M: AsyncImageMatcher> PairLocalizer<M> {
    /// Evaluate references like [`PairLocalizer::estimate_calibrated_blocking`] with a
    /// matcher that cannot block, such as a GPU model in a browser.
    ///
    /// # Errors
    /// Rejects invalid input, exceeded limits, and matcher failures.
    pub async fn estimate_calibrated(
        &mut self,
        frame: &Frame,
        prior: &PosePrior,
        references: &[(&ReferenceView, &CameraModel)],
    ) -> Result<FrameReport, WorkflowError> {
        let started = Instant::now();
        self.validate(frame, prior, references)?;
        let mut candidates = Vec::with_capacity(references.len());
        for (index, (reference, camera)) in references.iter().enumerate() {
            let start = Instant::now();
            let matches = self
                .matcher
                .match_images(&reference.image, &frame.image)
                .await
                .map_err(|source| WorkflowError::Backend {
                    stage: "match image pair",
                    source,
                })?;
            let identity = self.matcher.identity().to_owned();
            candidates.push(self.candidate(
                index,
                (reference, camera),
                (frame, prior),
                &matches,
                &identity,
                start.elapsed(),
            ));
        }
        Ok(FrameReport {
            extraction: Duration::ZERO,
            total: started.elapsed(),
            candidates,
        })
    }
}

impl<M: ImageMatcher> PairLocalizer<M> {
    /// Evaluate fixed references without image encoding or file I/O.
    ///
    /// # Errors
    /// Rejects invalid input, exceeded limits, and matcher failures.
    pub fn estimate_blocking(
        &mut self,
        frame: &Frame,
        prior: &PosePrior,
        references: &[ReferenceView],
    ) -> Result<FrameReport, WorkflowError> {
        let calibrated: Vec<_> = references.iter().map(|r| (r, &frame.camera)).collect();
        self.estimate_calibrated_blocking(frame, prior, &calibrated)
    }
    /// Evaluate references with separate camera calibration and no image copies.
    ///
    /// # Errors
    /// Rejects invalid input, exceeded limits, and matcher failures.
    pub fn estimate_calibrated_blocking(
        &mut self,
        frame: &Frame,
        prior: &PosePrior,
        references: &[(&ReferenceView, &CameraModel)],
    ) -> Result<FrameReport, WorkflowError> {
        let started = Instant::now();
        self.validate(frame, prior, references)?;
        let mut candidates = Vec::with_capacity(references.len());
        for (index, (reference, camera)) in references.iter().enumerate() {
            let start = Instant::now();
            let matches = self
                .matcher
                .match_images_blocking(&reference.image, &frame.image)
                .map_err(|source| WorkflowError::Backend {
                    stage: "match image pair",
                    source,
                })?;
            let identity = self.matcher.identity().to_owned();
            candidates.push(self.candidate(
                index,
                (reference, camera),
                (frame, prior),
                &matches,
                &identity,
                start.elapsed(),
            ));
        }
        Ok(FrameReport {
            extraction: Duration::ZERO,
            total: started.elapsed(),
            candidates,
        })
    }
}

#[cfg(test)]
mod tests;
