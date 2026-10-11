//! Image-pair models use the same bounds and geometric acceptance policy.
use crate::{CandidateReport, FrameReport, ResourceLimits, StageTimes, WorkflowError};
use navigate_visual::{
    Frame, ImageMatcher, LocalizerConfig, PosePrior, PoseVerifier, ReferenceView, VisualError,
};
use std::time::{Duration, Instant};

/// Bounded candidate evaluation for models with cross-image computation.
pub struct PairLocalizer<M> {
    matcher: M,
    verifier: PoseVerifier,
    limits: ResourceLimits,
}
impl<M: ImageMatcher> PairLocalizer<M> {
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
        let started = Instant::now();
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
        for view in references {
            view.validate(frame)?;
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
        let mut candidates = Vec::with_capacity(references.len());
        for (index, reference) in references.iter().enumerate() {
            let start = Instant::now();
            let matches = self
                .matcher
                .match_images_blocking(&reference.image, &frame.image)
                .map_err(|source| WorkflowError::Backend {
                    stage: "match image pair",
                    source,
                })?;
            let matching = start.elapsed();
            let start = Instant::now();
            let decision =
                self.verifier
                    .verify(frame, reference, prior, &matches, self.matcher.identity());
            candidates.push(CandidateReport {
                reference_index: index,
                matches: matches.len(),
                decision,
                times: StageTimes {
                    matching,
                    geometry: start.elapsed(),
                },
            });
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
