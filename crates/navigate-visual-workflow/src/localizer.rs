//! Query extraction occurs once for each bounded candidate batch.

use crate::{FeatureBackend, WorkflowError};
use navigate_visual::{
    CameraModel, Estimate, Frame, LocalizerConfig, PosePrior, PoseVerifier, ReferenceView,
    VisualError,
};
use std::time::Duration;
use web_time::Instant;

/// Limits for host-owned images, feature handles, and candidate work.
/// Model weights and runtime workspaces need a separate backend budget.
#[derive(Clone, Copy, Debug)]
pub struct ResourceLimits {
    /// Maximum pixels in one query or reference image.
    pub image_pixels: usize,
    /// Maximum features in one handle.
    pub features_per_image: usize,
    /// Maximum reference candidates per query.
    pub candidates: usize,
    /// Maximum bytes in all reference images, depths, and feature handles.
    pub reference_bytes: usize,
    /// Maximum bytes in the query feature handle.
    pub query_feature_bytes: usize,
}
impl ResourceLimits {
    pub(crate) fn validate(self) -> Result<Self, WorkflowError> {
        for (field, value) in [
            ("image_pixels", self.image_pixels),
            ("features_per_image", self.features_per_image),
            ("candidates", self.candidates),
            ("reference_bytes", self.reference_bytes),
            ("query_feature_bytes", self.query_feature_bytes),
        ] {
            if value == 0 {
                return Err(WorkflowError::InvalidLimit { field });
            }
        }
        Ok(self)
    }
}

/// An immutable reference with validated resident features.
pub struct PreparedReference<F> {
    view: ReferenceView,
    features: F,
    camera: Option<CameraModel>,
    resident_bytes: usize,
}
impl<F> PreparedReference<F> {
    /// Bind the camera used to produce the reference pixels and depth.
    /// This does not change or copy image and feature storage.
    ///
    /// # Errors
    /// Rejects invalid calibration or reference geometry.
    pub fn with_camera(mut self, camera: CameraModel) -> Result<Self, WorkflowError> {
        self.view.validate_camera(&camera)?;
        self.camera = Some(camera);
        Ok(self)
    }
    /// Reference geometry and image pixels.
    pub fn view(&self) -> &ReferenceView {
        &self.view
    }
    /// Bytes owned by this reference, excluding backend model state.
    pub fn resident_bytes(&self) -> usize {
        self.resident_bytes
    }
}

/// Wall-clock measurements for one candidate.
#[derive(Clone, Copy, Debug)]
pub struct StageTimes {
    /// Feature matching, including any device synchronization.
    pub matching: Duration,
    /// Pose fitting and acceptance checks.
    pub geometry: Duration,
}

/// One candidate's geometry decision. Rejection is a valid measured result.
pub struct CandidateReport {
    /// Index in the supplied reference list.
    pub reference_index: usize,
    /// Earlier candidate whose matches were reused. Geometry is always evaluated separately.
    pub matching_reused_from: Option<usize>,
    /// Pixel correspondences before geometric filtering.
    pub matches: usize,
    /// Shared geometry acceptance decision.
    pub decision: Result<Estimate, VisualError>,
    /// Stage wall-clock measurements.
    pub times: StageTimes,
}

/// Results from one query against a bounded candidate list.
pub struct FrameReport {
    /// Query extraction time. It is paid once for this report.
    pub extraction: Duration,
    /// Total processing time, including input validation.
    pub total: Duration,
    /// Candidate results in input order.
    pub candidates: Vec<CandidateReport>,
}

/// Executes feature stages and retains the common geometric acceptance policy.
pub struct FeatureLocalizer<B> {
    backend: B,
    verifier: PoseVerifier,
    limits: ResourceLimits,
}
impl<B: FeatureBackend> FeatureLocalizer<B> {
    /// Construct a workflow with explicit geometry and resource limits.
    ///
    /// # Errors
    /// Rejects invalid geometry or resource limits.
    pub fn new(
        backend: B,
        geometry: LocalizerConfig,
        limits: ResourceLimits,
    ) -> Result<Self, WorkflowError> {
        Ok(Self {
            backend,
            verifier: PoseVerifier::new(geometry)?,
            limits: limits.validate()?,
        })
    }

    /// Access the backend to import or export model-specific feature packages.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Extract and retain one reference without an offline package.
    ///
    /// # Errors
    /// Rejects invalid images, features, or resource requirements.
    pub fn prepare_reference_blocking(
        &mut self,
        view: ReferenceView,
    ) -> Result<PreparedReference<B::Features>, WorkflowError> {
        self.check_image(&view.image)?;
        let features = self
            .backend
            .extract_features_blocking(&view.image)
            .map_err(|source| WorkflowError::Backend {
                stage: "extract reference",
                source,
            })?;
        self.import_reference(view, features)
    }

    /// Bind imported features to the exact reference image and encoder contract.
    ///
    /// # Errors
    /// Rejects incompatible packages and resource requirements.
    pub fn import_reference(
        &self,
        view: ReferenceView,
        features: B::Features,
    ) -> Result<PreparedReference<B::Features>, WorkflowError> {
        self.check_image(&view.image)?;
        let metadata = self
            .backend
            .validate_features(&features, &view.image)
            .map_err(|source| WorkflowError::Backend {
                stage: "validate reference features",
                source,
            })?;
        check(
            "reference features",
            metadata.count,
            self.limits.features_per_image,
        )?;
        let bytes = metadata
            .resident_bytes
            .saturating_add(view.image.as_raw().len())
            .saturating_add(view.depth_m.len().saturating_mul(size_of::<f32>()));
        check("reference bytes", bytes, self.limits.reference_bytes)?;
        Ok(PreparedReference {
            view,
            features,
            camera: None,
            resident_bytes: bytes,
        })
    }

    /// Extract the query once and evaluate each supplied reference.
    ///
    /// No accepted estimate is reused as independent evidence. The caller owns
    /// retrieval and the prior. A backend error stops this call.
    ///
    /// # Errors
    /// Rejects invalid input, exhausted resource limits, or backend failures.
    pub fn estimate_blocking(
        &mut self,
        frame: &Frame,
        prior: &PosePrior,
        references: &[PreparedReference<B::Features>],
    ) -> Result<FrameReport, WorkflowError> {
        check(
            "reference candidates",
            references.len(),
            self.limits.candidates,
        )?;
        let selected: Vec<_> = references.iter().collect();
        self.estimate_selected_blocking(frame, prior, &selected)
    }

    /// Estimate against borrowed resident references without copying feature storage.
    ///
    /// # Errors
    /// Applies the same input, resource, and backend checks as [`Self::estimate_blocking`].
    pub fn estimate_selected_blocking(
        &mut self,
        frame: &Frame,
        prior: &PosePrior,
        references: &[&PreparedReference<B::Features>],
    ) -> Result<FrameReport, WorkflowError> {
        let started = Instant::now();
        self.check_image(&frame.image)?;
        frame.camera.validate()?;
        if frame.image.dimensions() != (frame.camera.width, frame.camera.height) {
            return Err(VisualError::Invalid {
                field: "query image calibration",
            }
            .into());
        }
        prior.validate()?;
        check(
            "reference candidates",
            references.len(),
            self.limits.candidates,
        )?;
        let bytes = references
            .iter()
            .fold(0_usize, |total, r| total.saturating_add(r.resident_bytes));
        check("reference bytes", bytes, self.limits.reference_bytes)?;
        for reference in references {
            reference
                .view
                .validate_camera(reference.camera.as_ref().unwrap_or(&frame.camera))?;
        }
        if references.is_empty() {
            return Ok(FrameReport {
                extraction: Duration::ZERO,
                total: started.elapsed(),
                candidates: vec![],
            });
        }
        let extraction_started = Instant::now();
        let query = self
            .backend
            .extract_features_blocking(&frame.image)
            .map_err(|source| WorkflowError::Backend {
                stage: "extract query",
                source,
            })?;
        let metadata = self
            .backend
            .validate_features(&query, &frame.image)
            .map_err(|source| WorkflowError::Backend {
                stage: "validate query features",
                source,
            })?;
        check(
            "query features",
            metadata.count,
            self.limits.features_per_image,
        )?;
        check(
            "query feature bytes",
            metadata.resident_bytes,
            self.limits.query_feature_bytes,
        )?;
        let extraction = extraction_started.elapsed();
        let candidates = self.evaluate_candidates(frame, prior, references, &query)?;
        Ok(FrameReport {
            extraction,
            total: started.elapsed(),
            candidates,
        })
    }

    fn evaluate_candidates(
        &mut self,
        frame: &Frame,
        prior: &PosePrior,
        references: &[&PreparedReference<B::Features>],
        query: &B::Features,
    ) -> Result<Vec<CandidateReport>, WorkflowError> {
        let mut reports: Vec<Option<CandidateReport>> =
            (0..references.len()).map(|_| None).collect();
        for (index, reference) in references.iter().enumerate() {
            if reports[index].is_some() {
                continue;
            }
            let start = Instant::now();
            let matches = self
                .backend
                .match_features_blocking(&reference.features, query)
                .map_err(|source| WorkflowError::Backend {
                    stage: "match features",
                    source,
                })?;
            let matching = start.elapsed();
            for (other, candidate) in references.iter().enumerate().skip(index) {
                if reports[other].is_some()
                    || (other != index
                        && !self
                            .backend
                            .equivalent_features(&reference.features, &candidate.features))
                {
                    continue;
                }
                let start = Instant::now();
                let decision = self.verifier.verify_calibrated_reference(
                    frame,
                    &candidate.view,
                    candidate.camera.as_ref().unwrap_or(&frame.camera),
                    prior,
                    &matches,
                    self.backend.identity(),
                );
                reports[other] = Some(CandidateReport {
                    reference_index: other,
                    matching_reused_from: (other != index).then_some(index),
                    matches: matches.len(),
                    decision,
                    times: StageTimes {
                        matching: if other == index {
                            matching
                        } else {
                            Duration::ZERO
                        },
                        geometry: start.elapsed(),
                    },
                });
            }
        }
        Ok(reports.into_iter().flatten().collect())
    }

    fn check_image(&self, image: &image::GrayImage) -> Result<(), WorkflowError> {
        check(
            "image pixels",
            image.as_raw().len(),
            self.limits.image_pixels,
        )
    }
}
pub(crate) fn check(
    resource: &'static str,
    requested: usize,
    limit: usize,
) -> Result<(), WorkflowError> {
    if requested > limit {
        Err(WorkflowError::ResourceLimit {
            resource,
            requested,
            limit,
        })
    } else {
        Ok(())
    }
}
#[cfg(test)]
mod tests;
