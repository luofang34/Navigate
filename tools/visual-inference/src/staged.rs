//! Separate resident XFeat extraction and descriptor matching stages.
use crate::{ExecutionConfig, InferenceError, features::Features, lighterglue, native_runtime};
use image::GrayImage;
use navigate_visual::{PixelMatch, VisualError};
use navigate_visual_workflow::{FeatureBackend, FeatureMetadata};
use ort::session::Session;
use sha2::{Digest, Sha256};
use std::path::Path;
mod detector;
mod packet;
mod scored;
mod window;
use detector::Detector;
pub use scored::ScoredPixelMatch;

/// XFeat detector input and output contract.
#[derive(Clone, Copy, Debug)]
pub enum XFeatGraph {
    /// Complete sparse detector with an 800 by 600 image input.
    Sparse,
    /// Normalized 800 by 576 grayscale input and dense output maps.
    Dense,
    /// 800 by 600 grayscale or RGB input with normalization inside the dense graph.
    DenseImage,
}

/// Immutable host features bound to an encoder and a source image.
#[derive(PartialEq)]
pub struct XFeatFeatures {
    contract: [u8; 32],
    image_digest: [u8; 32],
    image_size: [u32; 2],
    features: Features,
    normalized_pixels: Vec<f32>,
}
impl XFeatFeatures {
    /// Number of selected image features.
    pub fn len(&self) -> usize {
        self.features.pixels.len()
    }
    /// Whether this image has no selected features.
    pub fn is_empty(&self) -> bool {
        self.features.pixels.is_empty()
    }
    /// Feature coordinates in the source image.
    pub fn pixels(&self) -> &[[f64; 2]] {
        &self.features.pixels
    }
    /// Normalized descriptors in feature-major order, with 64 values per feature.
    pub fn descriptors(&self) -> &[f32] {
        &self.features.descriptors
    }
    /// Digest of the encoder, decoder, preprocessing, and feature limit.
    pub fn contract(&self) -> [u8; 32] {
        self.contract
    }
    /// Bytes retained by host allocations, excluding allocator overhead.
    pub fn resident_bytes(&self) -> usize {
        size_of::<Self>()
            + self.features.pixels.capacity() * size_of::<[f64; 2]>()
            + (self.features.model_pixels.capacity()
                + self.features.scores.capacity()
                + self.features.descriptors.capacity()
                + self.normalized_pixels.capacity())
                * size_of::<f32>()
    }
    fn new(
        contract: [u8; 32],
        image_digest: [u8; 32],
        image_size: [u32; 2],
        features: Features,
    ) -> Self {
        let normalized_pixels = features
            .model_pixels
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|p| [(p[0] - 400.0) / 400.0, (p[1] - 300.0) / 400.0])
            .collect();
        Self {
            contract,
            image_digest,
            image_size,
            features,
            normalized_pixels,
        }
    }
}

/// XFeat and optional LighterGlue with independent execution configurations.
pub struct XFeatStages {
    detector: Detector,
    matcher: Option<Session>,
    graph: XFeatGraph,
    limit: usize,
    contract: [u8; 32],
    identity: String,
}
impl XFeatStages {
    /// Load both stages once. No provider is chosen implicitly.
    ///
    /// # Errors
    /// Rejects invalid feature limits, model reads, and runtime errors.
    pub fn load_blocking(
        detector: &Path,
        graph: XFeatGraph,
        matcher: Option<&Path>,
        extraction: &ExecutionConfig,
        matching: &ExecutionConfig,
        limit: usize,
    ) -> Result<Self, InferenceError> {
        if !(32..=4096).contains(&limit) {
            return Err(InferenceError::Invalid(
                "feature limit must be 32 through 4096".into(),
            ));
        }
        let detector_hash = digest_file_blocking(detector)?;
        let contract = Sha256::digest(format!("xfeat-features-v1/{graph:?}/{detector_hash}/{limit}/pixel-centre-bilinear/normalized-f32-64")).into();
        let matcher_hash = matcher
            .map(digest_file_blocking)
            .transpose()?
            .unwrap_or_else(|| "mutual".into());
        let identity = format!(
            "xfeat-stages-v1/{graph:?}/{detector_hash}/{matcher_hash}/kp{limit}/extract-{:?}/match-{:?}",
            extraction.provider, matching.provider
        );
        Ok(Self {
            detector: Detector::Ort(native_runtime::session_blocking(detector, extraction)?),
            matcher: matcher
                .map(|p| native_runtime::session_blocking(p, matching))
                .transpose()?,
            graph,
            limit,
            contract,
            identity,
        })
    }
    /// Cumulative adapter transfer counters. None means this backend does not report them.
    pub fn tensor_statistics(&self) -> Option<crate::TensorIoStatistics> {
        self.detector.statistics()
    }
    /// Feature compatibility digest used by offline packages.
    pub fn contract(&self) -> [u8; 32] {
        self.contract
    }
    fn error(&self, source: InferenceError) -> VisualError {
        VisualError::Backend {
            backend: self.identity.clone(),
            source: Box::new(source),
        }
    }
    fn check_contract(&self, features: &XFeatFeatures) -> Result<(), InferenceError> {
        if features.contract != self.contract || features.len() > self.limit {
            return Err(InferenceError::Invalid(
                "features do not match the encoder contract or limit".into(),
            ));
        }
        Ok(())
    }
}
impl FeatureBackend for XFeatStages {
    type Features = XFeatFeatures;
    fn identity(&self) -> &str {
        &self.identity
    }
    fn extract_features_blocking(
        &mut self,
        image: &GrayImage,
    ) -> Result<XFeatFeatures, VisualError> {
        let min = image.as_raw().iter().copied().min().unwrap_or(0);
        let max = image.as_raw().iter().copied().max().unwrap_or(0);
        let features = if max.saturating_sub(min) < 2 {
            Features::empty(64)
        } else {
            self.detector
                .extract(self.graph, image, self.limit)
                .map_err(|e| self.error(e))?
        };
        Ok(XFeatFeatures::new(
            self.contract,
            image_digest(image),
            [image.width(), image.height()],
            features,
        ))
    }
    fn validate_features(
        &self,
        features: &XFeatFeatures,
        image: &GrayImage,
    ) -> Result<FeatureMetadata, VisualError> {
        self.check_contract(features).map_err(|e| self.error(e))?;
        if features.image_size != [image.width(), image.height()]
            || features.image_digest != image_digest(image)
        {
            return Err(self.error(InferenceError::Invalid(
                "features do not match the source image pixels".into(),
            )));
        }
        Ok(FeatureMetadata {
            resident_bytes: features.resident_bytes(),
            count: features.len(),
        })
    }
    fn equivalent_features(&self, first: &XFeatFeatures, second: &XFeatFeatures) -> bool {
        first == second
    }
    fn match_features_blocking(
        &mut self,
        reference: &XFeatFeatures,
        query: &XFeatFeatures,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        self.check_contract(reference).map_err(|e| self.error(e))?;
        self.check_contract(query).map_err(|e| self.error(e))?;
        match &mut self.matcher {
            Some(session) => lighterglue::match_prepared(
                session,
                &reference.features,
                &query.features,
                &reference.normalized_pixels,
                &query.normalized_pixels,
            )
            .map_err(|e| self.error(e)),
            None => Ok(crate::features::mutual(
                &reference.features,
                &query.features,
            )),
        }
    }
}
fn image_digest(image: &GrayImage) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(image.width().to_le_bytes());
    hash.update(image.height().to_le_bytes());
    hash.update(image.as_raw());
    hash.finalize().into()
}
fn digest_file_blocking(path: &Path) -> Result<String, InferenceError> {
    let bytes = std::fs::read(path).map_err(|source| InferenceError::Io {
        path: path.into(),
        source,
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
#[cfg(test)]
mod tests;
