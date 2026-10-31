//! Staged XFeat feature extraction and LighterGlue matching behind Navigate ports.
pub use crate::lighterglue::{FastMatchConfig, FastMatchStats};
use crate::{LighterGlue, MatchError, SparseFeatures, XFeat, traits::EngineKind};
use image::GrayImage;
use navigate_visual::{AsyncImageMatcher, PixelMatch, VisualError};
#[cfg(not(target_arch = "wasm32"))]
use navigate_visual::{FeatureBackend, FeatureMetadata};
use sha2::{Digest, Sha256};

/// Immutable sparse feature handle bound to one encoder contract and source image.
#[derive(Clone, Debug, PartialEq)]
pub struct BurnXFeatFeatures {
    contract: [u8; 32],
    image_digest: [u8; 32],
    image_size: [u32; 2],
    features: SparseFeatures,
}

impl BurnXFeatFeatures {
    /// Return the number of keypoints in this handle.
    pub fn len(&self) -> usize {
        self.features.len()
    }

    /// Return `true` when this handle contains no keypoints.
    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }

    /// Borrow the underlying [`SparseFeatures`].
    pub fn sparse(&self) -> &SparseFeatures {
        &self.features
    }

    /// Return the encoder contract digest.
    pub fn contract(&self) -> [u8; 32] {
        self.contract
    }

    /// Return approximate host memory retained by this handle in bytes.
    pub fn resident_bytes(&self) -> usize {
        size_of::<Self>()
            + self.features.pixels.capacity() * size_of::<[f64; 2]>()
            + self.features.model_pixels.capacity() * size_of::<[f32; 2]>()
            + (self.features.scores.capacity() + self.features.descriptors.capacity())
                * size_of::<f32>()
    }
}

/// Pure-Rust XFeat and optional LighterGlue matcher on burn.
#[derive(Clone)]
pub struct XFeatLighterGlueMatcher {
    detector: XFeat,
    matcher: Option<LighterGlue>,
    limit: usize,
    engine: EngineKind,
    contract: [u8; 32],
    identity: String,
}

impl XFeatLighterGlueMatcher {
    /// Construct a staged matcher with explicit runtime track and weight digests.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError::Output`] when `limit` is outside `32..=4096`.
    pub fn new(
        detector: XFeat,
        detector_sha256: &str,
        matcher: Option<(LighterGlue, &str)>,
        limit: usize,
        engine: EngineKind,
    ) -> Result<Self, MatchError> {
        if !(32..=4096).contains(&limit) {
            return Err(MatchError::Output("feature limit must be 32 through 4096"));
        }
        let contract: [u8; 32] = Sha256::digest(format!(
            "xfeat-burn-v1/{}/{detector_sha256}/{limit}",
            engine.as_str()
        ))
        .into();
        let (matcher_model, matcher_sha) = match matcher {
            Some((m, sha)) => (Some(m), sha),
            None => (None, "mutual"),
        };
        let identity = format!(
            "xfeat-lighterglue-burn/{}/{detector_sha256}/{matcher_sha}/kp{limit}",
            engine.as_str()
        );
        Ok(Self {
            detector,
            matcher: matcher_model,
            limit,
            engine,
            contract,
            identity,
        })
    }

    /// Return the active [`EngineKind`].
    pub fn engine_kind(&self) -> EngineKind {
        self.engine
    }

    /// Return the stable matcher identity string.
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Extract reusable [`BurnXFeatFeatures`] from `image`.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when extraction fails.
    pub async fn extract_features(
        &self,
        image: &GrayImage,
    ) -> Result<BurnXFeatFeatures, MatchError> {
        self.extract_features_masked(image, None).await
    }

    /// Extract reusable [`BurnXFeatFeatures`] inside `valid_mask` from `image`.
    ///
    /// When `valid_mask` is `Some`, this method hashes both `image` and `valid_mask` into the handle digest.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `valid_mask` dimensions do not match `image` or extraction fails.
    pub async fn extract_features_masked(
        &self,
        image: &GrayImage,
        valid_mask: Option<&GrayImage>,
    ) -> Result<BurnXFeatFeatures, MatchError> {
        let features = self
            .detector
            .extract_masked(image, valid_mask, self.limit)
            .await?;
        let image_digest = match valid_mask {
            Some(mask) => hash_masked_image(image, mask),
            None => hash_image(image),
        };
        Ok(BurnXFeatFeatures {
            contract: self.contract,
            image_digest,
            image_size: [image.width(), image.height()],
            features,
        })
    }

    /// Match two extracted [`BurnXFeatFeatures`] handles.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when feature contracts differ or matching fails.
    pub async fn match_extracted(
        &self,
        reference: &BurnXFeatFeatures,
        query: &BurnXFeatFeatures,
    ) -> Result<Vec<PixelMatch>, MatchError> {
        self.check_handle(reference)?;
        self.check_handle(query)?;
        match &self.matcher {
            Some(lg) => {
                lg.match_features(&reference.features, &query.features)
                    .await
            }
            None => Ok(reference.features.mutual_matches(&query.features)),
        }
    }

    /// Match two [`BurnXFeatFeatures`] handles with adaptive early exit and token pruning.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `config` is invalid, feature contracts differ, or matching fails.
    pub async fn match_extracted_fast(
        &self,
        reference: &BurnXFeatFeatures,
        query: &BurnXFeatFeatures,
        config: &FastMatchConfig,
    ) -> Result<(Vec<PixelMatch>, FastMatchStats), MatchError> {
        config.validate()?;
        self.check_handle(reference)?;
        self.check_handle(query)?;
        match &self.matcher {
            Some(lg) => {
                lg.match_features_fast(&reference.features, &query.features, config)
                    .await
            }
            None => {
                let ref_top = reference.features.truncate_top(config.max_keypoints);
                let query_top = query.features.truncate_top(config.max_keypoints);
                let matches = ref_top.mutual_matches(&query_top);
                Ok((
                    matches,
                    FastMatchStats {
                        layers_executed: 0,
                        initial_ref_tokens: ref_top.len(),
                        initial_query_tokens: query_top.len(),
                        pruned_ref_tokens: 0,
                        pruned_query_tokens: 0,
                        early_exited: false,
                    },
                ))
            }
        }
    }

    /// Match `reference` and `query` images asynchronously.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when extraction or matching fails.
    pub async fn match_pair(
        &self,
        reference: &GrayImage,
        query: &GrayImage,
    ) -> Result<Vec<PixelMatch>, MatchError> {
        let a = self.extract_features(reference).await?;
        let b = self.extract_features(query).await?;
        self.match_extracted(&a, &b).await
    }

    fn check_handle(&self, handle: &BurnXFeatFeatures) -> Result<(), MatchError> {
        if handle.contract != self.contract || handle.len() > self.limit {
            return Err(MatchError::Output(
                "feature handle does not match encoder contract or limit",
            ));
        }
        Ok(())
    }

    fn backend_error(&self, source: MatchError) -> VisualError {
        VisualError::Backend {
            backend: self.identity.clone(),
            source: Box::new(source),
        }
    }
}

impl AsyncImageMatcher for XFeatLighterGlueMatcher {
    fn identity(&self) -> &str {
        &self.identity
    }

    async fn match_images(
        &mut self,
        reference: &GrayImage,
        query: &GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        self.match_pair(reference, query)
            .await
            .map_err(|source| self.backend_error(source))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl navigate_visual::ImageMatcher for XFeatLighterGlueMatcher {
    fn identity(&self) -> &str {
        &self.identity
    }

    fn match_images_blocking(
        &mut self,
        reference: &GrayImage,
        query: &GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        pollster::block_on(self.match_pair(reference, query))
            .map_err(|source| self.backend_error(source))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl FeatureBackend for XFeatLighterGlueMatcher {
    type Features = BurnXFeatFeatures;

    fn identity(&self) -> &str {
        &self.identity
    }

    fn extract_features_blocking(
        &mut self,
        image: &GrayImage,
    ) -> Result<BurnXFeatFeatures, VisualError> {
        pollster::block_on(self.extract_features(image))
            .map_err(|source| self.backend_error(source))
    }

    fn validate_features(
        &self,
        features: &BurnXFeatFeatures,
        image: &GrayImage,
    ) -> Result<FeatureMetadata, VisualError> {
        self.check_handle(features)
            .map_err(|source| self.backend_error(source))?;
        if features.image_size != [image.width(), image.height()]
            || features.image_digest != hash_image(image)
        {
            return Err(self.backend_error(MatchError::Output(
                "feature handle does not match image pixels",
            )));
        }
        Ok(FeatureMetadata {
            resident_bytes: features.resident_bytes(),
            count: features.len(),
        })
    }

    fn equivalent_features(&self, first: &BurnXFeatFeatures, second: &BurnXFeatFeatures) -> bool {
        first == second
    }

    fn match_features_blocking(
        &mut self,
        reference: &BurnXFeatFeatures,
        query: &BurnXFeatFeatures,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        pollster::block_on(self.match_extracted(reference, query))
            .map_err(|source| self.backend_error(source))
    }
}

fn hash_image(image: &GrayImage) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(image.width().to_le_bytes());
    hasher.update(image.height().to_le_bytes());
    hasher.update(image.as_raw());
    hasher.finalize().into()
}

fn hash_masked_image(image: &GrayImage, mask: &GrayImage) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(image.width().to_le_bytes());
    hasher.update(image.height().to_le_bytes());
    hasher.update(image.as_raw());
    hasher.update(b"valid_mask");
    hasher.update(mask.width().to_le_bytes());
    hasher.update(mask.height().to_le_bytes());
    hasher.update(mask.as_raw());
    hasher.finalize().into()
}
