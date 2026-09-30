//! SuperPoint and LightGlue exports behind the Navigate pixel matcher port.
mod profile;
use crate::feature_cache::FeatureCache;
use crate::{ExecutionConfig, InferenceError, features::Features, native_runtime, superpoint};
use image::GrayImage;
use navigate_visual::{ImageMatcher, PixelMatch, VisualError};
use ort::{session::Session, value::Tensor};
use profile::KeypointProfile;
use sha2::{Digest, Sha256};
use std::path::Path;

/// Resident SuperPoint and full-depth LightGlue ONNX sessions.
///
/// The matcher takes pixel keypoints and row-major 256-value descriptors.
/// Model files must embed their weights. No weights are supplied by this crate.
/// Detector thresholds and assignment scores stay in this adapter.
pub struct LightGlueMatcher {
    detector: Session,
    matcher: Session,
    profile: Option<KeypointProfile>,
    size: [usize; 2],
    keypoints: usize,
    identity: String,
    features: FeatureCache,
}
impl LightGlueMatcher {
    /// Load explicit exports and their embedded image normalization size.
    ///
    /// # Errors
    /// Rejects invalid sizes, feature limits, files, and runtime setup.
    pub fn load_blocking(
        detector: &Path,
        matcher: &Path,
        image_size: [usize; 2],
        keypoints: usize,
        execution: &ExecutionConfig,
    ) -> Result<Self, InferenceError> {
        Self::load_inner_blocking(detector, matcher, image_size, keypoints, execution, false)
    }
    /// Load a fixed full-feature session and a dynamic session from the same model.
    ///
    /// The fixed session runs only when both images yield the exact feature limit.
    /// Other counts use the dynamic session. No features are padded or removed.
    /// Enable this only for a model/device combination that has passed quality
    /// and performance checks. The second resident session increases memory use.
    /// Device arithmetic can change assignments. Compare fitted poses as well as
    /// match counts. The second session does not add independent evidence.
    ///
    /// # Errors
    /// Rejects incompatible input dimensions and runtime or model failures.
    pub fn load_profiled_blocking(
        detector: &Path,
        matcher: &Path,
        image_size: [usize; 2],
        keypoints: usize,
        execution: &ExecutionConfig,
    ) -> Result<Self, InferenceError> {
        Self::load_inner_blocking(detector, matcher, image_size, keypoints, execution, true)
    }
    fn load_inner_blocking(
        detector: &Path,
        matcher: &Path,
        image_size: [usize; 2],
        keypoints: usize,
        execution: &ExecutionConfig,
        profiled: bool,
    ) -> Result<Self, InferenceError> {
        if !(32..=4096).contains(&keypoints)
            || image_size
                .iter()
                .any(|d| *d < 64 || *d > 1920 || d % 8 != 0)
        {
            return Err(InferenceError::Invalid(
                "invalid LightGlue export size or feature limit".into(),
            ));
        }
        let mut identity = format!(
            "superpoint-lightglue-full-v1/{}/{}/{}x{}/kp{keypoints}/ort-requested-{:?}",
            digest_blocking(detector)?,
            digest_blocking(matcher)?,
            image_size[0],
            image_size[1],
            execution.provider
        );
        let dynamic = native_runtime::session_blocking(matcher, execution)?;
        let profile = if profiled {
            identity.push_str("/exact-full-features-with-dynamic-fallback");
            Some(KeypointProfile::load_blocking(
                matcher, &dynamic, keypoints, execution,
            )?)
        } else {
            None
        };
        Ok(Self {
            detector: native_runtime::session_blocking(detector, execution)?,
            matcher: dynamic,
            profile,
            size: image_size,
            keypoints,
            identity,
            features: FeatureCache::default(),
        })
    }
    pub(crate) fn pairs_blocking(
        &mut self,
        a: &GrayImage,
        b: &GrayImage,
    ) -> Result<Vec<PixelMatch>, InferenceError> {
        let mut extract = |image: &GrayImage| {
            superpoint::extract_with_threshold(
                &mut self.detector,
                image,
                self.size,
                self.keypoints,
                0.0005,
            )
        };
        let a = self.features.get_or_extract(a, &mut extract)?;
        let b = self.features.get_or_extract(b, &mut extract)?;
        if a.pixels.len() < 2 || b.pixels.len() < 2 {
            return Ok(Vec::new());
        }
        let mut inputs = Vec::new();
        for (i, f) in [&a, &b].into_iter().enumerate() {
            for (name, dimensions, values) in [
                ("keypoints", 2, &f.model_pixels),
                ("descriptors", 256, &f.descriptors),
            ] {
                let tensor = Tensor::from_array(([1, f.pixels.len(), dimensions], values.clone()))
                    .map_err(|e| native_runtime::runtime("build LightGlue input", e))?;
                inputs.push((format!("{name}{i}"), tensor.into_dyn()));
            }
        }
        let specialized = self
            .profile
            .as_mut()
            .and_then(|p| p.session_for(a.pixels.len(), b.pixels.len()));
        tracing::debug!(
            reference_keypoints = a.pixels.len(),
            query_keypoints = b.pixels.len(),
            profiled = specialized.is_some(),
            "LightGlue execution shape"
        );
        let session = specialized.unwrap_or(&mut self.matcher);
        let output = session
            .run(inputs)
            .map_err(|e| native_runtime::runtime("run LightGlue", e))?;
        let (shape, indices) = output
            .get("matches0")
            .ok_or_else(|| InferenceError::Invalid("missing LightGlue matches0".into()))?
            .try_extract_tensor::<i64>()
            .map_err(|e| native_runtime::runtime("decode LightGlue matches0", e))?;
        if shape.as_ref() != [1, a.pixels.len() as i64] {
            return Err(InferenceError::Invalid(
                "invalid LightGlue match shape".into(),
            ));
        }
        decode(indices, &a, &b)
    }
}
impl ImageMatcher for LightGlueMatcher {
    fn identity(&self) -> &str {
        &self.identity
    }
    fn match_images_blocking(
        &mut self,
        reference: &GrayImage,
        query: &GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        if reference.dimensions() != query.dimensions() {
            return Err(VisualError::Dimensions {
                reference: reference.dimensions(),
                query: query.dimensions(),
            });
        }
        self.pairs_blocking(reference, query)
            .map_err(|source| VisualError::Backend {
                backend: self.identity.clone(),
                source: Box::new(source),
            })
    }
}
fn digest_blocking(path: &Path) -> Result<String, InferenceError> {
    let bytes = std::fs::read(path).map_err(|source| InferenceError::Io {
        path: path.to_owned(),
        source,
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
fn decode(indices: &[i64], a: &Features, b: &Features) -> Result<Vec<PixelMatch>, InferenceError> {
    if indices.len() != a.pixels.len() {
        return Err(InferenceError::Invalid(
            "LightGlue match count differs from keypoints".into(),
        ));
    }
    let mut pairs = Vec::new();
    let mut used = vec![false; b.pixels.len()];
    for (i, &j) in indices.iter().enumerate() {
        if j == -1 {
            continue;
        }
        let j = usize::try_from(j)
            .map_err(|_| InferenceError::Invalid("invalid LightGlue match index".into()))?;
        let pixel = b
            .pixels
            .get(j)
            .ok_or_else(|| InferenceError::Invalid("LightGlue index is out of range".into()))?;
        if used[j] {
            return Err(InferenceError::Invalid(
                "LightGlue assignments are not mutual".into(),
            ));
        }
        used[j] = true;
        pairs.push(PixelMatch {
            reference: a.pixels[i].into(),
            query: (*pixel).into(),
        });
    }
    Ok(pairs)
}
#[cfg(test)]
mod tests;
