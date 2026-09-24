//! Fixed-shape detector-free correspondences behind the Navigate matcher port.
use crate::{ExecutionConfig, InferenceError, native_runtime, preprocessing};
use image::GrayImage;
use navigate_visual::{ImageMatcher, PixelMatch, VisualError};
use ort::{session::Session, value::Tensor};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Resident LoFTR adapter for the 640 by 480 dual-softmax ONNX export.
///
/// The export takes grayscale `image0` and `image1` tensors and returns
/// `keypoints0`, `keypoints1`, and `confidence`. Scores stay inside this adapter.
/// The shared geometry verifier decides whether a camera pose is acceptable.
pub struct LoFtrMatcher {
    session: Session,
    identity: String,
}
impl LoFtrMatcher {
    /// Load an explicit model with the host's execution configuration.
    ///
    /// # Errors
    /// Returns model-file and runtime errors with their source context.
    pub fn load_blocking(path: &Path, execution: &ExecutionConfig) -> Result<Self, InferenceError> {
        let bytes = std::fs::read(path).map_err(|source| InferenceError::Io {
            path: path.to_owned(),
            source,
        })?;
        let identity = format!(
            "loftr-dual-softmax-640x480/{:x}/ort-requested-{:?}",
            Sha256::digest(&bytes),
            execution.provider
        );
        let session = native_runtime::session_blocking(path, execution)?;
        Ok(Self { session, identity })
    }
    fn pairs_blocking(
        &mut self,
        first: &GrayImage,
        second: &GrayImage,
    ) -> Result<Vec<PixelMatch>, InferenceError> {
        let a = preprocessing::prepare(first, 640, 480, 1)?;
        let b = preprocessing::prepare(second, 640, 480, 1)?;
        let tensor = |data: &[f32]| {
            Tensor::from_array(([1, 1, 480, 640], data.to_vec()))
                .map_err(|e| native_runtime::runtime("build LoFTR input", e))
        };
        let output = self
            .session
            .run(ort::inputs!["image0"=>tensor(&a.data)?, "image1"=>tensor(&b.data)?])
            .map_err(|e| native_runtime::runtime("run LoFTR", e))?;
        let read = |name: &str, columns: Option<i64>| -> Result<&[f32], InferenceError> {
            let (shape, data) = output
                .get(name)
                .ok_or_else(|| InferenceError::Invalid(format!("missing LoFTR {name}")))?
                .try_extract_tensor::<f32>()
                .map_err(|e| native_runtime::runtime("decode LoFTR output", e))?;
            if shape.len() != if columns.is_some() { 2 } else { 1 }
                || columns.is_some_and(|n| shape[1] != n)
            {
                return Err(InferenceError::Invalid(format!(
                    "invalid LoFTR {name} shape"
                )));
            }
            Ok(data)
        };
        decode(
            read("keypoints0", Some(2))?,
            read("keypoints1", Some(2))?,
            read("confidence", None)?,
            [&a, &b],
            [first, second],
        )
    }
}
impl ImageMatcher for LoFtrMatcher {
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
        if [reference, query].iter().any(|image| {
            let min = image.as_raw().iter().copied().min().unwrap_or(0);
            let max = image.as_raw().iter().copied().max().unwrap_or(0);
            max.saturating_sub(min) < 2
        }) {
            return Ok(Vec::new());
        }
        self.pairs_blocking(reference, query)
            .map_err(|source| VisualError::Backend {
                backend: self.identity.clone(),
                source: Box::new(source),
            })
    }
}
fn decode(
    a: &[f32],
    b: &[f32],
    confidence: &[f32],
    transforms: [&preprocessing::InputImage; 2],
    images: [&GrayImage; 2],
) -> Result<Vec<PixelMatch>, InferenceError> {
    if a.len() != confidence.len() * 2
        || b.len() != a.len()
        || a.iter().chain(b).chain(confidence).any(|v| !v.is_finite())
    {
        return Err(InferenceError::Invalid(
            "invalid LoFTR output values or lengths".into(),
        ));
    }
    let indices: Vec<_> = (0..confidence.len())
        .filter(|&i| confidence[i] > 0.2)
        .collect();
    Ok(indices
        .into_iter()
        .filter_map(|i| {
            let reference = transforms[0].pixel([f64::from(a[i * 2]), f64::from(a[i * 2 + 1])]);
            let query = transforms[1].pixel([f64::from(b[i * 2]), f64::from(b[i * 2 + 1])]);
            (preprocessing::inside(reference, images[0]) && preprocessing::inside(query, images[1]))
                .then(|| PixelMatch {
                    reference: reference.into(),
                    query: query.into(),
                })
        })
        .take(4096)
        .collect())
}
#[cfg(test)]
mod tests;
