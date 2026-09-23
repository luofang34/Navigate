//! Model selection is separate from execution provider selection.
use crate::{ExecutionConfig, InferenceError, features, native_runtime, superpoint, xfeat};
use image::GrayImage;
use navigate_visual::{ImageMatcher, PixelMatch, VisualError};
use ort::{session::Session, value::Tensor};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// Exports supported by the two concrete native adapters.
pub enum MatcherFiles {
    /// Sparse XFeat export with 800 by 600 model coordinates.
    XFeat {
        /// Model path. Weights are not supplied by this library.
        model: PathBuf,
    },
    /// Dense SuperPoint and dynamic-keypoint SuperGlue exports.
    SuperGlue {
        /// Dense score-logit and descriptor model.
        detector: PathBuf,
        /// Pixel-normalized SuperGlue model.
        matcher: PathBuf,
        /// Width and height embedded in this matcher's coordinate normalization.
        image_size: [usize; 2],
    },
}
enum Model {
    XFeat(Session),
    SuperGlue {
        detector: Session,
        matcher: Session,
        size: [usize; 2],
    },
}
/// A native learned matcher usable by [`navigate_visual::Localizer`].
///
/// Models and execution setup stay here. This type does not admit positions,
/// combine evidence, or interpret descriptor similarity as geographic confidence.
pub struct OnnxMatcher {
    model: Model,
    identity: String,
    keypoints: usize,
}
impl OnnxMatcher {
    /// Load explicit model files and keep sessions resident.
    ///
    /// # Errors
    /// Rejects invalid feature limits, model contracts, and runtime failures.
    pub fn load_blocking(
        files: MatcherFiles,
        execution: ExecutionConfig,
        keypoints: usize,
    ) -> Result<Self, InferenceError> {
        if !(32..=4096).contains(&keypoints) {
            return Err(InferenceError::Invalid(
                "keypoint limit must be 32 through 4096".into(),
            ));
        }
        let (model, identity) = match files {
            MatcherFiles::XFeat { model } => {
                let hash = digest_blocking(&model)?;
                (
                    Model::XFeat(native_runtime::session_blocking(&model, &execution)?),
                    format!("xfeat-mutual-v1/{hash}"),
                )
            }
            MatcherFiles::SuperGlue {
                detector,
                matcher,
                image_size,
            } => {
                if image_size
                    .iter()
                    .any(|d| *d < 64 || *d > 1920 || d % 8 != 0)
                {
                    return Err(InferenceError::Invalid(
                        "SuperGlue normalization size must match its export".into(),
                    ));
                }
                let identity = format!(
                    "superpoint-superglue-v1/{}/{}/{}x{}",
                    digest_blocking(&detector)?,
                    digest_blocking(&matcher)?,
                    image_size[0],
                    image_size[1]
                );
                (
                    Model::SuperGlue {
                        detector: native_runtime::session_blocking(&detector, &execution)?,
                        matcher: native_runtime::session_blocking(&matcher, &execution)?,
                        size: image_size,
                    },
                    identity,
                )
            }
        };
        Ok(Self {
            model,
            identity: format!(
                "{identity}/ort-requested-{:?}/kp{keypoints}",
                execution.provider
            ),
            keypoints,
        })
    }
    fn pairs_blocking(
        &mut self,
        first: &GrayImage,
        second: &GrayImage,
    ) -> Result<Vec<PixelMatch>, InferenceError> {
        // Empty visual evidence must not enter dynamic sparse model branches.
        if blank(first) || blank(second) {
            return Ok(vec![]);
        }
        match &mut self.model {
            Model::XFeat(session) => {
                let a = xfeat::extract(session, first, self.keypoints)?;
                let b = xfeat::extract(session, second, self.keypoints)?;
                Ok(features::mutual(&a, &b))
            }
            Model::SuperGlue {
                detector,
                matcher,
                size,
            } => {
                let a = superpoint::extract(detector, first, *size, self.keypoints)?;
                let b = superpoint::extract(detector, second, *size, self.keypoints)?;
                glue(matcher, &a, &b)
            }
        }
    }
}
impl ImageMatcher for OnnxMatcher {
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
fn blank(image: &GrayImage) -> bool {
    let min = image.as_raw().iter().copied().min().unwrap_or(0);
    let max = image.as_raw().iter().copied().max().unwrap_or(0);
    max.saturating_sub(min) < 2
}
fn digest_blocking(path: &std::path::Path) -> Result<String, InferenceError> {
    let bytes = std::fs::read(path).map_err(|source| InferenceError::Io {
        path: path.to_owned(),
        source,
    })?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
fn glue(
    session: &mut Session,
    a: &features::Features,
    b: &features::Features,
) -> Result<Vec<PixelMatch>, InferenceError> {
    use native_runtime::runtime;
    if a.pixels.is_empty() || b.pixels.is_empty() {
        return Ok(vec![]);
    }
    let mut inputs = Vec::new();
    for (i, f) in [a, b].into_iter().enumerate() {
        let n = f.pixels.len();
        for (name, shape, data) in [
            ("keypoints", vec![1, n, 2], f.model_pixels.clone()),
            ("scores", vec![1, n], f.scores.clone()),
            ("descriptors", vec![1, 256, n], f.channel_major()),
        ] {
            inputs.push((
                format!("{name}{i}"),
                Tensor::from_array((shape, data))
                    .map_err(|e| runtime("build SuperGlue input", e))?
                    .into_dyn(),
            ));
        }
    }
    let output = session
        .run(inputs)
        .map_err(|e| runtime("run SuperGlue", e))?;
    let value = output
        .get("matches0")
        .ok_or_else(|| InferenceError::Invalid("missing SuperGlue matches0".into()))?;
    let (_, indices) = value
        .try_extract_tensor::<i64>()
        .map_err(|e| runtime("decode SuperGlue indices", e))?;
    if indices.len() != a.pixels.len() {
        return Err(InferenceError::Invalid(
            "SuperGlue match count differs from keypoints".into(),
        ));
    }
    indices
        .iter()
        .enumerate()
        .filter_map(|(i, j)| {
            if *j < 0 {
                return None;
            }
            Some(
                b.pixels
                    .get(*j as usize)
                    .map(|p| PixelMatch {
                        reference: a.pixels[i].into(),
                        query: (*p).into(),
                    })
                    .ok_or_else(|| {
                        InferenceError::Invalid("SuperGlue match index is out of range".into())
                    }),
            )
        })
        .collect()
}
