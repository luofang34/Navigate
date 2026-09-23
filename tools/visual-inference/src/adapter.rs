//! Model selection is separate from execution provider selection.
use crate::{
    ExecutionConfig, InferenceError, features, lighterglue, native_runtime, superpoint, xfeat,
    xfeat_dense,
};
use image::GrayImage;
use navigate_visual::{ImageMatcher, PixelMatch, VisualError};
use ort::{session::Session, value::Tensor};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// Exports supported by the concrete native adapters.
pub enum MatcherFiles {
    /// Sparse XFeat export with 800 by 600 model coordinates.
    XFeat {
        /// Model path. Weights are not supplied by this library.
        model: PathBuf,
    },
    /// Sparse XFeat and normalized-keypoint LighterGlue exports.
    XFeatLighterGlue {
        /// Sparse XFeat detector.
        detector: PathBuf,
        /// LighterGlue log-assignment model.
        matcher: PathBuf,
    },
    /// Dense XFeat export with host preprocessing and sparse decoding.
    XFeatDense {
        /// Fixed 800 by 576 dense detector.
        detector: PathBuf,
        /// Optional LighterGlue matcher; absent uses mutual descriptors.
        matcher: Option<PathBuf>,
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
    XFeat {
        detector: Session,
        matcher: Option<Session>,
        dense: bool,
    },
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
        let (model, identity) = load_model_blocking(files, &execution)?;
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
            Model::XFeat {
                detector,
                matcher,
                dense,
            } => {
                let extract = if *dense {
                    xfeat_dense::extract
                } else {
                    xfeat::extract
                };
                let a = extract(detector, first, self.keypoints)?;
                let b = extract(detector, second, self.keypoints)?;
                match matcher {
                    Some(session) => lighterglue::match_features(session, &a, &b),
                    None => Ok(features::mutual(&a, &b)),
                }
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

fn load_model_blocking(
    files: MatcherFiles,
    execution: &ExecutionConfig,
) -> Result<(Model, String), InferenceError> {
    Ok(match files {
        MatcherFiles::XFeat { model } => {
            let hash = digest_blocking(&model)?;
            (
                Model::XFeat {
                    detector: native_runtime::session_blocking(&model, execution)?,
                    matcher: None,
                    dense: false,
                },
                format!("xfeat-mutual-v1/{hash}"),
            )
        }
        MatcherFiles::XFeatLighterGlue { detector, matcher } => {
            let identity = format!(
                "xfeat-lighterglue-v1/{}/{}",
                digest_blocking(&detector)?,
                digest_blocking(&matcher)?
            );
            (
                Model::XFeat {
                    detector: native_runtime::session_blocking(&detector, execution)?,
                    matcher: Some(native_runtime::session_blocking(&matcher, execution)?),
                    dense: false,
                },
                identity,
            )
        }
        MatcherFiles::XFeatDense { detector, matcher } => {
            load_dense_blocking(detector, matcher, execution)?
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
                    detector: native_runtime::session_blocking(&detector, execution)?,
                    matcher: native_runtime::session_blocking(&matcher, execution)?,
                    size: image_size,
                },
                identity,
            )
        }
    })
}

fn load_dense_blocking(
    detector: PathBuf,
    matcher: Option<PathBuf>,
    execution: &ExecutionConfig,
) -> Result<(Model, String), InferenceError> {
    let identity = format!(
        "xfeat-dense-v1/{}/{}",
        digest_blocking(&detector)?,
        matcher
            .as_ref()
            .map(|p| digest_blocking(p))
            .transpose()?
            .unwrap_or_else(|| "mutual".into())
    );
    Ok((
        Model::XFeat {
            detector: native_runtime::session_blocking(&detector, execution)?,
            matcher: matcher
                .map(|p| native_runtime::session_blocking(&p, execution))
                .transpose()?,
            dense: true,
        },
        identity,
    ))
}
