//! Host settings are separate from navigation evidence.
use crate::Error;
use navigate_visual_onnx::{ExecutionConfig, LoFtrMatcher, Provider, XFeatGraph, XFeatStages};
use navigate_visual_workflow::ResourceLimits;
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub runtime: PathBuf,
    pub detector: PathBuf,
    pub graph: Graph,
    #[serde(default)]
    pub engine: Engine,
    pub matcher: Option<PathBuf>,
    pub extraction: Execution,
    pub matching: Execution,
    pub keypoints: usize,
    pub limits: Limits,
    #[serde(default)]
    pub refinement_patches: bool,
}
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Graph {
    Sparse,
    Dense,
    DenseImage,
    Loftr,
    LoftrSplit,
}
#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Engine {
    #[default]
    Ort,
    Rustnn,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Execution {
    pub provider: Device,
    pub threads: usize,
    pub allow_cpu_fallback: bool,
    #[serde(default)]
    pub profile_prefix: Option<PathBuf>,
}
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Device {
    Cpu,
    CoremlGpu,
    CoremlAne,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub image_pixels: usize,
    pub features_per_image: usize,
    pub candidates: usize,
    pub reference_bytes: usize,
    pub query_feature_bytes: usize,
}
impl Limits {
    pub fn model(&self) -> ResourceLimits {
        ResourceLimits {
            image_pixels: self.image_pixels,
            features_per_image: self.features_per_image,
            candidates: self.candidates,
            reference_bytes: self.reference_bytes,
            query_feature_bytes: self.query_feature_bytes,
        }
    }
}
impl Execution {
    pub fn model(&self) -> ExecutionConfig {
        ExecutionConfig {
            provider: match self.provider {
                Device::Cpu => Provider::Cpu,
                Device::CoremlGpu => Provider::CoreMlGpu,
                Device::CoremlAne => Provider::CoreMlAne,
            },
            threads: self.threads,
            allow_cpu_fallback: self.allow_cpu_fallback,
            profile_prefix: self.profile_prefix.clone(),
            ..ExecutionConfig::default()
        }
    }
}
pub enum Backend {
    Features(Box<XFeatStages>),
    Pair(Box<LoFtrMatcher>),
}

impl Config {
    pub fn load_blocking(&self, root: &Path) -> Result<Backend, Error> {
        if self.keypoints > self.limits.features_per_image || self.limits.image_pixels > 8192 * 8192
        {
            return Err(Error::Input(
                "feature or image limit is incompatible with the configured model".into(),
            ));
        }
        navigate_visual_onnx::initialize_blocking(&root.join(&self.runtime))?;
        if matches!(self.engine, Engine::Rustnn) {
            return self.rustnn_blocking(root);
        }
        if matches!(self.graph, Graph::Loftr | Graph::LoftrSplit) {
            let mut matcher = if matches!(self.graph, Graph::LoftrSplit) {
                let fine = self
                    .matcher
                    .as_ref()
                    .ok_or_else(|| Error::Input("split LoFTR requires a fine model".into()))?;
                LoFtrMatcher::load_split_blocking(
                    &root.join(&self.detector),
                    &root.join(fine),
                    &self.extraction.model(),
                    &self.matching.model(),
                )?
            } else {
                if self.matcher.is_some() {
                    return Err(Error::Input(
                        "LoFTR does not use a separate matcher model".into(),
                    ));
                }
                LoFtrMatcher::load_blocking(&root.join(&self.detector), &self.matching.model())?
            };
            if self.refinement_patches {
                matcher = matcher.with_refinement_patches();
            }
            return Ok(Backend::Pair(Box::new(matcher)));
        }
        if self.refinement_patches {
            return Err(Error::Input(
                "patch refinement requires the LoFTR graph".into(),
            ));
        }
        let matcher = self.matcher.as_ref().map(|p| root.join(p));
        Ok(Backend::Features(Box::new(XFeatStages::load_blocking(
            &root.join(&self.detector),
            match self.graph {
                Graph::Sparse => XFeatGraph::Sparse,
                Graph::Dense => XFeatGraph::Dense,
                Graph::DenseImage => XFeatGraph::DenseImage,
                Graph::Loftr | Graph::LoftrSplit => {
                    return Err(Error::Input("LoFTR must use the pair path".into()));
                }
            },
            matcher.as_deref(),
            &self.extraction.model(),
            &self.matching.model(),
            self.keypoints,
        )?)))
    }
}

impl Config {
    fn rustnn_blocking(&self, root: &Path) -> Result<Backend, Error> {
        if !matches!(self.graph, Graph::DenseImage) || self.refinement_patches {
            return Err(Error::Input(
                "rustnn requires dense_image without patch refinement".into(),
            ));
        }
        self.rustnn_backend_blocking(root)
    }
    #[cfg(all(feature = "rustnn-coreml", target_os = "macos"))]
    fn rustnn_backend_blocking(&self, root: &Path) -> Result<Backend, Error> {
        let matcher = self.matcher.as_ref().map(|p| root.join(p));
        Ok(Backend::Features(Box::new(
            XFeatStages::load_rustnn_blocking(
                &root.join(&self.detector),
                matcher.as_deref(),
                self.extraction.model().provider,
                &self.matching.model(),
                self.keypoints,
            )?,
        )))
    }
    #[cfg(not(all(feature = "rustnn-coreml", target_os = "macos")))]
    fn rustnn_backend_blocking(&self, _: &Path) -> Result<Backend, Error> {
        Err(Error::Input(
            "build with rustnn-coreml on macOS for this engine".into(),
        ))
    }
}
