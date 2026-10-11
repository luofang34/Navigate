use super::XFeatGraph;
#[cfg(all(feature = "rustnn-coreml", target_os = "macos"))]
use super::{XFeatStages, digest_file_blocking};
use crate::{InferenceError, features::Features, xfeat, xfeat_dense, xfeat_image};
use image::GrayImage;
use ort::session::Session;

pub(super) enum Detector {
    Ort(Session),
    #[cfg(all(feature = "rustnn-coreml", target_os = "macos"))]
    Rustnn(Box<crate::rustnn_dense::DenseGraph>),
}
impl Detector {
    pub(super) fn statistics(&self) -> Option<crate::TensorIoStatistics> {
        match self {
            Self::Ort(_) => None,
            #[cfg(all(feature = "rustnn-coreml", target_os = "macos"))]
            Self::Rustnn(graph) => graph.statistics(),
        }
    }
    pub(super) fn extract(
        &mut self,
        graph: XFeatGraph,
        image: &GrayImage,
        limit: usize,
    ) -> Result<Features, InferenceError> {
        match self {
            Self::Ort(session) => match graph {
                XFeatGraph::Sparse => xfeat::extract(session, image, limit),
                XFeatGraph::Dense => xfeat_dense::extract(session, image, limit),
                XFeatGraph::DenseImage => xfeat_image::extract(session, image, limit),
            },
            #[cfg(all(feature = "rustnn-coreml", target_os = "macos"))]
            Self::Rustnn(graph) => graph.extract(image, limit),
        }
    }
}

#[cfg(all(feature = "rustnn-coreml", target_os = "macos"))]
impl XFeatStages {
    /// Extract the raw dense descriptor grid for a correlation experiment.
    ///
    /// The output is host-owned f32 in CHW order, with shape [64, 72, 100].
    /// Values are before per-cell normalization. This runs the encoder and reads
    /// all three graph outputs. It copies the descriptor output into a new vector.
    /// Correlation requires an explicit feature mask.
    ///
    /// # Errors
    /// Rejects an ORT detector or a failed dense graph execution.
    pub fn extract_dense_grid_blocking(
        &mut self,
        image: &GrayImage,
    ) -> Result<Vec<f32>, InferenceError> {
        match &mut self.detector {
            Detector::Rustnn(graph) => graph.grid_blocking(image),
            Detector::Ort(_) => Err(InferenceError::Invalid(
                "dense grid export requires a rustnn graph".into(),
            )),
        }
    }

    /// Load a rustnn dense image graph and an optional ORT matcher.
    ///
    /// The graph must expose fixed 800 by 600 grayscale or RGB input and dense XFeat maps.
    /// Core ML chooses kernels inside its graph. No ORT extraction partitions run.
    ///
    /// # Errors
    /// Rejects invalid feature limits, incompatible graphs, or device failures.
    pub fn load_rustnn_blocking(
        detector: &std::path::Path,
        matcher: Option<&std::path::Path>,
        extraction: crate::Provider,
        matching: &crate::ExecutionConfig,
        limit: usize,
    ) -> Result<Self, InferenceError> {
        use sha2::{Digest, Sha256};
        if !(32..=4096).contains(&limit) {
            return Err(InferenceError::Invalid(
                "feature limit must be 32 through 4096".into(),
            ));
        }
        let detector = crate::rustnn_dense::DenseGraph::load_blocking(detector, extraction)?;
        let contract = Sha256::digest(format!(
            "xfeat-rustnn-v1/{}/{limit}/pixel-centre-bilinear/normalized-f32-64",
            detector.digest
        ))
        .into();
        let matcher_hash = matcher
            .map(digest_file_blocking)
            .transpose()?
            .unwrap_or_else(|| "mutual".into());
        let identity = format!(
            "xfeat-rustnn-v1/{}/{matcher_hash}/kp{limit}/extract-{}/match-{:?}",
            detector.digest, detector.device, matching.provider
        );
        Ok(Self {
            detector: Detector::Rustnn(Box::new(detector)),
            matcher: matcher
                .map(|path| crate::native_runtime::session_blocking(path, matching))
                .transpose()?,
            graph: XFeatGraph::DenseImage,
            limit,
            contract,
            identity,
        })
    }
}
