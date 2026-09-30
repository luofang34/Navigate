//! CAMP global appearance retrieval behind the reference-ID boundary.
mod index;
mod pixels;
use crate::{ExecutionConfig, InferenceError, native_runtime};
use image::RgbImage;
pub use index::CampIndex;
use navigate_visual::place_retrieval::{
    ImageRetriever, ReferenceCatalog, ReferenceId, RetrievalError,
};
use ort::{session::Session, value::Tensor};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Resident fixed-shape CAMP model and its versioned reference index.
///
/// No model weights are embedded or downloaded. Descriptor similarity stays
/// inside this adapter and does not establish a pose or geographic confidence.
pub struct CampRetriever {
    session: Session,
    index: CampIndex,
    identity: String,
    cached: Option<([u8; 32], Vec<f32>)>,
}
impl CampRetriever {
    /// Load a 384-square RGB global descriptor export with embedded weights.
    ///
    /// # Errors
    /// Rejects a model/index identity mismatch or an execution failure.
    pub fn load_blocking(
        model: &Path,
        index: CampIndex,
        execution: &ExecutionConfig,
    ) -> Result<Self, InferenceError> {
        let bytes = std::fs::read(model).map_err(|source| InferenceError::Io {
            path: model.to_owned(),
            source,
        })?;
        let digest = format!("{:x}", Sha256::digest(bytes));
        if !digest.eq_ignore_ascii_case(&index.model_sha256) {
            return Err(InferenceError::Invalid(format!(
                "CAMP model {digest} does not match index model {}",
                index.model_sha256
            )));
        }
        let identity = format!(
            "camp-global-rgb-linear-u8-v1/{digest}/{}/ort-requested-{:?}",
            index.digest, execution.provider
        );
        Ok(Self {
            session: native_runtime::session_blocking(model, execution)?,
            index,
            identity,
            cached: None,
        })
    }

    fn descriptor_blocking(&mut self, image: &RgbImage) -> Result<&[f32], InferenceError> {
        let mut hash = Sha256::new();
        hash.update(image.width().to_le_bytes());
        hash.update(image.height().to_le_bytes());
        hash.update(image.as_raw());
        let digest: [u8; 32] = hash.finalize().into();
        if self.cached.as_ref().is_none_or(|(key, _)| *key != digest) {
            let data = pixels::prepare(image, 384)?;
            let tensor = Tensor::from_array(([1, 3, 384, 384], data))
                .map_err(|e| native_runtime::runtime("build CAMP RGB input", e))?;
            let output = self
                .session
                .run(ort::inputs!["image" => tensor])
                .map_err(|e| native_runtime::runtime("run CAMP global descriptor", e))?;
            let (shape, values) = output
                .get("descriptor")
                .ok_or_else(|| InferenceError::Invalid("missing CAMP descriptor output".into()))?
                .try_extract_tensor::<f32>()
                .map_err(|e| native_runtime::runtime("decode CAMP descriptor", e))?;
            if shape.as_ref() != [1, 1024] {
                return Err(InferenceError::Invalid(
                    "CAMP descriptor shape must be [1,1024]".into(),
                ));
            }
            index::validate_descriptor(values)?;
            self.cached = Some((digest, values.to_vec()));
        }
        self.cached
            .as_ref()
            .map(|(_, values)| values.as_slice())
            .ok_or_else(|| InferenceError::Invalid("CAMP descriptor cache is empty".into()))
    }
}

impl ImageRetriever for CampRetriever {
    fn identity(&self) -> &str {
        &self.identity
    }
    fn catalog(&self) -> &ReferenceCatalog {
        &self.index.catalog
    }
    fn rank_blocking(
        &mut self,
        query: &RgbImage,
        eligible: &[ReferenceId],
        limit: usize,
    ) -> Result<Vec<ReferenceId>, RetrievalError> {
        let rows = self.index.eligible_rows(eligible, limit)?;
        if rows.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let identity = self.identity.clone();
        let descriptor = self
            .descriptor_blocking(query)
            .map_err(|source| RetrievalError::Backend {
                backend: identity,
                source: Box::new(source),
            })?
            .to_vec();
        Ok(self.index.rank(&descriptor, &rows, limit))
    }
}
