//! Hardware-agnostic [`SparseFeatureExtractor`] trait and dense-map session adapter.
//!
//! [`SparseFeatureExtractor`] decouples bank builders and the live corridor tracker from
//! the dense neural execution engine. Burn [`XFeat`](super::XFeat) (`burn-f32`, `burn-f16`)
//! and [`ModelSessionExtractor`] (wrapping ONNX Runtime CoreML, TensorRT, CUDA, or a
//! vendor NPU [`ModelInference`](crate::ModelInference) session) share the exact same
//! preprocessing, masked peak filter, bicubic descriptor interpolation, and
//! [`XFeat::encoder_contract`](super::XFeat::encoder_contract) digest.

use super::{
    DenseMaps, GRID_HEIGHT, GRID_WIDTH, PREPARE_HEIGHT, PREPARE_WIDTH, SparseFeatures,
    XFEAT_INPUT_HEIGHT, XFEAT_INPUT_WIDTH, XFeat, decode, score::ScoredMaps,
};
use crate::{
    MatchError, input,
    traits::{HostTensor, ModelInference, NamedTensor},
};
use image::GrayImage;
use std::sync::Mutex;

/// Floating-point precision of the dense [`XFeat`] convolution backbone.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum XFeatPrecision {
    /// Full `f32` weights and activations across all layers.
    #[default]
    F32,
    /// `f16` convolution backbone with `f32` sigmoid, softmax, and L2 normalization heads.
    F16,
}

impl XFeatPrecision {
    /// Return the stable runtime label of this precision (`"burn-f32"` or `"burn-f16"`).
    #[must_use]
    pub const fn runtime_name(self) -> &'static str {
        match self {
            Self::F32 => "burn-f32",
            Self::F16 => "burn-f16",
        }
    }
}

/// Prepared letterbox geometry and normalized `[1, 1, 576, 800]` dense input buffer.
#[derive(Clone, Debug)]
pub struct PreparedXFeatImage {
    pub(super) inner: input::InputImage,
    /// Normalized `[1, 1, 576, 800]` row-major `f32` input values.
    pub normalized: Vec<f32>,
}

impl XFeat {
    /// Prepare `image` into the normalized `[1, 1, 576, 800]` buffer expected by dense `XFeat`.
    ///
    /// Returns `Ok(None)` when `image` has contrast below two grayscale levels.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `image` dimensions fall outside `16..=8192`.
    pub fn prepare_dense_input(
        image: &GrayImage,
    ) -> Result<Option<PreparedXFeatImage>, MatchError> {
        let min = image.as_raw().iter().copied().min().unwrap_or(0);
        let max = image.as_raw().iter().copied().max().unwrap_or(0);
        if max.saturating_sub(min) < 2 {
            return Ok(None);
        }
        let inner = input::prepare(image, [PREPARE_WIDTH, PREPARE_HEIGHT])?;
        let normalized = decode::normalize_resized(&inner.values);
        Ok(Some(PreparedXFeatImage { inner, normalized }))
    }

    /// Decode dense `XFeat` outputs into masked [`SparseFeatures`] on the host.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when tensor dimensions, `valid_mask`, or `limit` are invalid.
    pub fn decode_dense_masked(
        maps: &DenseMaps,
        prepared: &PreparedXFeatImage,
        image: &GrayImage,
        valid_mask: Option<&GrayImage>,
        limit: usize,
    ) -> Result<SparseFeatures, MatchError> {
        if limit == 0 || limit > 4096 {
            return Err(MatchError::Output("feature limit must be 1 through 4096"));
        }
        if let Some(mask) = valid_mask
            && mask.dimensions() != image.dimensions()
        {
            return Err(MatchError::Output(
                "valid mask dimensions do not match image",
            ));
        }
        let cells = GRID_HEIGHT * GRID_WIDTH;
        if maps.descriptors.len() != 64 * cells
            || maps.keypoint_logits.len() != 65 * cells
            || maps.reliability.len() != cells
        {
            return Err(MatchError::Output("dense XFeat map dimensions mismatch"));
        }
        let scores = decode::reference_scores(&maps.keypoint_logits, &maps.reliability);
        let scored = ScoredMaps {
            descriptors: maps.descriptors.clone(),
            scores,
        };
        decode::decode(&scored, &prepared.inner, image, valid_mask, limit)
    }
}

/// Thread-safe sparse `XFeat` keypoint and descriptor extractor across runtimes.
pub trait SparseFeatureExtractor: Send + Sync {
    /// Return the 32-byte canonical encoder contract digest.
    fn encoder_contract(&self) -> [u8; 32];

    /// Return the active runtime identifier (for example `"burn-f32"`, `"burn-f16"`, `"coreml"`, or `"tensorrt"`).
    fn runtime_name(&self) -> &str;

    /// Return `true` when `stamp` matches this extractor or `weights_file_sha256`.
    fn accepts_encoder_contract(&self, stamp: &[u8; 32], weights_file_sha256: &[u8; 32]) -> bool {
        *stamp == self.encoder_contract() || stamp == weights_file_sha256
    }

    /// Extract up to `limit` sparse features inside `valid_mask` from `image`.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when input dimensions, `limit`, or device execution fail.
    fn extract_masked_blocking(
        &self,
        image: &GrayImage,
        valid_mask: Option<&GrayImage>,
        limit: usize,
    ) -> Result<SparseFeatures, MatchError>;

    /// Extract up to `limit` unmasked sparse features from `image`.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when input dimensions, `limit`, or device execution fail.
    fn extract_blocking(
        &self,
        image: &GrayImage,
        limit: usize,
    ) -> Result<SparseFeatures, MatchError> {
        self.extract_masked_blocking(image, None, limit)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl SparseFeatureExtractor for XFeat {
    fn encoder_contract(&self) -> [u8; 32] {
        Self::encoder_contract(self)
    }

    fn runtime_name(&self) -> &str {
        self.precision().runtime_name()
    }

    fn extract_masked_blocking(
        &self,
        image: &GrayImage,
        valid_mask: Option<&GrayImage>,
        limit: usize,
    ) -> Result<SparseFeatures, MatchError> {
        pollster::block_on(self.extract_masked(image, valid_mask, limit))
    }
}

/// [`SparseFeatureExtractor`] adapter around any dense [`ModelInference`] session.
pub struct ModelSessionExtractor<S> {
    session: Mutex<S>,
    input_name: String,
    encoder_contract: [u8; 32],
    runtime_label: String,
}

impl<S> ModelSessionExtractor<S>
where
    S: ModelInference<Tensor = HostTensor> + Send,
{
    /// Wrap `session` with its `encoder_contract` and `runtime_label`.
    pub fn new(
        session: S,
        input_name: impl Into<String>,
        encoder_contract: [u8; 32],
        runtime_label: impl Into<String>,
    ) -> Self {
        Self {
            session: Mutex::new(session),
            input_name: input_name.into(),
            encoder_contract,
            runtime_label: runtime_label.into(),
        }
    }
}

impl<S> SparseFeatureExtractor for ModelSessionExtractor<S>
where
    S: ModelInference<Tensor = HostTensor> + Send,
    S::Error: std::fmt::Display,
{
    fn encoder_contract(&self) -> [u8; 32] {
        self.encoder_contract
    }

    fn runtime_name(&self) -> &str {
        &self.runtime_label
    }

    fn extract_masked_blocking(
        &self,
        image: &GrayImage,
        valid_mask: Option<&GrayImage>,
        limit: usize,
    ) -> Result<SparseFeatures, MatchError> {
        if limit == 0 || limit > 4096 {
            return Err(MatchError::Output("feature limit must be 1 through 4096"));
        }
        if let Some(mask) = valid_mask
            && mask.dimensions() != image.dimensions()
        {
            return Err(MatchError::Output(
                "valid mask dimensions do not match image",
            ));
        }
        let Some(prepared) = XFeat::prepare_dense_input(image)? else {
            return Ok(SparseFeatures::empty());
        };
        let input_tensor = HostTensor::new(
            vec![1, 1, XFEAT_INPUT_HEIGHT, XFEAT_INPUT_WIDTH],
            prepared.normalized.clone(),
        )
        .map_err(|_| MatchError::Output("invalid XFeat dense input tensor"))?;
        let inputs = [NamedTensor::new(self.input_name.clone(), input_tensor)];
        let outputs = {
            let mut guard = self.session.lock().map_err(|_| MatchError::Device {
                stage: "xfeat session lock",
                message: "mutex poisoned".into(),
            })?;
            guard.forward(&inputs).map_err(|e| MatchError::Device {
                stage: "xfeat model session forward",
                message: e.to_string(),
            })?
        };
        let find = |primary: &str, fallback_idx: usize| -> Result<Vec<f32>, MatchError> {
            outputs
                .iter()
                .find(|t| t.name == primary)
                .or_else(|| outputs.get(fallback_idx))
                .map(|t| t.tensor.as_slice().to_vec())
                .ok_or(MatchError::Output("missing dense XFeat session output"))
        };
        let maps = DenseMaps {
            descriptors: find("descriptors", 0)?,
            keypoint_logits: find("keypoint_logits", 1)?,
            reliability: find("reliability", 2)?,
        };
        XFeat::decode_dense_masked(&maps, &prepared, image, valid_mask, limit)
    }
}
