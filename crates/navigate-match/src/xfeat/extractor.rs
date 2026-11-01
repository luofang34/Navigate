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

    /// Extract up to `limit` sparse features across a batch of `(image, valid_mask)` pairs.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when any input fails validation or device execution fails.
    fn extract_batch_masked_blocking(
        &self,
        items: &[(&GrayImage, Option<&GrayImage>)],
        limit: usize,
    ) -> Result<Vec<SparseFeatures>, MatchError> {
        let mut out = Vec::with_capacity(items.len());
        for &(img, mask) in items {
            out.push(self.extract_masked_blocking(img, mask, limit)?);
        }
        Ok(out)
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

    fn extract_batch_masked_blocking(
        &self,
        items: &[(&GrayImage, Option<&GrayImage>)],
        limit: usize,
    ) -> Result<Vec<SparseFeatures>, MatchError> {
        pollster::block_on(self.extract_batch_masked(items, limit))
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
        let min = image.as_raw().iter().copied().min().unwrap_or(0);
        let max = image.as_raw().iter().copied().max().unwrap_or(0);
        if max.saturating_sub(min) < 2 {
            return Ok(SparseFeatures::empty());
        }
        let shape_hint: Vec<Option<usize>> = {
            let guard = self.session.lock().map_err(|_| MatchError::Device {
                stage: "xfeat session lock",
                message: "mutex poisoned".into(),
            })?;
            guard
                .input_shape_hint()
                .map_or_else(Vec::new, <[Option<usize>]>::to_vec)
        };
        let channels = shape_hint.get(1).copied().flatten().unwrap_or(1);
        if channels != 1 && channels != 3 {
            return Err(MatchError::Output(
                "unsupported XFeat session input channel count",
            ));
        }
        let static_h = shape_hint.get(2).copied().flatten();
        let static_w = shape_hint.get(3).copied().flatten();
        let in_w = static_w.unwrap_or(XFEAT_INPUT_WIDTH);
        let in_h = static_h.unwrap_or(XFEAT_INPUT_HEIGHT);
        let prep_w = static_w.unwrap_or(PREPARE_WIDTH);
        let prep_h = static_h.unwrap_or(PREPARE_HEIGHT);

        let prepared = input::prepare(image, [prep_w, prep_h])?;
        let plane =
            decode::normalize_resized_dims(&prepared.values, [prep_w, prep_h], [in_w, in_h]);
        let buffer = if channels == 3 {
            let mut rgb = Vec::with_capacity(3 * plane.len());
            for _ in 0..3 {
                rgb.extend_from_slice(&plane);
            }
            rgb
        } else {
            plane
        };
        let input_tensor = HostTensor::new(vec![1, channels, in_h, in_w], buffer)
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
        decode_session_outputs(
            &outputs,
            &prepared,
            [prep_w, prep_h],
            [in_w, in_h],
            image,
            valid_mask,
            limit,
        )
    }
}

fn decode_session_outputs(
    outputs: &[NamedTensor<HostTensor>],
    prepared: &input::InputImage,
    prep_dims: [usize; 2],
    in_dims: [usize; 2],
    image: &GrayImage,
    valid_mask: Option<&GrayImage>,
    limit: usize,
) -> Result<SparseFeatures, MatchError> {
    if let Some(sparse_kp) = outputs
        .iter()
        .find(|t| t.name == "keypoints" && t.tensor.shape().last() == Some(&2))
    {
        return decode_sparse_session_outputs(
            outputs, sparse_kp, prepared, prep_dims, in_dims, image, valid_mask, limit,
        );
    }

    let desc_tensor = outputs
        .iter()
        .find(|t| t.name == "descriptors" || t.name == "feats")
        .or_else(|| {
            outputs
                .iter()
                .find(|t| t.tensor.shape().len() == 4 && t.tensor.shape()[1] == 64)
        })
        .or_else(|| outputs.first())
        .ok_or(MatchError::Output("missing dense XFeat descriptor output"))?;

    let d_shape = desc_tensor.tensor.shape();
    let (gh, gw) = if d_shape.len() == 4 && d_shape[1] == 64 {
        (d_shape[2], d_shape[3])
    } else {
        (in_dims[1] / 8, in_dims[0] / 8)
    };
    let grid_in_dims = [gw * 8, gh * 8];
    let cells = gh * gw;
    let descriptors = desc_tensor.tensor.as_slice().to_vec();

    let logits_opt = outputs
        .iter()
        .find(|t| {
            t.name == "keypoint_logits"
                || (t.tensor.shape().len() == 4 && t.tensor.shape()[1] == 65)
        })
        .or_else(|| {
            outputs
                .get(1)
                .filter(|t| t.tensor.as_slice().len() == 65 * cells)
        });

    let scores = if let Some(logits_t) = logits_opt {
        let rel = outputs
            .iter()
            .find(|t| {
                t.name == "reliability"
                    || t.name == "heatmap"
                    || (t.tensor.shape().len() == 4
                        && t.tensor.shape()[1] == 1
                        && t.tensor.as_slice().len() == cells)
            })
            .map_or_else(|| vec![1.0_f32; cells], |t| t.tensor.as_slice().to_vec());
        decode::reference_scores_grid(logits_t.tensor.as_slice(), &rel, grid_in_dims)
    } else if let Some(heat_full) = outputs
        .iter()
        .find(|t| !std::ptr::eq(*t, desc_tensor) && t.tensor.as_slice().len() == cells * 64)
    {
        let rel = vec![1.0_f32; cells];
        decode::scores_from_heatmap_grid(heat_full.tensor.as_slice(), &rel, grid_in_dims)
    } else {
        return Err(MatchError::Output("missing dense XFeat keypoint output"));
    };

    let scored = ScoredMaps {
        descriptors,
        scores,
    };
    decode::decode_grid(
        &scored,
        prepared,
        prep_dims,
        grid_in_dims,
        image,
        valid_mask,
        limit,
    )
}

#[allow(clippy::too_many_arguments)]
fn decode_sparse_session_outputs(
    outputs: &[NamedTensor<HostTensor>],
    kp_tensor: &NamedTensor<HostTensor>,
    prepared: &input::InputImage,
    prep_dims: [usize; 2],
    in_dims: [usize; 2],
    image: &GrayImage,
    valid_mask: Option<&GrayImage>,
    limit: usize,
) -> Result<SparseFeatures, MatchError> {
    let desc_tensor = outputs
        .iter()
        .find(|t| t.name == "descriptors" || t.tensor.shape().last() == Some(&64))
        .ok_or(MatchError::Output(
            "missing sparse XFeat descriptors output",
        ))?;
    let kp_slice = kp_tensor.tensor.as_slice();
    let desc_slice = desc_tensor.tensor.as_slice();
    let count = (kp_slice.len() / 2).min(desc_slice.len() / 64);
    let score_slice = outputs
        .iter()
        .find(|t| t.name == "scores" || t.tensor.as_slice().len() == count)
        .map(|t| t.tensor.as_slice());

    let sx = prep_dims[0] as f32 / in_dims[0].max(1) as f32;
    let sy = prep_dims[1] as f32 / in_dims[1].max(1) as f32;
    let mut ranked: Vec<(usize, f32, [f64; 2], [f32; 2])> = Vec::with_capacity(count);
    for i in 0..count {
        let kx = kp_slice[2 * i];
        let ky = kp_slice[2 * i + 1];
        if !kx.is_finite() || !ky.is_finite() {
            continue;
        }
        let model = [kx * sx, ky * sy];
        let pixel = prepared.source([f64::from(model[0]), f64::from(model[1])]);
        if !input::inside(pixel, image) {
            continue;
        }
        if let Some(mask) = valid_mask
            && !decode::mask_allows(pixel, mask)
        {
            continue;
        }
        let score = score_slice
            .and_then(|s| s.get(i).copied())
            .unwrap_or(1.0 - (i as f32) / (count.max(1) as f32));
        if score.is_finite() && score > 0.0 {
            ranked.push((i, score, pixel, model));
        }
    }
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));

    let mut out = SparseFeatures::empty();
    for (idx, score, pixel, model) in ranked.into_iter().take(limit) {
        let raw_d = &desc_slice[idx * 64..(idx + 1) * 64];
        let norm = raw_d.iter().map(|v| v * v).sum::<f32>().sqrt();
        if !norm.is_finite() || norm <= 1e-6 {
            continue;
        }
        let desc: [f32; 64] = std::array::from_fn(|c| raw_d[c] / norm);
        out.pixels.push(pixel);
        out.model_pixels.push(model);
        out.scores.push(score);
        out.descriptors.extend_from_slice(&desc);
    }
    Ok(out)
}
