//! Pure-Rust XFeat dense backbone and sparse keypoint decoder on burn.
//!
//! [`XFeat`] executes the 27 folded convolution layers of `verlab/accelerated_features`
//! on any compiled burn [`Device`], scores keypoint candidates on the device, and
//! decodes unit-length 64-D descriptors and source image keypoints on the host.
mod decode;
mod extractor;
mod score;
mod sparse;
mod upsample;

pub use extractor::{
    ModelSessionExtractor, PreparedXFeatImage, SparseFeatureExtractor, XFeatPrecision,
};
pub use sparse::SparseFeatures;

use crate::{
    MatchError, Weights, input,
    loftr::layers::{Conv, upload},
};
use burn::{
    backend::{Backend, Dispatch},
    tensor::{DType, Device, FloatDType, Tensor, TensorData, Transaction, activation::sigmoid},
};
use image::GrayImage;
use score::ScoredMaps;
use upsample::Upsample;

/// Letterbox width before vertical resize to 576 pixels.
const PREPARE_WIDTH: usize = 800;
/// Letterbox height before vertical resize to 576 pixels.
const PREPARE_HEIGHT: usize = 600;
/// Dense network input width in pixels.
pub const XFEAT_INPUT_WIDTH: usize = 800;
/// Dense network input height in pixels.
pub const XFEAT_INPUT_HEIGHT: usize = 576;
/// Coarse grid height (`576 / 8`).
const GRID_HEIGHT: usize = 72;
/// Coarse grid width (`800 / 8`).
const GRID_WIDTH: usize = 100;

/// Dense feature maps produced by one XFeat forward pass.
#[derive(Clone, Debug)]
pub struct DenseMaps {
    /// Coarse descriptor map `[1, 64, 72, 100]`.
    pub descriptors: Vec<f32>,
    /// Keypoint cell logits `[1, 65, 72, 100]`.
    pub keypoint_logits: Vec<f32>,
    /// Reliability map `[1, 1, 72, 100]`.
    pub reliability: Vec<f32>,
}

/// Pure-Rust XFeat feature extractor on one burn [`Device`].
#[derive(Clone, Debug)]
pub struct XFeat {
    block1: [Conv; 4],
    skip1: Conv,
    block2: [Conv; 2],
    block3: [Conv; 3],
    block4: [Conv; 3],
    block5: [Conv; 4],
    block_fusion: [Conv; 3],
    heatmap_head: [Conv; 3],
    keypoint_head: [Conv; 4],
    upsample: Upsample,
    device: Device,
    precision: XFeatPrecision,
    encoder_contract: [u8; 32],
}

/// Device tensors of one dense forward pass before read-back.
struct DenseTensors {
    descriptors: Tensor<4>,
    logits: Tensor<4>,
    reliability: Tensor<4>,
}

impl XFeat {
    /// Canonical description of the dense encoder output that banks depend on.
    ///
    /// [`Self::encoder_contract`] hashes this text together with the trained weights.
    pub const ENCODER_CONTRACT_V1_DESCRIPTION: &str =
        "navigate-xfeat-v1:input=800x576:grid=100x72:dim=64:norm=instance:coord=half-pixel";

    /// Return the encoder contract digest of the loaded model.
    ///
    /// The digest is [`Weights::canonical_digest`] with
    /// [`Self::ENCODER_CONTRACT_V1_DESCRIPTION`] as the prefix. It does not depend
    /// on the file encoding or the runtime, so a bank that a burn session builds
    /// is valid for an ONNX Runtime or NPU session with the same trained weights.
    /// Retrained or changed weights give a different digest. Bank builders write
    /// it into the `encoder_contract` field of
    /// [`PrecomputedReferenceBank`](crate::PrecomputedReferenceBank) and
    /// [`CorridorLandmarkBank`](crate::CorridorLandmarkBank).
    pub fn encoder_contract(&self) -> [u8; 32] {
        self.encoder_contract
    }

    /// Return the encoder contract digest of `weights` without a device.
    ///
    /// An ONNX Runtime or NPU host has no burn model. It calls this function with
    /// the source safetensors file of its export to get the stamp that
    /// [`Self::encoder_contract`] gives for that file.
    pub fn encoder_contract_of(weights: &Weights) -> [u8; 32] {
        weights.canonical_digest(Self::ENCODER_CONTRACT_V1_DESCRIPTION)
    }

    /// Return `true` when a bank `encoder_contract` stamp is valid for this model.
    ///
    /// The stamp is valid when it is [`Self::encoder_contract`] or the SHA-256
    /// digest of the weight file that this model came from. Banks that record the
    /// weight-file digest stay valid only for that exact file.
    pub fn accepts_encoder_contract(
        &self,
        stamp: &[u8; 32],
        weights_file_sha256: &[u8; 32],
    ) -> bool {
        *stamp == self.encoder_contract || stamp == weights_file_sha256
    }

    /// Return `true` when `device` supports native [`XFeatPrecision::F16`] execution.
    pub fn supports_f16(device: &Device) -> bool {
        <Dispatch as Backend>::supports_dtype(device.as_dispatch(), DType::F16)
    }

    /// Return the active convolution precision of this model.
    pub fn precision(&self) -> XFeatPrecision {
        self.precision
    }

    /// Load named XFeat weights onto `device` in `f32` precision.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when any tensor is missing, has an unexpected shape,
    /// or remains unused.
    pub fn load(weights: Weights, device: &Device) -> Result<Self, MatchError> {
        Self::load_with_precision(weights, device, XFeatPrecision::F32)
    }

    /// Load named XFeat weights onto `device` with explicit `precision`.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `device` does not support `precision` or weight shapes fail.
    pub fn load_with_precision(
        mut weights: Weights,
        device: &Device,
        precision: XFeatPrecision,
    ) -> Result<Self, MatchError> {
        if precision == XFeatPrecision::F16 && !Self::supports_f16(device) {
            return Err(MatchError::Device {
                stage: "xfeat f16",
                message: "device does not support DType::F16".into(),
            });
        }
        let dtype = match precision {
            XFeatPrecision::F32 => FloatDType::F32,
            XFeatPrecision::F16 => FloatDType::F16,
        };
        let encoder_contract = Self::encoder_contract_of(&weights);
        let c = |w: &mut Weights, name: &str, shape: [usize; 3], stride: usize| {
            Conv::load_with_dtype(w, name, shape, stride, true, dtype, device)
        };
        let c32 = |w: &mut Weights, name: &str, shape: [usize; 3], stride: usize| {
            Conv::load_with_dtype(w, name, shape, stride, true, FloatDType::F32, device)
        };
        let model = Self {
            block1: [
                c(&mut weights, "block1.0", [4, 1, 3], 1)?,
                c(&mut weights, "block1.1", [8, 4, 3], 2)?,
                c(&mut weights, "block1.2", [8, 8, 3], 1)?,
                c(&mut weights, "block1.3", [24, 8, 3], 2)?,
            ],
            skip1: c(&mut weights, "skip1.1", [24, 1, 1], 1)?,
            block2: [
                c(&mut weights, "block2.0", [24, 24, 3], 1)?,
                c(&mut weights, "block2.1", [24, 24, 3], 1)?,
            ],
            block3: [
                c(&mut weights, "block3.0", [64, 24, 3], 2)?,
                c(&mut weights, "block3.1", [64, 64, 3], 1)?,
                c(&mut weights, "block3.2", [64, 64, 1], 1)?,
            ],
            block4: [
                c(&mut weights, "block4.0", [64, 64, 3], 2)?,
                c(&mut weights, "block4.1", [64, 64, 3], 1)?,
                c(&mut weights, "block4.2", [64, 64, 3], 1)?,
            ],
            block5: [
                c(&mut weights, "block5.0", [128, 64, 3], 2)?,
                c(&mut weights, "block5.1", [128, 128, 3], 1)?,
                c(&mut weights, "block5.2", [128, 128, 3], 1)?,
                c(&mut weights, "block5.3", [64, 128, 1], 1)?,
            ],
            block_fusion: [
                c(&mut weights, "block_fusion.0", [64, 64, 3], 1)?,
                c(&mut weights, "block_fusion.1", [64, 64, 3], 1)?,
                c32(&mut weights, "block_fusion.2", [64, 64, 1], 1)?,
            ],
            heatmap_head: [
                c32(&mut weights, "heatmap_head.0", [64, 64, 1], 1)?,
                c32(&mut weights, "heatmap_head.1", [64, 64, 1], 1)?,
                c32(&mut weights, "heatmap_head.2", [1, 64, 1], 1)?,
            ],
            keypoint_head: [
                c32(&mut weights, "keypoint_head.0", [64, 64, 1], 1)?,
                c32(&mut weights, "keypoint_head.1", [64, 64, 1], 1)?,
                c32(&mut weights, "keypoint_head.2", [64, 64, 1], 1)?,
                c32(&mut weights, "keypoint_head.3", [65, 64, 1], 1)?,
            ],
            upsample: Upsample::new_with_dtype(device, dtype),
            device: device.clone(),
            precision,
            encoder_contract,
        };
        weights.finish()?;
        Ok(model)
    }

    /// Borrow the burn device of this model.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Extract up to `limit` sparse keypoints and 64-D descriptors from `image`.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when the image dimensions are outside `16..=8192`,
    /// when `limit` is zero, or when device read-back fails.
    pub async fn extract(
        &self,
        image: &GrayImage,
        limit: usize,
    ) -> Result<SparseFeatures, MatchError> {
        self.extract_masked(image, None, limit).await
    }

    /// Extract up to `limit` sparse keypoints inside `valid_mask` from `image`.
    ///
    /// The decoder discards candidate keypoints where `valid_mask` has a pixel value below `128`.
    /// The decoder ranks the remaining candidates and keeps up to `limit` keypoints.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `valid_mask` dimensions do not match `image`.
    /// Returns [`MatchError`] when `image` dimensions are outside `16..=8192` or `limit` is outside `1..=4096`.
    /// Returns [`MatchError`] when device read-back fails.
    pub async fn extract_masked(
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
        let prepared = input::prepare(image, [PREPARE_WIDTH, PREPARE_HEIGHT])?;
        let normalized = decode::normalize_resized(&prepared.values);
        let maps = self.forward_scored(normalized).await?;
        decode::decode(&maps, &prepared, image, valid_mask, limit)
    }

    /// Extract up to `limit` sparse keypoints inside optional `valid_mask` across a batch of images.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when any image or mask fails validation or when device read-back fails.
    pub async fn extract_batch_masked(
        &self,
        items: &[(&GrayImage, Option<&GrayImage>)],
        limit: usize,
    ) -> Result<Vec<SparseFeatures>, MatchError> {
        if limit == 0 || limit > 4096 {
            return Err(MatchError::Output("feature limit must be 1 through 4096"));
        }
        if items.is_empty() {
            return Ok(Vec::new());
        }
        const MAX_BATCH: usize = 12;
        let mut results = vec![SparseFeatures::empty(); items.len()];
        let mut active_indices = Vec::with_capacity(items.len());
        let mut prepared_list = Vec::with_capacity(items.len());
        let mut normalized_list = Vec::with_capacity(items.len());

        for (idx, &(image, valid_mask)) in items.iter().enumerate() {
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
                continue;
            }
            let prepared = input::prepare(image, [PREPARE_WIDTH, PREPARE_HEIGHT])?;
            let normalized = decode::normalize_resized(&prepared.values);
            active_indices.push(idx);
            prepared_list.push(prepared);
            normalized_list.push(normalized);
        }

        let frame_elems = XFEAT_INPUT_HEIGHT * XFEAT_INPUT_WIDTH;
        let desc_stride = 64 * GRID_HEIGHT * GRID_WIDTH;
        let score_stride = XFEAT_INPUT_HEIGHT * XFEAT_INPUT_WIDTH;

        for start in (0..active_indices.len()).step_by(MAX_BATCH) {
            let end = (start + MAX_BATCH).min(active_indices.len());
            let chunk_len = end - start;
            let mut stacked = Vec::with_capacity(chunk_len * frame_elems);
            for norm in &normalized_list[start..end] {
                stacked.extend_from_slice(norm);
            }
            let batch_maps = self.forward_scored_batch(stacked, chunk_len).await?;
            for local in 0..chunk_len {
                let item_idx = active_indices[start + local];
                let (image, valid_mask) = items[item_idx];
                let scored = ScoredMaps {
                    descriptors: batch_maps.descriptors
                        [local * desc_stride..(local + 1) * desc_stride]
                        .to_vec(),
                    scores: batch_maps.scores[local * score_stride..(local + 1) * score_stride]
                        .to_vec(),
                };
                results[item_idx] = decode::decode(
                    &scored,
                    &prepared_list[start + local],
                    image,
                    valid_mask,
                    limit,
                )?;
            }
        }
        Ok(results)
    }

    /// Run the dense convolutional network on a normalized `576 × 800` buffer.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `normalized` does not have `576 * 800` values or
    /// when device execution fails.
    pub async fn forward_dense(&self, normalized: Vec<f32>) -> Result<DenseMaps, MatchError> {
        let maps = self.dense_tensors_batch(normalized, 1)?;
        let transaction = Transaction::default()
            .register(maps.descriptors)
            .register(maps.logits)
            .register(maps.reliability);
        let [descriptors, keypoint_logits, reliability] = read_back(transaction).await?;
        Ok(DenseMaps {
            descriptors,
            keypoint_logits,
            reliability,
        })
    }

    /// Run the dense network and score keypoint candidates on the device.
    async fn forward_scored(&self, normalized: Vec<f32>) -> Result<ScoredMaps, MatchError> {
        self.forward_scored_batch(normalized, 1).await
    }

    /// Run the dense network and score keypoint candidates for a batch of `batch` images.
    async fn forward_scored_batch(
        &self,
        normalized: Vec<f32>,
        batch: usize,
    ) -> Result<ScoredMaps, MatchError> {
        let maps = self.dense_tensors_batch(normalized, batch)?;
        let scores = score::score_map(maps.logits, maps.reliability, &self.upsample.reliability)?;
        let transaction = Transaction::default()
            .register(maps.descriptors)
            .register(scores);
        let [descriptors, scores] = read_back(transaction).await?;
        Ok(ScoredMaps {
            descriptors,
            scores,
        })
    }

    fn dense_tensors_batch(
        &self,
        normalized: Vec<f32>,
        batch: usize,
    ) -> Result<DenseTensors, MatchError> {
        let frame_elems = XFEAT_INPUT_HEIGHT * XFEAT_INPUT_WIDTH;
        if batch == 0 || normalized.len() != batch * frame_elems {
            return Err(MatchError::Output("dense XFeat input buffer size mismatch"));
        }
        let x_f32 = upload(
            normalized,
            [batch, 1, XFEAT_INPUT_HEIGHT, XFEAT_INPUT_WIDTH],
            &self.device,
        );
        let pooled_f32 = x_f32
            .clone()
            .reshape([batch, 1, GRID_HEIGHT * 2, 4, GRID_WIDTH * 2, 4])
            .mean_dim(5)
            .squeeze_dim::<5>(5)
            .mean_dim(3)
            .squeeze_dim::<4>(3);
        let patches = x_f32
            .clone()
            .reshape([batch, GRID_HEIGHT, 8, GRID_WIDTH, 8])
            .permute([0, 2, 4, 1, 3])
            .reshape([batch, 64, GRID_HEIGHT, GRID_WIDTH]);
        let (x, pooled) = if self.precision == XFeatPrecision::F16 {
            (
                x_f32.cast(FloatDType::F16),
                pooled_f32.cast(FloatDType::F16),
            )
        } else {
            (x_f32, pooled_f32)
        };
        let mut x1 = self.block1[0].forward_relu(x);
        for layer in &self.block1[1..] {
            x1 = layer.forward_relu(x1);
        }
        let skip = self.skip1.forward(pooled);
        let x2 = self.block2[1].forward_relu(self.block2[0].forward_relu(x1 + skip));
        let x3 = self.block3[2]
            .forward_relu(self.block3[1].forward_relu(self.block3[0].forward_relu(x2)));
        let x4 = self.block4[2]
            .forward_relu(self.block4[1].forward_relu(self.block4[0].forward_relu(x3.clone())));
        let x5 = self.block5[3]
            .forward_relu(self.block5[2].forward_relu(
                self.block5[1].forward_relu(self.block5[0].forward_relu(x4.clone())),
            ));
        let fused = x3 + self.upsample.x4.apply(x4)? + self.upsample.x5.apply(x5)?;
        let fused_h = self.block_fusion[1].forward_relu(self.block_fusion[0].forward_relu(fused));
        let fused_f32 = if self.precision == XFeatPrecision::F16 {
            fused_h.cast(FloatDType::F32)
        } else {
            fused_h
        };
        let descriptors = self.block_fusion[2].forward(fused_f32);
        let heat_f32 = self.heatmap_head[2].forward(
            self.heatmap_head[1]
                .forward_relu(self.heatmap_head[0].forward_relu(descriptors.clone())),
        );
        let logits = self.keypoint_head[3].forward(self.keypoint_head[2].forward_relu(
            self.keypoint_head[1].forward_relu(self.keypoint_head[0].forward_relu(patches)),
        ));
        Ok(DenseTensors {
            descriptors,
            logits,
            reliability: sigmoid(heat_f32),
        })
    }
}

/// Read back `N` registered tensors as finite host values.
async fn read_back<const N: usize>(transaction: Transaction) -> Result<[Vec<f32>; N], MatchError> {
    let data: Vec<TensorData> =
        transaction
            .execute_async()
            .await
            .map_err(|error| MatchError::Device {
                stage: "xfeat read-back",
                message: format!("{error:?}"),
            })?;
    let values: Vec<Vec<f32>> = data.iter().map(|d| d.iter::<f32>().collect()).collect();
    if values.iter().flatten().any(|v| !v.is_finite()) {
        return Err(MatchError::Output("non-finite dense XFeat output"));
    }
    <[_; N]>::try_from(values).map_err(|_| MatchError::Output("xfeat read-back count"))
}

#[cfg(test)]
pub(crate) mod tests;
