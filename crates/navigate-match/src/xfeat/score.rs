//! Device keypoint scoring for [`super::XFeat`] sparse extraction.
//!
//! The sparse decoder needs one value per model pixel: the keypoint heatmap at
//! each 5 x 5 local maximum above the detection threshold, times the reliability
//! sampled at that pixel. Computation on the device replaces the host softmax
//! over all logit channels and removes the logit and reliability read-back.
use super::{GRID_HEIGHT, GRID_WIDTH, upsample::Resample};
use crate::MatchError;
use burn::tensor::{Tensor, activation::softmax, module::max_pool2d, ops::MaxPoolOptions};

/// Smallest heatmap value that a keypoint candidate must exceed.
pub(super) const DETECTION_THRESHOLD: f32 = 0.05;
/// Side of the square non-maximum suppression window in model pixels.
const NMS_WINDOW: usize = 5;

/// Dense descriptors and per-pixel keypoint scores read back from one forward pass.
#[derive(Clone, Debug)]
pub(super) struct ScoredMaps {
    /// Coarse descriptor map `[1, 64, 72, 100]`.
    pub(super) descriptors: Vec<f32>,
    /// Keypoint score map `[1, 1, 576, 800]`; zero at every non-candidate pixel.
    pub(super) scores: Vec<f32>,
}

/// Convert `[1, 65, 72, 100]` logits and `[1, 1, 72, 100]` reliability to a
/// `[1, 1, 576, 800]` score map.
pub(super) fn score_map(
    logits: Tensor<4>,
    reliability: Tensor<4>,
    upsample: &Resample,
) -> Result<Tensor<4>, MatchError> {
    let (height, width) = (GRID_HEIGHT * 8, GRID_WIDTH * 8);
    // Channel `c = dy * 8 + dx` of cell `(cy, cx)` is model pixel
    // `(cy * 8 + dy, cx * 8 + dx)`; channel 64 is the "no keypoint" dustbin.
    let heatmap = softmax(logits, 1)
        .slice([0..1, 0..64, 0..GRID_HEIGHT, 0..GRID_WIDTH])
        .reshape([1, 8, 8, GRID_HEIGHT, GRID_WIDTH])
        .permute([0, 3, 1, 4, 2])
        .reshape([1, 1, height, width]);
    let pooled = max_pool2d(
        heatmap.clone(),
        MaxPoolOptions::new([NMS_WINDOW; 2])
            .with_stride([1, 1])
            .with_padding([NMS_WINDOW / 2; 2]),
    );
    let candidate = heatmap
        .clone()
        .equal(pooled)
        .bool_and(heatmap.clone().greater_elem(DETECTION_THRESHOLD));
    // A multiply, not a mask fill, keeps a non-finite heatmap or reliability
    // value visible to the read-back check.
    Ok(heatmap * upsample.apply(reliability)? * candidate.float())
}
