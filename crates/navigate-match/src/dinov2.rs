//! DINOv2 ViT-S/14: a self-supervised image transformer whose tokens describe places.
//!
//! The model follows `transformers.Dinov2Model` with the `facebook/dinov2-small`
//! weights: a 14-pixel patch embedding, a class token, position embeddings resized
//! to the input grid, twelve pre-norm blocks with LayerScale, and a final layer norm.
//! It returns the normalized tokens, as the `last_hidden_state` of the reference.
mod block;
mod position;

use crate::{
    MatchError, Weights,
    loftr::layers::{LayerNorm, Linear, upload},
};
use block::Block;
use burn::tensor::{Device, Tensor};

/// Token width.
pub const DINOV2_WIDTH: usize = 384;
const PATCH: usize = 14;
const LAYERS: usize = 12;
const HEADS: usize = 6;
const EPSILON: f64 = 1e-6;

/// DINOv2 ViT-S/14 with weights on one burn device, for one square input size.
#[derive(Clone)]
pub struct Dinov2 {
    patch: Linear,
    class: Tensor<3>,
    position: Tensor<3>,
    blocks: Vec<Block>,
    norm: LayerNorm,
    side: usize,
    device: Device,
}

impl Dinov2 {
    /// Put the weights of `facebook/dinov2-small` on `device` for inputs `input_px`
    /// pixels square. The input side must be a multiple of 14.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is missing, has another shape, or is not used.
    pub fn load(
        mut weights: Weights,
        input_px: usize,
        device: &Device,
    ) -> Result<Self, MatchError> {
        if !input_px.is_multiple_of(PATCH) || input_px == 0 {
            return Err(MatchError::Output(
                "DINOv2 input side must be a multiple of 14",
            ));
        }
        let side = input_px / PATCH;
        let width = DINOV2_WIDTH;
        let projection = "embeddings.patch_embeddings.projection";
        let kernel = weights.take(&format!("{projection}.weight"), &[width, 3, PATCH, PATCH])?;
        let mut patch_weights = Weights::default();
        patch_weights.insert(
            "patch.weight",
            crate::WeightTensor {
                shape: vec![width, 3 * PATCH * PATCH],
                values: kernel,
            },
        );
        let bias = weights.take(&format!("{projection}.bias"), &[width])?;
        patch_weights.insert(
            "patch.bias",
            crate::WeightTensor {
                shape: vec![width],
                values: bias,
            },
        );
        let patch = Linear::load(
            &mut patch_weights,
            "patch",
            [width, 3 * PATCH * PATCH],
            true,
            device,
        )?;
        let class = upload(
            weights.take("embeddings.cls_token", &[1, 1, width])?,
            [1, 1, width],
            device,
        );
        weights.take("embeddings.mask_token", &[1, width])?;
        let table = weights.take("embeddings.position_embeddings", &[1, 1370, width])?;
        let resized = position::resize(&table, 37, side, width);
        let position = upload(resized, [1, 1 + side * side, width], device);
        let blocks = (0..LAYERS)
            .map(|i| {
                Block::load(
                    &mut weights,
                    &format!("encoder.layer.{i}"),
                    width,
                    HEADS,
                    EPSILON,
                    device,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let norm = LayerNorm::load_with_epsilon(&mut weights, "layernorm", width, EPSILON, device)?;
        weights.finish()?;
        Ok(Self {
            patch,
            class,
            position,
            blocks,
            norm,
            side,
            device: device.clone(),
        })
    }

    /// Input side in pixels.
    pub fn input_px(&self) -> usize {
        self.side * PATCH
    }

    /// The normalized tokens of `batch` images: the class token, then the patch tokens
    /// row by row, each [`DINOV2_WIDTH`] values. `images` holds normalized RGB, image
    /// by image and channel by channel, `input_px` pixels square.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong buffer length or a failed device read-back.
    pub async fn tokens(&self, images: &[f32], batch: usize) -> Result<Vec<f32>, MatchError> {
        let n = self.input_px();
        if images.len() != batch * 3 * n * n || batch == 0 {
            return Err(MatchError::Output(
                "DINOv2 image buffer length does not match its size",
            ));
        }
        let patches = unfold(images, batch, self.side);
        let x = self.patch.forward(upload(
            patches,
            [batch, self.side * self.side, 3 * PATCH * PATCH],
            &self.device,
        ));
        let class = self.class.clone().expand([batch, 1, DINOV2_WIDTH]);
        let mut x = Tensor::cat(vec![class, x], 1) + self.position.clone();
        for block in &self.blocks {
            x = block.forward(x);
        }
        let x = self.norm.forward(x);
        crate::loftr::read_floats(x, "DINOv2 read-back").await
    }
}

/// The 14 × 14 patches of each image as rows: channel, then row, then column.
fn unfold(images: &[f32], batch: usize, side: usize) -> Vec<f32> {
    let n = side * PATCH;
    let row = 3 * PATCH * PATCH;
    let mut out = vec![0.0; batch * side * side * row];
    for b in 0..batch {
        for py in 0..side {
            for px in 0..side {
                let base = (b * side * side + py * side + px) * row;
                for c in 0..3 {
                    for y in 0..PATCH {
                        let source = ((b * 3 + c) * n + py * PATCH + y) * n + px * PATCH;
                        let target = base + (c * PATCH + y) * PATCH;
                        out[target..target + PATCH]
                            .copy_from_slice(&images[source..source + PATCH]);
                    }
                }
            }
        }
    }
    out
}
