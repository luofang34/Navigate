//! LoFTR-DS: detector-free matching with the outdoor dual-softmax weights.
//!
//! The model follows `kornia.feature.LoFTR` 0.8.1 with its default configuration:
//! ResNet-FPN 8/2, four self and cross pairs of linear attention at 1/8 scale,
//! dual-softmax coarse matching, and one pair at 1/2 scale for refinement.
mod backbone;
mod coarse;
mod fine;
mod interpolate;
mod layers;
mod position;
mod transformer;

use crate::{MatchError, Weights};
use backbone::Backbone;
use burn::tensor::{Device, Tensor, Transaction};
use fine::Fine;
use layers::upload;
use transformer::Transformer;

const COARSE_WIDTH: usize = 256;
/// Image pixels per coarse cell.
const COARSE_SCALE: f32 = 8.0;
/// Image pixels per fine window cell, times the window radius.
const FINE_REACH: f32 = 4.0;

/// One correspondence in model-input pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Correspondence {
    /// Position in the first image.
    pub first: [f32; 2],
    /// Position in the second image.
    pub second: [f32; 2],
    /// Coarse dual-softmax confidence, above 0.2.
    pub confidence: f32,
}

/// LoFTR-DS with weights on one burn device.
#[derive(Clone)]
pub struct Loftr {
    backbone: Backbone,
    coarse: Transformer,
    fine: Fine,
    device: Device,
}

impl Loftr {
    /// Put the weights of [`crate::loftr_from_checkpoint`] on `device`.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is missing, has another shape, or is not used.
    pub fn load(mut weights: Weights, device: &Device) -> Result<Self, MatchError> {
        let model = Self {
            backbone: Backbone::load(&mut weights, device)?,
            coarse: Transformer::load(&mut weights, "loftr_coarse", COARSE_WIDTH, 4, device)?,
            fine: Fine::load(&mut weights, device)?,
            device: device.clone(),
        };
        weights.finish()?;
        Ok(model)
    }

    /// Match two grayscale images of the same size.
    ///
    /// Each image is row-major, `width × height` values in 0 through 1. Both sides
    /// must be multiples of 8 and at least 48. The result is in ascending order of
    /// the first-image cell, as the reference model gives it.
    ///
    /// # Errors
    ///
    /// Returns an error for a wrong image size or a failed device read-back.
    pub async fn correspondences(
        &self,
        first: Vec<f32>,
        second: Vec<f32>,
        [width, height]: [usize; 2],
    ) -> Result<Vec<Correspondence>, MatchError> {
        if width % 8 != 0 || height % 8 != 0 || width < 48 || height < 48 {
            return Err(MatchError::Output(
                "image sides must be multiples of 8 and at least 48",
            ));
        }
        if first.len() != width * height || second.len() != width * height {
            return Err(MatchError::Output(
                "image buffer length does not match its size",
            ));
        }
        let images = upload(
            [first, second].concat(),
            [2, 1, height, width],
            &self.device,
        );
        let features = self.backbone.forward(images);
        let grid = [height / 8, width / 8];
        let coarse = features.coarse + position::encoding(COARSE_WIDTH, grid, &self.device);
        let length = grid[0] * grid[1];
        let sequences = coarse.reshape([2, COARSE_WIDTH, length]).swap_dims(1, 2);
        let a = sequences.clone().narrow(0, 0, 1);
        let b = sequences.narrow(0, 1, 1);
        let (a, b) = self.coarse.forward(a, b);
        let matches = self.coarse_matches(a.clone(), b.clone(), grid).await?;
        if matches.is_empty() {
            return Ok(Vec::new());
        }
        let offsets = self.fine.offsets(features.fine, [a, b], &matches, grid[1]);
        let offsets = read_floats(offsets, "fine read-back").await?;
        if offsets.len() != 2 * matches.len() || offsets.iter().any(|v| !v.is_finite()) {
            return Err(MatchError::Output("fine offsets"));
        }
        Ok(matches
            .iter()
            .zip(offsets.as_chunks::<2>().0)
            .map(|(m, offset)| Correspondence {
                first: cell_position(m.i, grid[1]),
                second: {
                    let [x, y] = cell_position(m.j, grid[1]);
                    [x + offset[0] * FINE_REACH, y + offset[1] * FINE_REACH]
                },
                confidence: m.confidence,
            })
            .collect())
    }

    async fn coarse_matches(
        &self,
        a: Tensor<3>,
        b: Tensor<3>,
        grid: [usize; 2],
    ) -> Result<Vec<coarse::CoarseMatch>, MatchError> {
        let reductions = coarse::reduce(a, b);
        let data = Transaction::default()
            .register(reductions.row_max)
            .register(reductions.row_argmax)
            .register(reductions.column_max)
            .execute_async()
            .await
            .map_err(|error| device_error("coarse read-back", &error))?;
        let [row_max, row_argmax, column_max] =
            <[_; 3]>::try_from(data).map_err(|_| MatchError::Output("coarse read-back count"))?;
        let row_max: Vec<f32> = row_max.iter::<f32>().collect();
        let row_argmax: Vec<i64> = row_argmax.iter::<i64>().collect();
        let column_max: Vec<f32> = column_max.iter::<f32>().collect();
        Ok(coarse::select(
            &row_max,
            &row_argmax,
            &column_max,
            [grid; 2],
        ))
    }
}

/// Image position of the top-left corner of coarse cell `cell`, as the reference does.
fn cell_position(cell: usize, width: usize) -> [f32; 2] {
    [
        (cell % width) as f32 * COARSE_SCALE,
        (cell / width) as f32 * COARSE_SCALE,
    ]
}

async fn read_floats<const D: usize>(
    tensor: Tensor<D>,
    stage: &'static str,
) -> Result<Vec<f32>, MatchError> {
    let data = tensor
        .into_data_async()
        .await
        .map_err(|error| device_error(stage, &error))?;
    Ok(data.iter::<f32>().collect())
}

fn device_error(stage: &'static str, error: &dyn std::fmt::Debug) -> MatchError {
    MatchError::Device {
        stage,
        message: format!("{error:?}"),
    }
}

#[cfg(test)]
mod tests;
