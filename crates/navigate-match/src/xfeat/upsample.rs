//! Static interpolation matrices for [`super::XFeat`], uploaded once per model.
//!
//! The dense network has a fixed input size, so every resampling step has fixed
//! source and target shapes. Upload of the matrices at load time keeps the
//! per-frame forward pass free of host-to-device weight transfers.
use super::{GRID_HEIGHT, GRID_WIDTH};
use crate::{
    MatchError,
    kernels::{Layout, product},
    loftr::layers::upload,
};
use burn::tensor::{Device, Tensor};

/// Separable resampling `[rows] x input x [cols]^T` between two fixed shapes.
#[derive(Clone, Debug)]
pub(super) struct Resample {
    source: [usize; 2],
    target: [usize; 2],
    rows: Tensor<3>,
    cols: Tensor<3>,
    row_bias: Tensor<1>,
    col_bias: Tensor<1>,
}

impl Resample {
    fn new(
        source: [usize; 2],
        target: [usize; 2],
        rows: Vec<f32>,
        cols: Vec<f32>,
        device: &Device,
    ) -> Self {
        let [h, w] = source;
        let [height, width] = target;
        Self {
            source,
            target,
            rows: upload(rows, [1, height, h], device),
            cols: upload(cols, [1, width, w], device),
            row_bias: upload(vec![0.0; w], [w], device),
            col_bias: upload(vec![0.0; width], [width], device),
        }
    }

    /// Half-pixel bilinear resize with edge clamping (`align_corners = false`).
    fn half_pixel(source: [usize; 2], target: [usize; 2], device: &Device) -> Self {
        let rows = half_pixel_weights(source[0], target[0]);
        let cols = half_pixel_weights(source[1], target[1]);
        Self::new(source, target, rows, cols, device)
    }

    /// Resample `x` of shape `[n, c, source]` to `[n, c, target]`.
    pub(super) fn apply(&self, x: Tensor<4>) -> Result<Tensor<4>, MatchError> {
        let [n, c, h, w] = x.dims();
        if [h, w] != self.source {
            return Err(MatchError::Output("xfeat resample source shape mismatch"));
        }
        let [height, width] = self.target;
        let tall = product(
            self.rows.clone(),
            x.reshape([n * c, h, w]),
            self.row_bias.clone(),
            Layout::default(),
        );
        let layout = Layout {
            trans_b: true,
            ..Layout::default()
        };
        Ok(
            product(tall, self.cols.clone(), self.col_bias.clone(), layout)
                .reshape([n, c, height, width]),
        )
    }
}

/// Every fixed resampling step of one XFeat forward pass.
#[derive(Clone, Debug)]
pub(super) struct Upsample {
    /// Block 4 output to the coarse grid.
    pub(super) x4: Resample,
    /// Block 5 output to the coarse grid.
    pub(super) x5: Resample,
    /// Coarse reliability to model pixels with the decoder's sampling rule.
    pub(super) reliability: Resample,
}

impl Upsample {
    pub(super) fn new(device: &Device) -> Self {
        let grid = [GRID_HEIGHT, GRID_WIDTH];
        let pixels = [GRID_HEIGHT * 8, GRID_WIDTH * 8];
        Self {
            x4: Resample::half_pixel([GRID_HEIGHT / 2, GRID_WIDTH / 2], grid, device),
            x5: Resample::half_pixel([GRID_HEIGHT / 4, GRID_WIDTH / 4], grid, device),
            reliability: Resample::new(
                grid,
                pixels,
                grid_point_weights(GRID_HEIGHT, pixels[0]),
                grid_point_weights(GRID_WIDTH, pixels[1]),
                device,
            ),
        }
    }
}

pub(super) fn half_pixel_weights(input: usize, output: usize) -> Vec<f32> {
    let mut matrix = vec![0.0; output * input];
    if input == 0 || output == 0 {
        return matrix;
    }
    let last = input.saturating_sub(1);
    for row in 0..output {
        let source = (row as f64 + 0.5) * (input as f64 / output as f64) - 0.5;
        if source <= 0.0 {
            matrix[row * input] = 1.0;
        } else if source >= last as f64 {
            matrix[row * input + last] = 1.0;
        } else {
            let low = source.floor() as usize;
            let high = (low + 1).min(last);
            let fraction = (source - low as f64) as f32;
            matrix[row * input + low] += 1.0 - fraction;
            matrix[row * input + high] += fraction;
        }
    }
    matrix
}

/// Bilinear weights of `decode::grid_point` with zero padding outside the grid.
///
/// The sparse decoder samples reliability at `p / (pixels - 1) * cells - 0.5` and
/// reads zero outside the grid, so edge rows do not clamp like
/// [`half_pixel_weights`].
pub(super) fn grid_point_weights(cells: usize, pixels: usize) -> Vec<f32> {
    let mut matrix = vec![0.0; pixels * cells];
    if cells == 0 || pixels < 2 {
        return matrix;
    }
    for row in 0..pixels {
        let source = row as f32 / (pixels - 1) as f32 * cells as f32 - 0.5;
        let low = source.floor();
        let fraction = source - low;
        for (index, weight) in [(low, 1.0 - fraction), (low + 1.0, fraction)] {
            if index >= 0.0 && (index as usize) < cells {
                matrix[row * cells + index as usize] += weight;
            }
        }
    }
    matrix
}
