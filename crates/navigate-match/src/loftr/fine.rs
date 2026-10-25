//! Fine refinement: 5 × 5 windows around each coarse match, a small transformer,
//! and the expected position of the best window cell.
use super::{
    coarse::CoarseMatch,
    layers::{Linear, upload},
    transformer::Transformer,
};
use crate::kernels::{Layout, product};
use crate::{MatchError, Weights};
use burn::tensor::{Device, Int, Tensor, TensorData, activation::softmax, ops::PadMode};

/// Window side in fine cells.
pub(crate) const WINDOW: usize = 5;
const CELLS: usize = WINDOW * WINDOW;
const RADIUS: usize = WINDOW / 2;
/// Fine cells per coarse cell along each axis.
const STRIDE: usize = 4;
const COARSE_WIDTH: usize = 256;
const FINE_WIDTH: usize = 128;

pub(crate) struct Fine {
    down_proj: Linear,
    merge_feat: Linear,
    transformer: Transformer,
}

impl Fine {
    pub(crate) fn load(weights: &mut Weights, device: &Device) -> Result<Self, MatchError> {
        Ok(Self {
            down_proj: Linear::load(
                weights,
                "fine_preprocess.down_proj",
                [FINE_WIDTH, COARSE_WIDTH],
                true,
                device,
            )?,
            merge_feat: Linear::load(
                weights,
                "fine_preprocess.merge_feat",
                [FINE_WIDTH, 2 * FINE_WIDTH],
                true,
                device,
            )?,
            transformer: Transformer::load(weights, "loftr_fine", FINE_WIDTH, 1, device)?,
        })
    }

    /// Normalized window offsets `[M, 2]`, `(x, y)` in −1 through 1, of each match.
    ///
    /// `fine` is `[2, 128, H/2, W/2]` for both images. `coarse` holds the coarse
    /// transformer output of each image, `[1, L, 256]`.
    pub(crate) fn offsets(
        &self,
        fine: Tensor<4>,
        coarse: [Tensor<3>; 2],
        matches: &[CoarseMatch],
        coarse_width: usize,
    ) -> Tensor<2> {
        let device = fine.device();
        let [_, _, height, width] = fine.dims();
        let padded = fine.pad([(RADIUS, RADIUS), (RADIUS, RADIUS)], PadMode::Constant(0.0));
        let padded_width = width + 2 * RADIUS;
        let flat = padded.reshape([2, FINE_WIDTH, (height + 2 * RADIUS) * padded_width]);
        let cells =
            |pick: fn(&CoarseMatch) -> usize| -> Vec<usize> { matches.iter().map(pick).collect() };
        let (first, second) = (cells(|m| m.i), cells(|m| m.j));
        let window = |image: usize, ids: &[usize]| {
            let index = indices(&window_indices(ids, coarse_width, padded_width), &device);
            flat.clone()
                .narrow(0, image, 1)
                .select(2, index)
                .reshape([FINE_WIDTH, ids.len(), CELLS])
                .permute([1, 2, 0])
        };
        let windows = Tensor::cat(vec![window(0, &first), window(1, &second)], 0);
        let [coarse0, coarse1] = coarse;
        let centre = |features: Tensor<3>, ids: &[usize]| {
            let rows = coarse_rows(&features);
            features
                .reshape([rows, COARSE_WIDTH])
                .select(0, indices(ids, &device))
        };
        let context = self.down_proj.forward(Tensor::cat(
            vec![centre(coarse0, &first), centre(coarse1, &second)],
            0,
        ));
        let count = matches.len();
        let context = context
            .unsqueeze_dim::<3>(1)
            .expand([2 * count, CELLS, FINE_WIDTH]);
        let merged = self
            .merge_feat
            .forward(Tensor::cat(vec![windows, context], 2));
        let a = merged.clone().narrow(0, 0, count);
        let b = merged.narrow(0, count, count);
        let (a, b) = self.transformer.forward(a, b);
        expectation(a, b)
    }
}

fn coarse_rows(features: &Tensor<3>) -> usize {
    let [n, length, _] = features.dims();
    n * length
}

fn indices(values: &[usize], device: &Device) -> Tensor<1, Int> {
    let values: Vec<i64> = values.iter().map(|&v| v as i64).collect();
    let length = values.len();
    Tensor::from_data(TensorData::new(values, [length]), device)
}

/// Flat indices into the zero-padded fine map of each window cell, channel-free.
///
/// Coarse cell `(r, c)` covers fine rows `4r − 2` through `4r + 2`. With two cells
/// of padding these rows start at `4r` in the padded map. Cell order is row-major,
/// as `F.unfold` gives.
pub(crate) fn window_indices(
    cells: &[usize],
    coarse_width: usize,
    padded_width: usize,
) -> Vec<usize> {
    let mut out = Vec::with_capacity(cells.len() * CELLS);
    for &cell in cells {
        let (row, column) = (cell / coarse_width.max(1), cell % coarse_width.max(1));
        for u in 0..WINDOW {
            for v in 0..WINDOW {
                out.push((STRIDE * row + u) * padded_width + STRIDE * column + v);
            }
        }
    }
    out
}

/// Expected position of the centre feature of `a` in the window of `b`.
fn expectation(a: Tensor<3>, b: Tensor<3>) -> Tensor<2> {
    let [count, _, width] = b.dims();
    let device = b.device();
    let centre = a.narrow(1, CELLS / 2, 1);
    let layout = Layout {
        trans_b: true,
        ..Layout::default()
    };
    let zero = Tensor::<1>::zeros([1], &device);
    let similarity =
        product(b, centre, zero, layout).reshape([count, CELLS]) / (width as f64).sqrt();
    let heatmap = softmax(similarity, 1);
    let grid = upload(grid(), [CELLS, 2], &device);
    heatmap.matmul(grid)
}

/// Normalized `(x, y)` of each window cell, as `create_meshgrid(normalized=True)`.
pub(crate) fn grid() -> Vec<f32> {
    let step = 2.0 / (WINDOW as f32 - 1.0);
    let mut out = Vec::with_capacity(2 * CELLS);
    for u in 0..WINDOW {
        for v in 0..WINDOW {
            out.push(-1.0 + step * v as f32);
            out.push(-1.0 + step * u as f32);
        }
    }
    out
}
