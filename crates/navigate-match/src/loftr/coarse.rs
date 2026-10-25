//! Dual-softmax coarse matching with border removal and mutual nearest neighbours.
//!
//! The device makes the confidence matrix and reduces it to three vectors. The
//! host then selects the matches. This keeps the read-back small: the matrix of a
//! 640 × 480 pair has 4800 × 4800 values, but each vector has 4800.
use crate::kernels::{Layout, product};
use burn::tensor::{Int, Tensor, activation::softmax};

/// Dual-softmax temperature of the trained model.
const TEMPERATURE: f64 = 0.1;
/// Minimum confidence of a coarse match.
pub(crate) const THRESHOLD: f32 = 0.2;
/// Coarse cells next to each image border that cannot match.
const BORDER: usize = 2;

/// Device reductions of the confidence matrix of one pair.
pub(crate) struct Reductions {
    /// Highest confidence in each row, `[L]`.
    pub row_max: Tensor<1>,
    /// Column of that confidence, `[L]`.
    pub row_argmax: Tensor<1, Int>,
    /// Highest confidence in each column, `[S]`.
    pub column_max: Tensor<1>,
}

/// Reduce the confidence matrix of `a` `[1, L, C]` against `b` `[1, S, C]`.
pub(crate) fn reduce(a: Tensor<3>, b: Tensor<3>) -> Reductions {
    let [_, length, channels] = a.dims();
    let [_, sources, _] = b.dims();
    let scale = (channels as f64).sqrt();
    let zeros = Tensor::<1>::zeros([sources], &a.device());
    let layout = Layout {
        trans_b: true,
        ..Layout::default()
    };
    let similarity = product(a / scale, b / scale, zeros, layout) / TEMPERATURE;
    let confidence = softmax(similarity.clone(), 1) * softmax(similarity, 2);
    let (row_max, row_argmax) = confidence.clone().max_dim_with_indices(2);
    let column_max = confidence.max_dim(1);
    Reductions {
        row_max: row_max.reshape([length]),
        row_argmax: row_argmax.reshape([length]),
        column_max: column_max.reshape([sources]),
    }
}

/// One coarse match: cell `i` of the first map, cell `j` of the second map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CoarseMatch {
    pub i: usize,
    pub j: usize,
    pub confidence: f32,
}

/// Select matches from read-back reductions, in ascending order of `i`.
///
/// `grids` are the `[height, width]` of the two coarse maps. A match needs a
/// confidence above [`THRESHOLD`], both cells away from the border, and the same
/// value as the maximum of its column.
pub(crate) fn select(
    row_max: &[f32],
    row_argmax: &[i64],
    column_max: &[f32],
    grids: [[usize; 2]; 2],
) -> Vec<CoarseMatch> {
    row_max
        .iter()
        .zip(row_argmax)
        .enumerate()
        .filter_map(|(i, (&confidence, &j))| {
            let j = usize::try_from(j).ok()?;
            let mutual = column_max.get(j).is_some_and(|&c| c == confidence);
            (confidence > THRESHOLD && mutual && inside(i, grids[0]) && inside(j, grids[1]))
                .then_some(CoarseMatch { i, j, confidence })
        })
        .collect()
}

fn inside(cell: usize, [height, width]: [usize; 2]) -> bool {
    let (row, column) = (cell / width.max(1), cell % width.max(1));
    row >= BORDER && column >= BORDER && row + BORDER < height && column + BORDER < width
}
