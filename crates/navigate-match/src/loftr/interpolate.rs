//! Bilinear resampling with aligned corners, as `F.interpolate(align_corners=True)`.
//!
//! The resampling is two matrix products with fixed weights. This gives the same
//! result on every backend, without a dependency on how a backend defines corners.
use super::layers::upload;
use burn::tensor::Tensor;

/// Resample `[N, C, H, W]` to `[N, C, height, width]`.
pub(crate) fn upsample_align_corners(x: Tensor<4>, [height, width]: [usize; 2]) -> Tensor<4> {
    let [n, c, h, w] = x.dims();
    let device = x.device();
    let rows = upload(weights(h, height), [height, h], &device);
    let columns = upload(weights(w, width), [width, w], &device).transpose();
    let batch = n * c;
    let rows = rows.unsqueeze::<3>().expand([batch, height, h]);
    let columns = columns.unsqueeze::<3>().expand([batch, w, width]);
    rows.matmul(x.reshape([batch, h, w]))
        .matmul(columns)
        .reshape([n, c, height, width])
}

/// Row-major `[output, input]` weights of linear interpolation with aligned corners.
pub(crate) fn weights(input: usize, output: usize) -> Vec<f32> {
    let mut matrix = vec![0.0; output * input];
    let scale = if output > 1 {
        (input as f64 - 1.0) / (output as f64 - 1.0)
    } else {
        0.0
    };
    for row in 0..output {
        let source = row as f64 * scale;
        let low = (source.floor() as usize).min(input.saturating_sub(1));
        let high = (low + 1).min(input.saturating_sub(1));
        let fraction = (source - low as f64) as f32;
        matrix[row * input + low] += 1.0 - fraction;
        matrix[row * input + high] += fraction;
    }
    matrix
}
