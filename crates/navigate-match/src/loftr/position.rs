//! Sine position encoding of the coarse feature map.
//!
//! The outdoor weights were trained with the original LoFTR encoding
//! (`temp_bug_fix = False`). In that form, `-log(10000) / d_model // 2` evaluates to
//! exactly `-1` because of operator precedence, so frequency `k` is `exp(-2k)`. The
//! weights depend on this form; do not replace it with the corrected one.
use super::layers::upload;
use burn::tensor::{Device, Tensor};

/// Encoding `[1, channels, height, width]` to add to a coarse feature map.
pub(crate) fn encoding(channels: usize, [height, width]: [usize; 2], device: &Device) -> Tensor<4> {
    upload(
        values(channels, [height, width]),
        [1, channels, height, width],
        device,
    )
}

/// Row-major `[channels, height, width]` encoding values.
///
/// Channel `4k` is `sin(x f_k)`, `4k + 1` is `cos(x f_k)`, `4k + 2` is `sin(y f_k)`,
/// and `4k + 3` is `cos(y f_k)`, with `x` and `y` counted from 1.
pub(crate) fn values(channels: usize, [height, width]: [usize; 2]) -> Vec<f32> {
    let plane = height * width;
    let mut out = vec![0.0_f32; channels * plane];
    for group in 0..channels / 4 {
        let frequency = (-2.0 * group as f32).exp();
        for row in 0..height {
            let y = (row + 1) as f32 * frequency;
            for column in 0..width {
                let x = (column + 1) as f32 * frequency;
                let at = row * width + column;
                out[(4 * group) * plane + at] = x.sin();
                out[(4 * group + 1) * plane + at] = x.cos();
                out[(4 * group + 2) * plane + at] = y.sin();
                out[(4 * group + 3) * plane + at] = y.cos();
            }
        }
    }
    out
}
