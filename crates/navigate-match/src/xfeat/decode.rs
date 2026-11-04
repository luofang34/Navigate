//! Host preprocessing and sparse peak/bicubic decoding for [`super::XFeat`].
use super::{SparseFeatures, XFEAT_INPUT_HEIGHT, XFEAT_INPUT_WIDTH, score::ScoredMaps};
use crate::{MatchError, input};
use image::GrayImage;

const PREPARE_HEIGHT: usize = 600;

#[derive(Clone, Copy)]
struct Grid {
    width: usize,
    height: usize,
}

impl Grid {
    fn cells(self) -> usize {
        self.width * self.height / 64
    }
}

/// Resize `800 × 600` grayscale values vertically to `800 × 576` and instance-normalize.
pub(super) fn normalize_resized(prepared: &[f32]) -> Vec<f32> {
    normalize_resized_dims(
        prepared,
        [XFEAT_INPUT_WIDTH, PREPARE_HEIGHT],
        [XFEAT_INPUT_WIDTH, XFEAT_INPUT_HEIGHT],
    )
}

/// Resize `prep_dims` (`[width, height]`) values to `in_dims` (`[width, height]`) and instance-normalize.
pub(super) fn normalize_resized_dims(
    prepared: &[f32],
    prep_dims: [usize; 2],
    in_dims: [usize; 2],
) -> Vec<f32> {
    if prep_dims == in_dims {
        return normalize_resized_owned(prepared.to_vec(), prep_dims, in_dims);
    }
    let [prep_w, prep_h] = prep_dims;
    let [in_w, in_h] = in_dims;
    let mut resized = if prep_w == in_w {
        let mut buf = vec![0.0; in_w * in_h];
        for y in 0..in_h {
            let sy = (y as f64 + 0.5) * prep_h as f64 / in_h as f64 - 0.5;
            let top = sy.floor().max(0.0) as usize;
            let fraction = (sy - sy.floor()) as f32;
            for x in 0..in_w {
                buf[y * in_w + x] = prepared[top * prep_w + x] * (1.0 - fraction)
                    + prepared[(top + 1).min(prep_h - 1) * prep_w + x] * fraction;
            }
        }
        buf
    } else {
        let mut buf = vec![0.0; in_w * in_h];
        for y in 0..in_h {
            let sy = ((y as f64 + 0.5) * prep_h as f64 / in_h as f64 - 0.5)
                .clamp(0.0, (prep_h.saturating_sub(1)) as f64);
            let y0 = sy.floor() as usize;
            let y1 = (y0 + 1).min(prep_h.saturating_sub(1));
            let fy = (sy - y0 as f64) as f32;
            for x in 0..in_w {
                let sx = ((x as f64 + 0.5) * prep_w as f64 / in_w as f64 - 0.5)
                    .clamp(0.0, (prep_w.saturating_sub(1)) as f64);
                let x0 = sx.floor() as usize;
                let x1 = (x0 + 1).min(prep_w.saturating_sub(1));
                let fx = (sx - x0 as f64) as f32;
                let v0 = prepared[y0 * prep_w + x0] * (1.0 - fx) + prepared[y0 * prep_w + x1] * fx;
                let v1 = prepared[y1 * prep_w + x0] * (1.0 - fx) + prepared[y1 * prep_w + x1] * fx;
                buf[y * in_w + x] = v0 * (1.0 - fy) + v1 * fy;
            }
        }
        buf
    };
    normalize_values_in_place(&mut resized);
    resized
}

/// Resize or reuse an owned `prepared` buffer into `in_dims` (`[width, height]`) and instance-normalize.
pub(super) fn normalize_resized_owned(
    mut prepared: Vec<f32>,
    prep_dims: [usize; 2],
    in_dims: [usize; 2],
) -> Vec<f32> {
    if prep_dims == in_dims && prepared.len() == in_dims[0] * in_dims[1] {
        normalize_values_in_place(&mut prepared);
        return prepared;
    }
    normalize_resized_dims(&prepared, prep_dims, in_dims)
}

fn normalize_values_in_place(resized: &mut [f32]) {
    let count = resized.len().max(1) as f64;
    let mean = resized.iter().map(|v| f64::from(*v)).sum::<f64>() / count;
    let variance = resized
        .iter()
        .map(|v| (f64::from(*v) - mean).powi(2))
        .sum::<f64>()
        / count;
    let scale = (variance + 1e-5).sqrt().recip();
    for value in resized {
        *value = ((f64::from(*value) - mean) * scale) as f32;
    }
}

/// Decode dense maps into up to `limit` sparse keypoints and L2-normalized descriptors.
pub(super) fn decode(
    maps: &ScoredMaps,
    prepared: &input::InputImage,
    image: &GrayImage,
    valid_mask: Option<&GrayImage>,
    limit: usize,
) -> Result<SparseFeatures, MatchError> {
    decode_grid(
        maps,
        prepared,
        [XFEAT_INPUT_WIDTH, PREPARE_HEIGHT],
        [XFEAT_INPUT_WIDTH, XFEAT_INPUT_HEIGHT],
        image,
        valid_mask,
        limit,
    )
}

/// Decode dense maps at arbitrary multiple-of-8 `in_dims` (`[width, height]`) and letterbox `prep_dims`.
pub(super) fn decode_grid(
    maps: &ScoredMaps,
    prepared: &input::InputImage,
    prep_dims: [usize; 2],
    in_dims: [usize; 2],
    image: &GrayImage,
    valid_mask: Option<&GrayImage>,
    limit: usize,
) -> Result<SparseFeatures, MatchError> {
    let grid = Grid {
        width: in_dims[0],
        height: in_dims[1],
    };
    if maps.scores.len() != grid.width * grid.height {
        return Err(MatchError::Output("dense XFeat map dimensions mismatch"));
    }
    decode_peaks_grid(
        &maps.descriptors,
        candidates(&maps.scores, grid),
        prepared,
        prep_dims,
        in_dims,
        image,
        valid_mask,
        limit,
    )
}

/// Decode sparse peak `(x, y, score)` tuples and raw `[1, 64, H/8, W/8]` descriptors without cloning dense maps.
#[expect(
    clippy::too_many_arguments,
    reason = "sparse peak decoder borrows raw descriptors, pre-extracted peaks, letterbox geometry, source image, and mask"
)]
pub(super) fn decode_peaks_grid(
    descriptors: &[f32],
    mut candidates: Vec<(usize, usize, f32)>,
    prepared: &input::InputImage,
    prep_dims: [usize; 2],
    in_dims: [usize; 2],
    image: &GrayImage,
    valid_mask: Option<&GrayImage>,
    limit: usize,
) -> Result<SparseFeatures, MatchError> {
    if in_dims[0] < 16
        || in_dims[1] < 16
        || !in_dims[0].is_multiple_of(8)
        || !in_dims[1].is_multiple_of(8)
    {
        return Err(MatchError::Output(
            "dense XFeat input dimensions must be multiples of 8",
        ));
    }
    let grid = Grid {
        width: in_dims[0],
        height: in_dims[1],
    };
    let cells = grid.cells();
    if descriptors.len() != 64 * cells {
        return Err(MatchError::Output("dense XFeat map dimensions mismatch"));
    }
    let mut inv_norm = vec![0.0_f32; cells];
    for c in 0..64 {
        let plane = &descriptors[c * cells..(c + 1) * cells];
        for (acc, &v) in inv_norm.iter_mut().zip(plane) {
            *acc += v * v;
        }
    }
    for acc in &mut inv_norm {
        *acc = acc.sqrt().max(1e-6).recip();
    }
    let sx = prep_dims[0] as f32 / grid.width as f32;
    let sy = prep_dims[1] as f32 / grid.height as f32;
    if let Some(mask) = valid_mask {
        candidates.retain(|&(x, y, _)| {
            let model = [x as f32 * sx, y as f32 * sy];
            let pixel = prepared.source([f64::from(model[0]), f64::from(model[1])]);
            input::inside(pixel, image) && mask_allows(pixel, mask)
        });
    }
    candidates.sort_by(|a, b| b.2.total_cmp(&a.2));
    let mut out = SparseFeatures::empty();
    for (x, y, score) in candidates.into_iter().take(4096) {
        let model = [x as f32 * sx, y as f32 * sy];
        let pixel = prepared.source([f64::from(model[0]), f64::from(model[1])]);
        if !input::inside(pixel, image) {
            continue;
        }
        let stencil = Bicubic::new(grid_point(x, y, grid), grid);
        let mut desc: [f32; 64] = std::array::from_fn(|c| {
            stencil.sample_normalized(&descriptors[c * cells..(c + 1) * cells], &inv_norm)
        });
        let norm = desc.iter().map(|v| v * v).sum::<f32>().sqrt();
        if !norm.is_finite() || norm <= 1e-6 {
            continue;
        }
        for value in &mut desc {
            *value /= norm;
        }
        out.pixels.push(pixel);
        out.model_pixels.push(model);
        out.scores.push(score);
        out.descriptors.extend_from_slice(&desc);
        if out.pixels.len() == limit {
            break;
        }
    }
    Ok(out)
}

pub(super) fn mask_allows(pixel: [f64; 2], mask: &GrayImage) -> bool {
    let (width, height) = mask.dimensions();
    if width == 0
        || height == 0
        || !pixel[0].is_finite()
        || !pixel[1].is_finite()
        || pixel[0] < 0.0
        || pixel[1] < 0.0
    {
        return false;
    }
    let fx = (pixel[0].floor() as u32).min(width - 1);
    let fy = (pixel[1].floor() as u32).min(height - 1);
    let rx = (pixel[0].round() as u32).min(width - 1);
    let ry = (pixel[1].round() as u32).min(height - 1);
    mask.get_pixel(fx, fy)[0] >= 128 && mask.get_pixel(rx, ry)[0] >= 128
}

/// Every positive device score in raster order, except at model pixel `(0, 0)`.
fn candidates(scores: &[f32], grid: Grid) -> Vec<(usize, usize, f32)> {
    scores
        .iter()
        .enumerate()
        .skip(1)
        .filter(|&(_, &score)| score > 0.0)
        .map(|(index, &score)| (index % grid.width, index / grid.width, score))
        .collect()
}

/// Host score map equivalent to [`super::score::score_map`] for `[1, 65, 72, 100]` logits.
#[cfg(test)]
pub(super) fn reference_scores(logits: &[f32], reliability: &[f32]) -> Vec<f32> {
    reference_scores_grid(logits, reliability, [XFEAT_INPUT_WIDTH, XFEAT_INPUT_HEIGHT])
}

/// Host score map for `[1, 65, H/8, W/8]` logits and `[1, 1, H/8, W/8]` reliability at `in_dims`.
#[cfg(test)]
pub(super) fn reference_scores_grid(
    logits: &[f32],
    reliability: &[f32],
    in_dims: [usize; 2],
) -> Vec<f32> {
    let grid = Grid {
        width: in_dims[0],
        height: in_dims[1],
    };
    let heatmap = build_heatmap(logits, grid);
    scores_from_heatmap_grid(&heatmap, reliability, in_dims)
}

/// Host NMS peak list directly from `[1, 65, H/8, W/8]` logits and optional `[1, 1, H/8, W/8]` reliability.
pub(super) fn peaks_from_logits_grid(
    logits: &[f32],
    reliability: Option<&[f32]>,
    in_dims: [usize; 2],
) -> Vec<(usize, usize, f32)> {
    let grid = Grid {
        width: in_dims[0],
        height: in_dims[1],
    };
    let heatmap = build_heatmap(logits, grid);
    find_peaks_opt(&heatmap, reliability, grid)
}

/// Host NMS peak list directly from a `[1, 1, H, W]` heatmap and optional `[1, 1, H/8, W/8]` reliability.
pub(super) fn peaks_from_heatmap_grid(
    heatmap: &[f32],
    reliability: Option<&[f32]>,
    in_dims: [usize; 2],
) -> Vec<(usize, usize, f32)> {
    let grid = Grid {
        width: in_dims[0],
        height: in_dims[1],
    };
    find_peaks_opt(heatmap, reliability, grid)
}

/// Host NMS peak score map for a `[1, 1, H, W]` heatmap and `[1, 1, H/8, W/8]` reliability.
#[cfg(test)]
pub(super) fn scores_from_heatmap_grid(
    heatmap: &[f32],
    reliability: &[f32],
    in_dims: [usize; 2],
) -> Vec<f32> {
    let grid = Grid {
        width: in_dims[0],
        height: in_dims[1],
    };
    let mut scores = vec![0.0; grid.width * grid.height];
    for (x, y, score) in find_peaks_opt(heatmap, Some(reliability), grid) {
        scores[y * grid.width + x] = score;
    }
    scores
}

fn build_heatmap(logits: &[f32], grid: Grid) -> Vec<f32> {
    let cells = grid.cells();
    let gw = grid.width / 8;
    let mut heatmap = vec![0.0; grid.width * grid.height];
    if cells == 0 {
        return heatmap;
    }
    let mut max_v = logits[..cells].to_vec();
    for c in 1..65 {
        for (m, &v) in max_v.iter_mut().zip(&logits[c * cells..(c + 1) * cells]) {
            if v > *m {
                *m = v;
            }
        }
    }
    let mut sum_exp = vec![0.0_f32; cells];
    for c in 0..64 {
        let cx = c % 8;
        let cy = c / 8;
        let plane = &logits[c * cells..(c + 1) * cells];
        for (cell, (&v, &m)) in plane.iter().zip(&max_v).enumerate() {
            let w = (v - m).exp();
            sum_exp[cell] += w;
            let x = (cell % gw) * 8 + cx;
            let y = (cell / gw) * 8 + cy;
            heatmap[y * grid.width + x] = w;
        }
    }
    for ((s, &v), &m) in sum_exp
        .iter_mut()
        .zip(&logits[64 * cells..65 * cells])
        .zip(&max_v)
    {
        *s += (v - m).exp();
    }
    for (cell, &s) in sum_exp.iter().enumerate() {
        let inv = s.recip();
        let bx = (cell % gw) * 8;
        let by = (cell / gw) * 8;
        for dy in 0..8 {
            let row = &mut heatmap[(by + dy) * grid.width + bx..(by + dy) * grid.width + bx + 8];
            for val in row {
                *val *= inv;
            }
        }
    }
    heatmap
}

fn find_peaks_opt(
    heatmap: &[f32],
    reliability: Option<&[f32]>,
    grid: Grid,
) -> Vec<(usize, usize, f32)> {
    let mut peaks = Vec::new();
    for y in 0..grid.height {
        for x in 0..grid.width {
            let value = heatmap[y * grid.width + x];
            if value <= super::score::DETECTION_THRESHOLD || (x == 0 && y == 0) {
                continue;
            }
            let mut maximum = true;
            'window: for dy in y.saturating_sub(2)..=(y + 2).min(grid.height - 1) {
                let row = &heatmap[dy * grid.width..];
                for &nbr in &row[x.saturating_sub(2)..=(x + 2).min(grid.width - 1)] {
                    if nbr > value {
                        maximum = false;
                        break 'window;
                    }
                }
            }
            if maximum {
                let score = match reliability {
                    Some(rel) => value * bilinear(rel, grid_point(x, y, grid), grid),
                    None => value,
                };
                if score > 0.0 {
                    peaks.push((x, y, score));
                }
            }
        }
    }
    peaks
}

fn grid_point(x: usize, y: usize, grid: Grid) -> [f32; 2] {
    [
        x as f32 / (grid.width - 1) as f32 * (grid.width / 8) as f32 - 0.5,
        y as f32 / (grid.height - 1) as f32 * (grid.height / 8) as f32 - 0.5,
    ]
}

fn sample(data: &[f32], x: i32, y: i32, grid: Grid) -> f32 {
    if x < 0 || y < 0 || x >= (grid.width / 8) as i32 || y >= (grid.height / 8) as i32 {
        0.0
    } else {
        data[y as usize * (grid.width / 8) + x as usize]
    }
}

fn bilinear(data: &[f32], p: [f32; 2], grid: Grid) -> f32 {
    let x = p[0].floor() as i32;
    let y = p[1].floor() as i32;
    let fx = p[0] - x as f32;
    let fy = p[1] - y as f32;
    sample(data, x, y, grid) * (1.0 - fx) * (1.0 - fy)
        + sample(data, x + 1, y, grid) * fx * (1.0 - fy)
        + sample(data, x, y + 1, grid) * (1.0 - fx) * fy
        + sample(data, x + 1, y + 1, grid) * fx * fy
}

fn cubic(t: f32) -> f32 {
    let x = t.abs();
    if x <= 1.0 {
        (1.25 * x - 2.25) * x * x + 1.0
    } else if x < 2.0 {
        ((-0.75 * x + 3.75) * x - 6.0) * x + 3.0
    } else {
        0.0
    }
}

struct Bicubic {
    grid: Grid,
    x: i32,
    y: i32,
    x_weights: [f32; 4],
    y_weights: [f32; 4],
}

impl Bicubic {
    fn new(point: [f32; 2], grid: Grid) -> Self {
        let x = point[0].floor() as i32;
        let y = point[1].floor() as i32;
        Self {
            grid,
            x,
            y,
            x_weights: std::array::from_fn(|i| cubic(point[0] - (x + i as i32 - 1) as f32)),
            y_weights: std::array::from_fn(|i| cubic(point[1] - (y + i as i32 - 1) as f32)),
        }
    }

    fn sample_normalized(&self, data: &[f32], inv_norm: &[f32]) -> f32 {
        let mut result = 0.0;
        let gw = (self.grid.width / 8) as i32;
        let gh = (self.grid.height / 8) as i32;
        for (dy, wy) in self.y_weights.iter().enumerate() {
            let cy = self.y + dy as i32 - 1;
            for (dx, wx) in self.x_weights.iter().enumerate() {
                let cx = self.x + dx as i32 - 1;
                let s = if cx < 0 || cy < 0 || cx >= gw || cy >= gh {
                    0.0
                } else {
                    let idx = cy as usize * gw as usize + cx as usize;
                    data[idx] * inv_norm[idx]
                };
                result += s * wx * wy;
            }
        }
        result
    }
}
