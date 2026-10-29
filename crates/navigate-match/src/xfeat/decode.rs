//! Host preprocessing and sparse peak/bicubic decoding for [`super::XFeat`].
use super::{DenseMaps, SparseFeatures, XFEAT_INPUT_HEIGHT, XFEAT_INPUT_WIDTH};
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
    let mut resized = vec![0.0; XFEAT_INPUT_WIDTH * XFEAT_INPUT_HEIGHT];
    for y in 0..XFEAT_INPUT_HEIGHT {
        let sy = (y as f64 + 0.5) * PREPARE_HEIGHT as f64 / XFEAT_INPUT_HEIGHT as f64 - 0.5;
        let top = sy.floor().max(0.0) as usize;
        let fraction = (sy - sy.floor()) as f32;
        for x in 0..XFEAT_INPUT_WIDTH {
            resized[y * XFEAT_INPUT_WIDTH + x] = prepared[top * XFEAT_INPUT_WIDTH + x]
                * (1.0 - fraction)
                + prepared[(top + 1).min(PREPARE_HEIGHT - 1) * XFEAT_INPUT_WIDTH + x] * fraction;
        }
    }
    let count = resized.len() as f64;
    let mean = resized.iter().map(|v| f64::from(*v)).sum::<f64>() / count;
    let variance = resized
        .iter()
        .map(|v| (f64::from(*v) - mean).powi(2))
        .sum::<f64>()
        / count;
    let scale = (variance + 1e-5).sqrt().recip();
    for value in &mut resized {
        *value = ((f64::from(*value) - mean) * scale) as f32;
    }
    resized
}

/// Decode dense maps into up to `limit` sparse keypoints and L2-normalized descriptors.
pub(super) fn decode(
    maps: &DenseMaps,
    prepared: &input::InputImage,
    image: &GrayImage,
    limit: usize,
) -> Result<SparseFeatures, MatchError> {
    let grid = Grid {
        width: XFEAT_INPUT_WIDTH,
        height: XFEAT_INPUT_HEIGHT,
    };
    let cells = grid.cells();
    if maps.descriptors.len() != 64 * cells
        || maps.keypoint_logits.len() != 65 * cells
        || maps.reliability.len() != cells
    {
        return Err(MatchError::Output("dense XFeat map dimensions mismatch"));
    }
    let mut descriptors = maps.descriptors.clone();
    for cell in 0..cells {
        let norm = (0..64)
            .map(|c| descriptors[c * cells + cell].powi(2))
            .sum::<f32>()
            .sqrt()
            .max(1e-12);
        for c in 0..64 {
            descriptors[c * cells + cell] /= norm;
        }
    }
    let heatmap = build_heatmap(&maps.keypoint_logits, grid);
    let mut candidates = find_peaks(&heatmap, &maps.reliability, grid);
    candidates.sort_by(|a, b| b.2.total_cmp(&a.2));
    let mut out = SparseFeatures::empty();
    for (x, y, score) in candidates.into_iter().take(4096) {
        let model = [
            x as f32,
            y as f32 * PREPARE_HEIGHT as f32 / grid.height as f32,
        ];
        let pixel = prepared.source([f64::from(model[0]), f64::from(model[1])]);
        if !input::inside(pixel, image) {
            continue;
        }
        let stencil = Bicubic::new(grid_point(x, y, grid), grid);
        let mut desc: [f32; 64] =
            std::array::from_fn(|c| stencil.sample(&descriptors[c * cells..(c + 1) * cells]));
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

fn build_heatmap(logits: &[f32], grid: Grid) -> Vec<f32> {
    let cells = grid.cells();
    let mut heatmap = vec![0.0; grid.width * grid.height];
    for cell in 0..cells {
        let max = (0..65)
            .map(|c| logits[c * cells + cell])
            .fold(f32::NEG_INFINITY, f32::max);
        let weights: [f32; 65] = std::array::from_fn(|c| (logits[c * cells + cell] - max).exp());
        let sum: f32 = weights.iter().sum();
        for (c, weight) in weights.iter().enumerate().take(64) {
            let x = (cell % (grid.width / 8)) * 8 + c % 8;
            let y = (cell / (grid.width / 8)) * 8 + c / 8;
            heatmap[y * grid.width + x] = weight / sum;
        }
    }
    heatmap
}

fn find_peaks(heatmap: &[f32], reliability: &[f32], grid: Grid) -> Vec<(usize, usize, f32)> {
    let mut peaks = Vec::new();
    for y in 0..grid.height {
        for x in 0..grid.width {
            let value = heatmap[y * grid.width + x];
            if value <= 0.05 || (x == 0 && y == 0) {
                continue;
            }
            let mut maximum = true;
            for dy in y.saturating_sub(2)..=(y + 2).min(grid.height - 1) {
                for dx in x.saturating_sub(2)..=(x + 2).min(grid.width - 1) {
                    maximum &= heatmap[dy * grid.width + dx] <= value;
                }
            }
            if maximum {
                let score = value * bilinear(reliability, grid_point(x, y, grid), grid);
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

    fn sample(&self, data: &[f32]) -> f32 {
        let mut result = 0.0;
        for (dy, wy) in self.y_weights.iter().enumerate() {
            for (dx, wx) in self.x_weights.iter().enumerate() {
                result += sample(
                    data,
                    self.x + dx as i32 - 1,
                    self.y + dy as i32 - 1,
                    self.grid,
                ) * wx
                    * wy;
            }
        }
        result
    }
}
