//! Fixed-shape XFeat convolution outputs with float32 sparse decoding.
use crate::{InferenceError, features::Features, native_runtime::runtime, preprocessing};
use image::GrayImage;
use ort::{session::Session, value::Tensor};
const WIDTH: usize = 800;
const HEIGHT: usize = 576;
#[cfg(test)]
const CELLS: usize = WIDTH * HEIGHT / 64;
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

pub(crate) fn extract(
    session: &mut Session,
    image: &GrayImage,
    limit: usize,
) -> Result<Features, InferenceError> {
    let input = preprocessing::prepare(image, WIDTH, 600, 1)?;
    let normalized = normalize_resized(&input.data);
    let tensor = Tensor::from_array(([1, 1, HEIGHT, WIDTH], normalized))
        .map_err(|e| runtime("build dense XFeat input", e))?;
    let output = session
        .run(ort::inputs!["image"=>tensor])
        .map_err(|e| runtime("run dense XFeat", e))?;
    let get = |name: &str, channels: usize| -> Result<&[f32], InferenceError> {
        let (shape, data) = output
            .get(name)
            .ok_or_else(|| InferenceError::Invalid(format!("missing dense XFeat {name}")))?
            .try_extract_tensor::<f32>()
            .map_err(|e| runtime("read dense XFeat output", e))?;
        if shape.as_ref() != [1, channels as i64, (HEIGHT / 8) as i64, (WIDTH / 8) as i64]
            || data.iter().any(|v| !v.is_finite())
        {
            return Err(InferenceError::Invalid(format!(
                "invalid dense XFeat {name}"
            )));
        }
        Ok(data)
    };
    let descriptors = get("descriptors", 64)?;
    let logits = get("keypoint_logits", 65)?;
    let reliability = get("reliability", 1)?;
    decode(descriptors, logits, reliability, &input, image, limit)
}

fn normalize_resized(input: &[f32]) -> Vec<f32> {
    let mut resized = vec![0.0; WIDTH * HEIGHT];
    for y in 0..HEIGHT {
        let sy = (y as f64 + 0.5) * 600.0 / HEIGHT as f64 - 0.5;
        let top = sy.floor() as usize;
        let fraction = (sy - sy.floor()) as f32;
        for x in 0..WIDTH {
            resized[y * WIDTH + x] = input[top * WIDTH + x] * (1.0 - fraction)
                + input[(top + 1).min(599) * WIDTH + x] * fraction;
        }
    }
    let mean = resized.iter().map(|v| f64::from(*v)).sum::<f64>() / resized.len() as f64;
    let variance = resized
        .iter()
        .map(|v| (f64::from(*v) - mean).powi(2))
        .sum::<f64>()
        / resized.len() as f64;
    let scale = (variance + 1e-5).sqrt().recip();
    resized
        .iter_mut()
        .for_each(|v| *v = ((f64::from(*v) - mean) * scale) as f32);
    resized
}

pub(crate) fn decode(
    descriptors: &[f32],
    logits: &[f32],
    reliability: &[f32],
    input: &preprocessing::InputImage,
    image: &GrayImage,
    limit: usize,
) -> Result<Features, InferenceError> {
    decode_in_place(
        &mut descriptors.to_vec(),
        logits,
        reliability,
        input,
        image,
        limit,
    )
}

pub(crate) fn decode_in_place(
    descriptors: &mut [f32],
    logits: &[f32],
    reliability: &[f32],
    input: &preprocessing::InputImage,
    image: &GrayImage,
    limit: usize,
) -> Result<Features, InferenceError> {
    let grid = Grid {
        width: input.width / 32 * 32,
        height: input.height / 32 * 32,
    };
    if descriptors.len() != 64 * grid.cells()
        || logits.len() != 65 * grid.cells()
        || reliability.len() != grid.cells()
    {
        return Err(InferenceError::Invalid(
            "dense decoder tensor dimensions".into(),
        ));
    }
    let heatmap = heatmap(logits, grid);
    let mut candidates = peaks(&heatmap, reliability, grid);
    candidates.sort_by(|a, b| b.2.total_cmp(&a.2));
    for cell in 0..grid.cells() {
        let norm = (0..64)
            .map(|c| descriptors[c * grid.cells() + cell].powi(2))
            .sum::<f32>()
            .sqrt()
            .max(1e-12);
        for c in 0..64 {
            descriptors[c * grid.cells() + cell] /= norm;
        }
    }
    let mut result = Features::empty(64);
    for (x, y, score) in candidates.into_iter().take(4096) {
        let prepared = [
            x as f32,
            y as f32 * input.height as f32 / grid.height as f32,
        ];
        let pixel = input.pixel([f64::from(prepared[0]), f64::from(prepared[1])]);
        let model = [
            prepared[0] * (800.0 / input.width as f32),
            prepared[1] * (600.0 / input.height as f32),
        ];
        if !preprocessing::inside(pixel, image) {
            continue;
        }
        let stencil = Bicubic::new(grid_point(x, y, grid), grid);
        let descriptor: [f32; 64] = std::array::from_fn(|c| {
            stencil.sample(&descriptors[c * grid.cells()..(c + 1) * grid.cells()])
        });
        result.push(pixel, model, score, &descriptor)?;
        if result.pixels.len() == limit {
            break;
        }
    }
    Ok(result)
}

fn heatmap(logits: &[f32], grid: Grid) -> Vec<f32> {
    let mut heatmap = vec![0.0; grid.width * grid.height];
    for cell in 0..grid.cells() {
        let max = (0..65)
            .map(|c| logits[c * grid.cells() + cell])
            .fold(f32::NEG_INFINITY, f32::max);
        let weights: [f32; 65] =
            std::array::from_fn(|c| (logits[c * grid.cells() + cell] - max).exp());
        let sum: f32 = weights.iter().sum();
        for (c, weight) in weights.iter().enumerate().take(64) {
            let x = (cell % (grid.width / 8)) * 8 + c % 8;
            let y = (cell / (grid.width / 8)) * 8 + c / 8;
            heatmap[y * grid.width + x] = weight / sum;
        }
    }
    heatmap
}

fn peaks(heatmap: &[f32], reliability: &[f32], grid: Grid) -> Vec<(usize, usize, f32)> {
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
#[cfg(test)]
mod tests;
