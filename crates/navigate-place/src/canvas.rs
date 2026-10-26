//! The model input: a north-up square of ground, and the tokens that a model gives.
//!
//! All encoder backends use [`Canvas::model_input`], so a query encoded in a browser
//! and an index encoded natively see the same pixels.

/// Side of a rendered canvas, pixels.
pub const CANVAS_PX: usize = 448;
/// Side of the patch grid that pooling uses; finer model grids are averaged to it.
pub const GRID: usize = 16;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];
/// Least share of a grid cell that must hold imagery for the cell to count.
const CELL_VALID: f32 = 0.5;

/// A north-up square of ground, [`CANVAS_PX`] pixels on a side: RGB from 0 to 255,
/// row by row, and which pixels hold imagery.
#[derive(Clone, Debug, PartialEq)]
pub struct Canvas {
    /// Colour of each pixel.
    pub rgb: Vec<[f32; 3]>,
    /// True where the pixel holds imagery.
    pub valid: Vec<bool>,
}

/// The tokens of one canvas: the class token, then [`GRID`] × [`GRID`] patch tokens
/// row by row, each `width` values.
#[derive(Clone, Debug, PartialEq)]
pub struct Tokens {
    /// Token width.
    pub width: usize,
    /// `(1 + GRID * GRID) * width` values.
    pub values: Vec<f32>,
}

impl Canvas {
    /// An empty canvas.
    pub fn empty() -> Self {
        Self {
            rgb: vec![[0.0; 3]; CANVAS_PX * CANVAS_PX],
            valid: vec![false; CANVAS_PX * CANVAS_PX],
        }
    }

    /// Share of the pixels that hold imagery.
    pub fn valid_share(&self) -> f64 {
        self.valid.iter().filter(|v| **v).count() as f64 / self.valid.len().max(1) as f64
    }

    /// The normalized model input, channel by channel, `size` pixels square. Each
    /// input pixel is the mean of the valid canvas pixels under it, or the dataset
    /// mean where most of them are empty.
    pub fn model_input(&self, size: usize) -> Vec<f32> {
        let step = (CANVAS_PX / size.max(1)).max(1);
        let mut data = vec![0.0; 3 * size * size];
        for y in 0..size {
            for x in 0..size {
                let (mut sum, mut count) = ([0.0_f32; 3], 0);
                for dy in 0..step {
                    for dx in 0..step {
                        let i = (y * step + dy) * CANVAS_PX + x * step + dx;
                        if self.valid.get(i).copied().unwrap_or(false) {
                            let p = self.rgb[i];
                            sum = [sum[0] + p[0], sum[1] + p[1], sum[2] + p[2]];
                            count += 1;
                        }
                    }
                }
                for c in 0..3 {
                    let value = if count * 2 >= step * step {
                        sum[c] / count as f32 / 255.0
                    } else {
                        MEAN[c]
                    };
                    data[c * size * size + y * size + x] = (value - MEAN[c]) / STD[c];
                }
            }
        }
        data
    }
}

/// True for each grid cell, row by row, where enough of the canvas holds imagery.
pub fn cell_weights(canvas: &Canvas) -> Vec<bool> {
    let cell = CANVAS_PX / GRID;
    (0..GRID * GRID)
        .map(|c| {
            let (row, column) = (c / GRID, c % GRID);
            let mut count = 0;
            for y in row * cell..(row + 1) * cell {
                for x in column * cell..(column + 1) * cell {
                    count += usize::from(canvas.valid[y * CANVAS_PX + x]);
                }
            }
            count as f32 >= CELL_VALID * (cell * cell) as f32
        })
        .collect()
}
