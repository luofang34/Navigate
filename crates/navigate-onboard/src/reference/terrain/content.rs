//! Whether a frame holds anything that a match can use.
//!
//! A frame of bias and noise, as a camera gives at night, has no structure
//! that the map can explain, and a match of it costs as much as a match of
//! a good frame. The check compares the contrast of the image at the scale
//! of a block of pixels with its contrast from one pixel to the next.
//! White noise averages over a block: its block contrast is its pixel
//! contrast over the side of the block. Ground texture is wider than a
//! block, so its block contrast is much more than that. The ratio does
//! not depend on the level of the noise.

use image::GrayImage;

/// Side of a block, pixels.
const BLOCK: u32 = 8;
/// Block contrast, in counts, below which a frame is flat whatever its
/// pixel contrast: the image has eight bits, and less than a quarter of a
/// count between blocks is below what the sensor resolves.
const FLAT_COUNTS: f64 = 0.25;

/// The contrast of one frame at two scales, in counts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Content {
    /// Mean absolute difference of the means of adjacent blocks.
    pub block: f64,
    /// Mean absolute difference of adjacent pixels in a row.
    pub pixel: f64,
}

impl Content {
    /// The contrast of `image`.
    pub(crate) fn of(image: &GrayImage) -> Self {
        let (width, height) = image.dimensions();
        let (columns, rows) = (width / BLOCK, height / BLOCK);
        let pixels = image.as_raw();
        let at = |x: u32, y: u32| {
            pixels
                .get(y as usize * width as usize + x as usize)
                .copied()
                .unwrap_or(0)
        };
        let mut means = vec![0.0_f64; columns as usize * rows as usize];
        let (mut pixel_sum, mut pixel_count) = (0_u64, 0_u64);
        for row in 0..rows {
            for column in 0..columns {
                let mut sum = 0_u32;
                for y in row * BLOCK..(row + 1) * BLOCK {
                    let mut left = at(column * BLOCK, y);
                    sum += u32::from(left);
                    for x in column * BLOCK + 1..(column + 1) * BLOCK {
                        let value = at(x, y);
                        sum += u32::from(value);
                        pixel_sum += u64::from(value.abs_diff(left));
                        left = value;
                    }
                }
                pixel_count += u64::from(BLOCK * (BLOCK - 1));
                if let Some(mean) = means.get_mut((row * columns + column) as usize) {
                    *mean = f64::from(sum) / f64::from(BLOCK * BLOCK);
                }
            }
        }
        let mean_at = |column: u32, row: u32| {
            means
                .get((row * columns + column) as usize)
                .copied()
                .unwrap_or(0.0)
        };
        let (mut block_sum, mut block_count) = (0.0_f64, 0_u32);
        for row in 0..rows {
            for column in 0..columns {
                if column + 1 < columns {
                    block_sum += (mean_at(column + 1, row) - mean_at(column, row)).abs();
                    block_count += 1;
                }
                if row + 1 < rows {
                    block_sum += (mean_at(column, row + 1) - mean_at(column, row)).abs();
                    block_count += 1;
                }
            }
        }
        Self {
            block: block_sum / f64::from(block_count.max(1)),
            pixel: pixel_sum as f64 / pixel_count.max(1) as f64,
        }
    }

    /// The block contrast as a multiple of the block contrast of white
    /// noise with the pixel contrast of the frame: one for noise alone.
    pub(crate) fn structure(&self) -> f64 {
        if self.pixel > 0.0 {
            self.block / (self.pixel / f64::from(BLOCK))
        } else {
            0.0
        }
    }

    /// Whether the block contrast of the frame is less than
    /// `min_block_contrast` counts.
    pub(crate) fn is_faint(&self, min_block_contrast: f64) -> bool {
        self.block < min_block_contrast
    }

    /// Whether the frame has no structure for a match: it is flat, or its
    /// block contrast is less than `min_structure` times that of noise.
    pub(crate) fn is_featureless(&self, min_structure: f64) -> bool {
        self.block < FLAT_COUNTS || self.structure() < min_structure
    }
}
