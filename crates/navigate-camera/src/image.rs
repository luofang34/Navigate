//! Borrowed row-strided pixels.

use crate::CameraError;

/// Bins of the histogram of [`ImageView::to_gray8_percentile`].
const HISTOGRAM_BINS: usize = 1024;

/// Row-strided pixel samples. A stride counts samples, not bytes, and
/// is at least the row length in samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Pixels<'a> {
    /// 8-bit gray, one sample per pixel.
    Gray8 {
        /// Samples, row after row.
        data: &'a [u8],
        /// Samples from the start of a row to the start of the next row.
        stride: usize,
    },
    /// 16-bit gray, one sample per pixel, such as raw counts of a thermal
    /// or star camera.
    Gray16 {
        /// Samples, row after row.
        data: &'a [u16],
        /// Samples from the start of a row to the start of the next row.
        stride: usize,
        /// Bits of each sample that the sensor uses, from 1 to 16.
        significant_bits: u8,
    },
    /// 8-bit red, green, blue, three samples per pixel.
    Rgb8 {
        /// Samples, row after row.
        data: &'a [u8],
        /// Samples from the start of a row to the start of the next row.
        stride: usize,
    },
}

/// An image that borrows its pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageView<'a> {
    width: u32,
    height: u32,
    pixels: Pixels<'a>,
}

impl<'a> ImageView<'a> {
    /// An image of `width` by `height` pixels.
    ///
    /// # Errors
    /// Refuses an empty image, a stride shorter than a row, a buffer
    /// shorter than the image, and `significant_bits` outside 1 to 16.
    pub fn new(width: u32, height: u32, pixels: Pixels<'a>) -> Result<Self, CameraError> {
        let view = Self {
            width,
            height,
            pixels,
        };
        view.validate()?;
        Ok(view)
    }

    /// Image width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Image height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// The pixel samples.
    #[must_use]
    pub const fn pixels(&self) -> Pixels<'a> {
        self.pixels
    }

    /// Checks the size, the stride, and the buffer length.
    ///
    /// # Errors
    /// As for [`Self::new`].
    pub fn validate(&self) -> Result<(), CameraError> {
        if self.width == 0 || self.height == 0 {
            return Err(CameraError::Invalid {
                field: "image size",
            });
        }
        let (len, stride, channels) = match self.pixels {
            Pixels::Gray8 { data, stride } => (data.len(), stride, 1),
            Pixels::Gray16 {
                data,
                stride,
                significant_bits,
            } => {
                if !(1..=16).contains(&significant_bits) {
                    return Err(CameraError::Invalid {
                        field: "significant bits",
                    });
                }
                (data.len(), stride, 1)
            }
            Pixels::Rgb8 { data, stride } => (data.len(), stride, 3),
        };
        let row = self.width as usize * channels;
        if stride < row {
            return Err(CameraError::Invalid {
                field: "row stride",
            });
        }
        let needed = stride * (self.height as usize - 1) + row;
        if len < needed {
            return Err(CameraError::BufferTooShort {
                needed,
                actual: len,
            });
        }
        Ok(())
    }

    /// The brightness of a pixel from 0 to 1. RGB uses the Rec. 709 luma
    /// weights. A pixel outside the image gives `None`.
    #[must_use]
    pub fn luma(&self, x: u32, y: u32) -> Option<f32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let (x, y) = (x as usize, y as usize);
        match self.pixels {
            Pixels::Gray8 { data, stride } => {
                data.get(y * stride + x).map(|v| f32::from(*v) / 255.0)
            }
            Pixels::Gray16 {
                data,
                stride,
                significant_bits,
            } => {
                let full = f32::from(u16::MAX >> (16 - significant_bits));
                data.get(y * stride + x).map(|v| f32::from(*v) / full)
            }
            Pixels::Rgb8 { data, stride } => {
                let start = y * stride + 3 * x;
                let px = data.get(start..start + 3)?;
                let [r, g, b] =
                    [px.first()?, px.get(1)?, px.get(2)?].map(|v| f32::from(*v) / 255.0);
                Some(0.2126 * r + 0.7152 * g + 0.0722 * b)
            }
        }
    }

    /// Writes the image as tightly packed 8-bit gray into `out`, with a
    /// linear stretch from the `low` percentile to the `high` percentile of
    /// the brightness. It is the automatic gain control for a matcher that
    /// accepts 8-bit gray only. The result is a function of the pixels only.
    ///
    /// # Errors
    /// Refuses `out` with a length other than `width * height`, and
    /// percentiles outside `0 ≤ low < high ≤ 1`.
    pub fn to_gray8_percentile(
        &self,
        low: f32,
        high: f32,
        out: &mut [u8],
    ) -> Result<(), CameraError> {
        let needed = self.width as usize * self.height as usize;
        if out.len() != needed {
            return Err(CameraError::OutputSize {
                needed,
                actual: out.len(),
            });
        }
        if !(0.0..1.0).contains(&low) || !(high > low && high <= 1.0) {
            return Err(CameraError::Invalid {
                field: "percentile",
            });
        }
        let mut histogram = [0_u32; HISTOGRAM_BINS];
        for y in 0..self.height {
            for x in 0..self.width {
                if let Some(count) = self.luma(x, y).and_then(|l| histogram.get_mut(bin_of(l))) {
                    *count = count.saturating_add(1);
                }
            }
        }
        let lo = percentile_bin(&histogram, needed, low);
        let hi = percentile_bin(&histogram, needed, high).max(lo + 1);
        let span = (hi - lo) as f32;
        for (i, slot) in out.iter_mut().enumerate() {
            let (x, y) = (
                (i % self.width as usize) as u32,
                (i / self.width as usize) as u32,
            );
            let value = self.luma(x, y).unwrap_or(0.0) * (HISTOGRAM_BINS - 1) as f32;
            let stretched = ((value - lo as f32) / span).clamp(0.0, 1.0);
            *slot = (stretched * 255.0 + 0.5) as u8;
        }
        Ok(())
    }
}

fn bin_of(luma: f32) -> usize {
    (luma.clamp(0.0, 1.0) * (HISTOGRAM_BINS - 1) as f32 + 0.5) as usize
}

/// The first bin at which the cumulative count reaches `fraction` of `total`.
fn percentile_bin(histogram: &[u32; HISTOGRAM_BINS], total: usize, fraction: f32) -> usize {
    let target = (fraction * total as f32) as u64;
    let mut cumulative = 0_u64;
    for (bin, count) in histogram.iter().enumerate() {
        cumulative += u64::from(*count);
        if cumulative > target {
            return bin;
        }
    }
    HISTOGRAM_BINS - 1
}

#[cfg(test)]
mod tests;
