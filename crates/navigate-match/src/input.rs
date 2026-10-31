//! Letterboxed model input with the pixel-centre transform back to the source image.
//!
//! The transform is the same as the ONNX adapters in `tools/visual-inference` use:
//! uniform scale to fit, centred, bilinear sampling of pixel centres, zero outside.
use crate::MatchError;
use image::GrayImage;

/// A model input and the transform from input pixels to source pixels.
#[derive(Clone, Debug)]
pub(crate) struct InputImage {
    pub values: Vec<f32>,
    scale: f64,
    offset: [f64; 2],
}

impl InputImage {
    /// Source-image position of an input-image position.
    pub(crate) fn source(&self, p: [f64; 2]) -> [f64; 2] {
        [
            (p[0] - self.offset[0]) / self.scale,
            (p[1] - self.offset[1]) / self.scale,
        ]
    }
}

/// Resample `image` into `width × height` model pixels with values in 0 through 1.
pub(crate) fn prepare(
    image: &GrayImage,
    [width, height]: [usize; 2],
) -> Result<InputImage, MatchError> {
    let (iw, ih) = image.dimensions();
    if !(16..=8192).contains(&iw) || !(16..=8192).contains(&ih) {
        return Err(MatchError::ImageSize {
            width: iw,
            height: ih,
        });
    }
    let scale = (width as f64 / f64::from(iw)).min(height as f64 / f64::from(ih));
    let x0 = (width as f64 - f64::from(iw) * scale) / 2.0;
    let y0 = (height as f64 - f64::from(ih) * scale) / 2.0;
    let mut values = vec![0.0; width * height];
    for py in 0..height {
        for px in 0..width {
            let sx = (px as f64 - x0 + 0.5) / scale - 0.5;
            let sy = (py as f64 - y0 + 0.5) / scale - 0.5;
            if sx < -0.5 || sy < -0.5 || sx >= f64::from(iw) - 0.5 || sy >= f64::from(ih) - 0.5 {
                continue;
            }
            values[py * width + px] = bilinear(image, sx, sy) as f32;
        }
    }
    Ok(InputImage {
        values,
        scale,
        offset: [x0 + (scale - 1.0) / 2.0, y0 + (scale - 1.0) / 2.0],
    })
}

fn bilinear(image: &GrayImage, sx: f64, sy: f64) -> f64 {
    let (iw, ih) = image.dimensions();
    let (left, top) = (sx.floor(), sy.floor());
    let (fx, fy) = (sx - left, sy - top);
    let mut value = 0.0;
    for dy in 0..2 {
        for dx in 0..2 {
            let ix = (left + f64::from(dx)).clamp(0.0, f64::from(iw - 1)) as u32;
            let iy = (top + f64::from(dy)).clamp(0.0, f64::from(ih - 1)) as u32;
            let wx = if dx == 0 { 1.0 - fx } else { fx };
            let wy = if dy == 0 { 1.0 - fy } else { fy };
            value += f64::from(image.get_pixel(ix, iy)[0]) * wx * wy / 255.0;
        }
    }
    value
}

/// True when `p` is finite and at least 4 pixels inside `image`.
pub(crate) fn inside(p: [f64; 2], image: &GrayImage) -> bool {
    p.iter().all(|v| v.is_finite())
        && p[0] >= 4.0
        && p[1] >= 4.0
        && p[0] < f64::from(image.width()) - 4.0
        && p[1] < f64::from(image.height()) - 4.0
}
