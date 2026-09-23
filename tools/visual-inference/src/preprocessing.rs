//! Pixel-centre transforms for fixed-shape model exports.
use crate::InferenceError;
use image::GrayImage;

pub(crate) struct InputImage {
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
    pub scale: f64,
    pub offset: [f64; 2],
}
impl InputImage {
    pub fn pixel(&self, p: [f64; 2]) -> [f64; 2] {
        [
            (p[0] - self.offset[0]) / self.scale,
            (p[1] - self.offset[1]) / self.scale,
        ]
    }
}
pub(crate) fn prepare(
    image: &GrayImage,
    width: usize,
    height: usize,
    channels: usize,
) -> Result<InputImage, InferenceError> {
    let (iw, ih) = image.dimensions();
    if !(16..=8192).contains(&iw) || !(16..=8192).contains(&ih) || ![1, 3].contains(&channels) {
        return Err(InferenceError::Invalid(
            "unsupported image dimensions or channels".into(),
        ));
    }
    let scale = (width as f64 / f64::from(iw)).min(height as f64 / f64::from(ih));
    let x = (width as f64 - f64::from(iw) * scale) / 2.0;
    let y = (height as f64 - f64::from(ih) * scale) / 2.0;
    let mut data = vec![0.0; width * height * channels];
    for py in 0..height {
        for px in 0..width {
            let sx = (px as f64 - x + 0.5) / scale - 0.5;
            let sy = (py as f64 - y + 0.5) / scale - 0.5;
            if sx < -0.5 || sy < -0.5 || sx >= f64::from(iw) - 0.5 || sy >= f64::from(ih) - 0.5 {
                continue;
            }
            let left = sx.floor();
            let top = sy.floor();
            let fx = sx - left;
            let fy = sy - top;
            let mut value = 0.0;
            for dy in 0..2 {
                for dx in 0..2 {
                    let ix = (left + f64::from(dx)).clamp(0.0, f64::from(iw - 1)) as u32;
                    let iy = (top + f64::from(dy)).clamp(0.0, f64::from(ih - 1)) as u32;
                    value += f64::from(image.get_pixel(ix, iy)[0])
                        * if dx == 0 { 1.0 - fx } else { fx }
                        * if dy == 0 { 1.0 - fy } else { fy }
                        / 255.0;
                }
            }
            for channel in 0..channels {
                data[channel * width * height + py * width + px] = value as f32;
            }
        }
    }
    Ok(InputImage {
        data,
        width,
        height,
        scale,
        offset: [x + (scale - 1.0) / 2.0, y + (scale - 1.0) / 2.0],
    })
}

pub(crate) fn inside(p: [f64; 2], image: &GrayImage) -> bool {
    p.iter().all(|v| v.is_finite())
        && p[0] >= 4.0
        && p[1] >= 4.0
        && p[0] < f64::from(image.width()) - 4.0
        && p[1] < f64::from(image.height()) - 4.0
}
#[cfg(test)]
mod tests;
