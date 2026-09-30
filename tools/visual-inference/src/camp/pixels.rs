use crate::InferenceError;
use image::RgbImage;

pub(super) fn prepare(image: &RgbImage, size: u32) -> Result<Vec<f32>, InferenceError> {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || width > 8192 || height > 8192 || size == 0 {
        return Err(InferenceError::Invalid(
            "invalid CAMP image dimensions".into(),
        ));
    }
    let plane = size as usize * size as usize;
    let mut data = vec![0.0; 3 * plane];
    for y in 0..size {
        let sy = (f64::from(y) + 0.5) * f64::from(height) / f64::from(size) - 0.5;
        for x in 0..size {
            let sx = (f64::from(x) + 0.5) * f64::from(width) / f64::from(size) - 0.5;
            let pixel = bilinear(image, sx, sy);
            for c in 0..3 {
                data[c * plane + (y * size + x) as usize] = pixel[c].round() as f32;
            }
        }
    }
    Ok(data)
}

fn bilinear(image: &RgbImage, x: f64, y: f64) -> [f64; 3] {
    let left = x.floor();
    let top = y.floor();
    let (fx, fy) = (x - left, y - top);
    let mut value = [0.0; 3];
    for dy in 0..2 {
        for dx in 0..2 {
            let ix = (left + f64::from(dx)).clamp(0.0, f64::from(image.width() - 1)) as u32;
            let iy = (top + f64::from(dy)).clamp(0.0, f64::from(image.height() - 1)) as u32;
            let weight = if dx == 0 { 1.0 - fx } else { fx } * if dy == 0 { 1.0 - fy } else { fy };
            for (c, v) in value.iter_mut().enumerate() {
                *v += f64::from(image.get_pixel(ix, iy)[c]) * weight;
            }
        }
    }
    value
}

#[cfg(test)]
mod tests;
