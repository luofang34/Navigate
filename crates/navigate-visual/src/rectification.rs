//! Bounded image rectification into one shared pinhole pixel space.
mod error;
use crate::CameraModel;
pub use error::RectificationError;
use image::GrayImage;
use nalgebra::Vector2;

/// A reusable Brown-Conrady map from pinhole pixels to raw camera pixels.
///
/// Use the returned image and target camera together for matching, tracking and
/// geometry. No invalid border pixels are synthesized. Map storage is eight
/// bytes per output pixel; source and output image storage are separate.
/// Residuals in this pixel space are not raw-camera pixel residuals. Use
/// [`Self::raw_pixel`] when validation requires the source camera pixel metric.
pub struct RectificationMap {
    source: CameraModel,
    target: CameraModel,
    coefficients: [f64; 5],
    pixels: Vec<[f32; 2]>,
}
impl RectificationMap {
    /// Build a map with OpenCV-order coefficients `[k1, k2, p1, p2, k3]`.
    ///
    /// The target intrinsics explicitly select the crop and resolution. Both
    /// images use integer pixel centres and the same camera optical axes.
    ///
    /// # Errors
    /// Rejects invalid calibration, dimensions above 8192, excess map storage,
    /// folded mappings, or any target pixel outside the source image.
    pub fn new(
        source: CameraModel,
        target: CameraModel,
        coefficients: [f64; 5],
        max_map_bytes: usize,
    ) -> Result<Self, RectificationError> {
        source.validate()?;
        target.validate()?;
        if [source.width, source.height, target.width, target.height]
            .iter()
            .any(|&n| n > 8192)
        {
            return Err(RectificationError::Dimensions);
        }
        if !coefficients.iter().all(|v| v.is_finite()) {
            return Err(RectificationError::Coefficients);
        }
        let count = target.width as usize * target.height as usize;
        let required = count * std::mem::size_of::<[f32; 2]>();
        if required > max_map_bytes {
            return Err(RectificationError::Budget {
                required,
                limit: max_map_bytes,
            });
        }
        let mut result = Self {
            source,
            target,
            coefficients,
            pixels: Vec::new(),
        };
        result.pixels.try_reserve_exact(count)?;
        for y in 0..target.height {
            for x in 0..target.width {
                let pixel = Vector2::new(f64::from(x), f64::from(y));
                let raw = result
                    .raw_pixel(pixel)
                    .ok_or(RectificationError::Mapping { pixel: [x, y] })?;
                result.pixels.push([raw.x as f32, raw.y as f32]);
            }
        }
        Ok(result)
    }

    /// Camera calibration of every rectified output image.
    pub fn camera(&self) -> CameraModel {
        self.target
    }

    /// Logical bytes retained for pixel coordinates, excluding the images.
    pub fn map_bytes(&self) -> usize {
        self.pixels.len() * std::mem::size_of::<[f32; 2]>()
    }

    /// Map a rectified pixel to raw image coordinates without rounding.
    ///
    /// Returns `None` outside either image or at a locally folded mapping.
    pub fn raw_pixel(&self, pixel: Vector2<f64>) -> Option<Vector2<f64>> {
        if !inside(pixel, self.target) {
            return None;
        }
        let x = (pixel.x - self.target.cx) / self.target.fx;
        let y = (pixel.y - self.target.cy) / self.target.fy;
        let [k1, k2, p1, p2, k3] = self.coefficients;
        let r2 = x * x + y * y;
        let radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
        let derivative = k1 + r2 * (2.0 * k2 + 3.0 * r2 * k3);
        let a = radial + 2.0 * x * x * derivative + 2.0 * p1 * y + 6.0 * p2 * x;
        let b = 2.0 * x * y * derivative + 2.0 * p1 * x + 2.0 * p2 * y;
        let d = radial + 2.0 * y * y * derivative + 6.0 * p1 * y + 2.0 * p2 * x;
        if !(a > 0.0 && d > 0.0 && a * d - b * b > 0.0) {
            return None;
        }
        let raw = Vector2::new(
            self.source.fx * (x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x))
                + self.source.cx,
            self.source.fy * (y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y)
                + self.source.cy,
        );
        inside(raw, self.source).then_some(raw)
    }

    /// Apply the map with bilinear sampling and nearest-integer gray values.
    ///
    /// # Errors
    /// Rejects a source with dimensions different from the bound calibration,
    /// or an output allocation failure. Source pixels are never modified.
    pub fn rectify(&self, source: &GrayImage) -> Result<GrayImage, RectificationError> {
        let expected = (self.source.width, self.source.height);
        if source.dimensions() != expected {
            return Err(RectificationError::SourceSize {
                expected,
                received: source.dimensions(),
            });
        }
        let mut data = Vec::new();
        data.try_reserve_exact(self.pixels.len())?;
        for &[x, y] in &self.pixels {
            data.push(sample(source, x, y));
        }
        GrayImage::from_raw(self.target.width, self.target.height, data)
            .ok_or(RectificationError::OutputSize)
    }
}
fn inside(p: Vector2<f64>, camera: CameraModel) -> bool {
    p.iter().all(|v| v.is_finite())
        && p.x >= 0.0
        && p.y >= 0.0
        && p.x <= f64::from(camera.width - 1)
        && p.y <= f64::from(camera.height - 1)
}
fn sample(image: &GrayImage, x: f32, y: f32) -> u8 {
    let ix = x.floor() as u32;
    let iy = y.floor() as u32;
    let right = (ix + 1).min(image.width() - 1);
    let bottom = (iy + 1).min(image.height() - 1);
    let dx = x - ix as f32;
    let dy = y - iy as f32;
    let top = f32::from(image.get_pixel(ix, iy)[0]) * (1.0 - dx)
        + f32::from(image.get_pixel(right, iy)[0]) * dx;
    let bottom = f32::from(image.get_pixel(ix, bottom)[0]) * (1.0 - dx)
        + f32::from(image.get_pixel(right, bottom)[0]) * dx;
    (top * (1.0 - dy) + bottom * dy).round() as u8
}
#[cfg(test)]
mod tests;
