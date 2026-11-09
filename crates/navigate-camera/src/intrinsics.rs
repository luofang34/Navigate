//! Pinhole intrinsics.

use nalgebra::{Matrix3, Vector2, Vector3};

use crate::CameraError;

/// The smallest depth along `+Z` that projects, in the units of the point.
const MIN_DEPTH: f64 = 1e-9;

/// A pinhole camera for an image without lens distortion.
///
/// Pixel centres have integer coordinates: the centre of the top-left pixel
/// is `(0, 0)`, and the centre of an 800 by 600 image is `(399.5, 299.5)`.
/// Columns increase to the right and rows increase down the image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Intrinsics {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Horizontal focal length in pixels.
    pub fx: f64,
    /// Vertical focal length in pixels.
    pub fy: f64,
    /// Principal point column in pixels.
    pub cx: f64,
    /// Principal point row in pixels.
    pub cy: f64,
}

impl Intrinsics {
    /// Intrinsics with the principal point at the image centre and a
    /// horizontal field of view of `horizontal_fov_rad`. The pixels are
    /// square.
    ///
    /// # Errors
    /// Refuses an empty image and a field of view outside `(0, π)`.
    pub fn centred(width: u32, height: u32, horizontal_fov_rad: f64) -> Result<Self, CameraError> {
        if !(horizontal_fov_rad > 0.0 && horizontal_fov_rad < core::f64::consts::PI) {
            return Err(CameraError::Invalid {
                field: "horizontal field of view",
            });
        }
        let half = navigate_geodesy::float_math::tan(horizontal_fov_rad * 0.5);
        let f = f64::from(width) * 0.5 / half;
        let model = Self {
            width,
            height,
            fx: f,
            fy: f,
            cx: (f64::from(width) - 1.0) * 0.5,
            cy: (f64::from(height) - 1.0) * 0.5,
        };
        model.validate()?;
        Ok(model)
    }

    /// Checks that the intrinsics define a usable camera.
    ///
    /// # Errors
    /// Refuses an empty image, a value that is not finite, and a focal
    /// length that is not positive.
    pub fn validate(&self) -> Result<(), CameraError> {
        if self.width == 0 || self.height == 0 {
            return Err(CameraError::Invalid {
                field: "image size",
            });
        }
        let finite = [self.fx, self.fy, self.cx, self.cy]
            .iter()
            .all(|v| v.is_finite());
        if !finite || self.fx <= 0.0 || self.fy <= 0.0 {
            return Err(CameraError::Invalid {
                field: "focal length or principal point",
            });
        }
        Ok(())
    }

    /// The calibration matrix `K`.
    #[must_use]
    pub fn matrix(&self) -> Matrix3<f64> {
        Matrix3::new(self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0)
    }

    /// The pixel of a point in the camera axes. A point at or behind the
    /// optical centre gives `None`.
    #[must_use]
    pub fn project(&self, camera_point: &Vector3<f64>) -> Option<Vector2<f64>> {
        (camera_point.z > MIN_DEPTH).then(|| {
            Vector2::new(
                self.fx * camera_point.x / camera_point.z + self.cx,
                self.fy * camera_point.y / camera_point.z + self.cy,
            )
        })
    }

    /// The normalized image coordinates `(x/z, y/z)` of a pixel.
    #[must_use]
    pub fn normalize(&self, pixel: &Vector2<f64>) -> Vector2<f64> {
        Vector2::new((pixel.x - self.cx) / self.fx, (pixel.y - self.cy) / self.fy)
    }

    /// The pixel of normalized image coordinates.
    #[must_use]
    pub fn denormalize(&self, normalized: &Vector2<f64>) -> Vector2<f64> {
        Vector2::new(
            normalized.x * self.fx + self.cx,
            normalized.y * self.fy + self.cy,
        )
    }

    /// The unit ray of a pixel in the camera axes.
    #[must_use]
    pub fn ray(&self, pixel: &Vector2<f64>) -> Vector3<f64> {
        let n = self.normalize(pixel);
        Vector3::new(n.x, n.y, 1.0).normalize()
    }

    /// Whether a pixel centre lies inside the image area, with the pixel
    /// edges at `-0.5` and `size - 0.5`.
    #[must_use]
    pub fn contains(&self, pixel: &Vector2<f64>) -> bool {
        pixel.x >= -0.5
            && pixel.y >= -0.5
            && pixel.x < f64::from(self.width) - 0.5
            && pixel.y < f64::from(self.height) - 0.5
    }

    /// Intrinsics for the same rays after a resize to `size`.
    ///
    /// Pixel centres map with `(source + 0.5) * scale - 0.5`.
    ///
    /// # Errors
    /// Refuses an empty output size and invalid source intrinsics.
    pub fn resized(&self, size: [u32; 2]) -> Result<Self, CameraError> {
        self.validate()?;
        let [width, height] = size;
        let sx = f64::from(width) / f64::from(self.width);
        let sy = f64::from(height) / f64::from(self.height);
        let model = Self {
            width,
            height,
            fx: self.fx * sx,
            fy: self.fy * sy,
            cx: (self.cx + 0.5) * sx - 0.5,
            cy: (self.cy + 0.5) * sy - 0.5,
        };
        model.validate()?;
        Ok(model)
    }
}

#[cfg(test)]
mod tests;
