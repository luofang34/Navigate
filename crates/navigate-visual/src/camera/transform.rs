//! Preserve optical rays when an image window changes.
use super::{CameraModel, VisualError};

impl CameraModel {
    /// Transform intrinsics for an integer crop followed by image resizing.
    ///
    /// `crop` is `[left, top, width, height]` in the calibrated source image.
    /// `size` is the output image size. Resizing maps pixel centres with
    /// `(source - crop_origin + 0.5) * scale - 0.5`. This does not estimate
    /// calibration, apply lens correction, rotate pixels, or change camera pose.
    ///
    /// # Errors
    /// Rejects an empty or out-of-bounds crop, invalid source calibration,
    /// and invalid output intrinsics or dimensions.
    pub fn crop_and_resize(&self, crop: [u32; 4], size: [u32; 2]) -> Result<Self, VisualError> {
        self.validate()?;
        let [left, top, width, height] = crop;
        if width == 0
            || height == 0
            || left.checked_add(width).is_none_or(|end| end > self.width)
            || top.checked_add(height).is_none_or(|end| end > self.height)
        {
            return Err(VisualError::Invalid {
                field: "camera image crop",
            });
        }
        let sx = f64::from(size[0]) / f64::from(width);
        let sy = f64::from(size[1]) / f64::from(height);
        let camera = Self {
            width: size[0],
            height: size[1],
            fx: self.fx * sx,
            fy: self.fy * sy,
            cx: (self.cx - f64::from(left) + 0.5) * sx - 0.5,
            cy: (self.cy - f64::from(top) + 0.5) * sy - 0.5,
        };
        camera.validate()?;
        Ok(camera)
    }
}
