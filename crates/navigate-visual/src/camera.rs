//! Calibrated camera geometry in a local east, north, up frame.

use nalgebra::{UnitQuaternion, Vector2, Vector3};

use crate::VisualError;

/// Pinhole intrinsics for an image with lens distortion removed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraModel {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Horizontal focal length in pixels.
    pub fx: f64,
    /// Vertical focal length in pixels.
    pub fy: f64,
    /// Principal point column. Pixel centres have integer coordinates.
    pub cx: f64,
    /// Principal point row. Rows increase down the image.
    pub cy: f64,
}

impl CameraModel {
    /// Check that the intrinsics define a usable camera.
    ///
    /// # Errors
    ///
    /// Rejects images smaller than 32 pixels per side, non-finite intrinsics,
    /// and non-positive focal lengths.
    pub fn validate(&self) -> Result<(), VisualError> {
        if self.width < 32
            || self.height < 32
            || ![self.fx, self.fy, self.cx, self.cy]
                .iter()
                .all(|x| x.is_finite())
            || self.fx <= 0.0
            || self.fy <= 0.0
        {
            return Err(VisualError::Invalid {
                field: "camera intrinsics",
            });
        }
        Ok(())
    }

    /// Project a local world point. Points behind the camera return `None`.
    pub fn project(&self, pose: &CameraPose, world: Vector3<f64>) -> Option<Vector2<f64>> {
        let eye = pose.orientation.inverse() * (world - pose.position);
        let depth = -eye.z;
        (depth > 0.01).then(|| {
            Vector2::new(
                self.fx * eye.x / depth + self.cx,
                self.cy - self.fy * eye.y / depth,
            )
        })
    }

    /// Recover a local world point from its positive optical-axis depth in metres.
    pub fn unproject(&self, pose: &CameraPose, pixel: Vector2<f64>, depth: f64) -> Vector3<f64> {
        let eye = Vector3::new(
            (pixel.x - self.cx) * depth / self.fx,
            (self.cy - pixel.y) * depth / self.fy,
            -depth,
        );
        pose.position + pose.orientation * eye
    }
}

/// Camera pose in one declared local east, north, up frame.
#[derive(Clone, Copy, Debug)]
pub struct CameraPose {
    /// Camera position in metres from the map anchor.
    pub position: Vector3<f64>,
    /// Eye-to-world rotation. Eye axes are right, up, and back.
    pub orientation: UnitQuaternion<f64>,
}

impl CameraPose {
    /// Check that the pose is finite.
    ///
    /// # Errors
    ///
    /// Rejects non-finite position or rotation values and non-unit quaternions.
    pub fn validate(&self) -> Result<(), VisualError> {
        if !self
            .position
            .iter()
            .chain(self.orientation.coords.iter())
            .all(|x| x.is_finite())
        {
            return Err(VisualError::Invalid {
                field: "camera pose",
            });
        }
        if (self.orientation.norm_squared() - 1.0).abs() > 1e-8 {
            return Err(VisualError::Invalid {
                field: "camera quaternion norm",
            });
        }
        Ok(())
    }

    pub(crate) fn increment(&self, delta: &nalgebra::SVector<f64, 6>) -> Self {
        Self {
            position: self.position + Vector3::new(delta[0], delta[1], delta[2]),
            orientation: self.orientation
                * UnitQuaternion::from_scaled_axis(
                    Vector3::new(delta[3], delta[4], delta[5]) * 0.001,
                ),
        }
    }
}

/// Search bounds around an externally supplied camera pose.
#[derive(Clone, Copy, Debug)]
pub struct PosePrior {
    /// Initial camera pose. This is not a visual measurement.
    pub pose: CameraPose,
    /// Maximum accepted translation from the initial pose, in metres.
    pub position_radius_m: f64,
    /// Maximum accepted angular change, in radians.
    pub attitude_radius_rad: f64,
}

impl PosePrior {
    /// Validate the pose and both search bounds.
    ///
    /// # Errors
    ///
    /// Rejects an invalid pose, non-positive or non-finite bounds, and attitude
    /// bounds greater than pi radians. Bounds do not guarantee matcher capture.
    pub fn validate(&self) -> Result<(), VisualError> {
        self.pose.validate()?;
        if !self.position_radius_m.is_finite()
            || self.position_radius_m <= 0.0
            || !self.attitude_radius_rad.is_finite()
            || self.attitude_radius_rad <= 0.0
            || self.attitude_radius_rad > std::f64::consts::PI
        {
            return Err(VisualError::Invalid {
                field: "prior bounds",
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
