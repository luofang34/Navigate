//! Conversions of [`VirtualViewCamera`] to the `navigate-camera` model (ADR-0021).
//!
//! The virtual view projects with `u = fx·x/z + cx` and `v = fy·y/z + cy` in
//! camera axes `+X` right, `+Y` down, `+Z` forward, as
//! [`navigate_camera::Intrinsics::project`] does.

use navigate_camera::{CameraError, Intrinsics, OpticalPose};

use super::VirtualViewCamera;

impl VirtualViewCamera {
    /// The optical centre and the rotation from the local east, north, up
    /// axes into the camera axes.
    ///
    /// # Errors
    /// Refuses a `world_to_cam` that is not a proper rotation, and a value
    /// that is not finite.
    pub fn optical_pose(&self) -> Result<OpticalPose, CameraError> {
        OpticalPose::from_row_major(self.eye_enu, self.world_to_cam)
    }

    /// The pinhole intrinsics of the virtual image.
    #[must_use]
    pub const fn intrinsics(&self) -> Intrinsics {
        let [fx, fy, cx, cy] = self.intrinsics;
        let [width, height] = self.image_size;
        Intrinsics {
            width,
            height,
            fx,
            fy,
            cx,
            cy,
        }
    }
}

#[cfg(test)]
mod tests;
