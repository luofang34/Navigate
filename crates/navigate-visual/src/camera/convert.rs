//! Lossless conversions to and from the `navigate-camera` model (ADR-0021).
//!
//! [`CameraModel`] and [`navigate_camera::Intrinsics`] use the same pixel
//! convention, so the fields copy. [`CameraPose`] holds the GL eye axes
//! (right, up, back). [`navigate_camera::OpticalPose`] holds the optical
//! axes (right, down, forward). The conversion turns the eye axes half a
//! turn about `+X`.

use navigate_camera::{Intrinsics, OpticalPose};

use super::{CameraModel, CameraPose};

impl From<CameraModel> for Intrinsics {
    fn from(camera: CameraModel) -> Self {
        Self {
            width: camera.width,
            height: camera.height,
            fx: camera.fx,
            fy: camera.fy,
            cx: camera.cx,
            cy: camera.cy,
        }
    }
}

impl From<Intrinsics> for CameraModel {
    fn from(intrinsics: Intrinsics) -> Self {
        Self {
            width: intrinsics.width,
            height: intrinsics.height,
            fx: intrinsics.fx,
            fy: intrinsics.fy,
            cx: intrinsics.cx,
            cy: intrinsics.cy,
        }
    }
}

impl From<CameraPose> for OpticalPose {
    fn from(pose: CameraPose) -> Self {
        Self::from_gl(pose.position, pose.orientation)
    }
}

impl From<OpticalPose> for CameraPose {
    fn from(pose: OpticalPose) -> Self {
        Self {
            position: pose.eye,
            orientation: pose.world_from_gl_eye(),
        }
    }
}
