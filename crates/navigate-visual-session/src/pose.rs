//! Rigid camera poses and their conversion to `navigate_visual::CameraPose`.

use nalgebra::{Isometry3, Matrix3, Translation3, Vector3};
use navigate_visual::CameraPose;

/// World-from-camera rigid transform. Camera axes are right, up, and back.
pub type Pose = Isometry3<f64>;

/// Convert a visual camera pose into a rigid transform.
pub fn from_camera(pose: &CameraPose) -> Pose {
    Isometry3::from_parts(Translation3::from(pose.position), pose.orientation)
}

/// Convert a rigid transform into a visual camera pose.
pub fn to_camera(pose: &Pose) -> CameraPose {
    CameraPose {
        position: pose.translation.vector,
        orientation: pose.rotation,
    }
}

/// Skew-symmetric matrix of a vector.
pub(crate) fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

/// Interpolate two corrections. `t` is in `0..=1`.
pub(crate) fn blend(a: &Pose, b: &Pose, t: f64) -> Pose {
    let translation = a.translation.vector.lerp(&b.translation.vector, t);
    let rotation = a
        .rotation
        .try_slerp(&b.rotation, t, 1e-12)
        .unwrap_or(a.rotation);
    Isometry3::from_parts(Translation3::from(translation), rotation)
}

/// Angle between the camera view axis and straight down, in radians.
pub(crate) fn off_nadir(pose: &Pose) -> f64 {
    let view = pose.rotation * Vector3::new(0.0, 0.0, -1.0);
    (-view.z).clamp(-1.0, 1.0).acos()
}

/// Check that a pose has only finite values.
pub(crate) fn finite(pose: &Pose) -> bool {
    pose.translation.vector.iter().all(|x| x.is_finite())
        && pose.rotation.coords.iter().all(|x| x.is_finite())
}
