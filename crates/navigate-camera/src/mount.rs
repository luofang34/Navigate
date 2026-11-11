//! A camera on a vehicle body.

use nalgebra::{Rotation3, UnitQuaternion, Vector3};

use navigate_geodesy::float_math::{cos, sin};

use crate::OpticalPose;

/// The installation of a camera on a body with forward, right, and down
/// (FRD) axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraMount {
    /// Optical centre relative to the body reference point, in metres, in
    /// the body axes.
    pub lever_arm_body_m: Vector3<f64>,
    /// Rotation from the body axes into the camera axes (`+X` right, `+Y`
    /// down, `+Z` forward).
    pub cam_from_body: UnitQuaternion<f64>,
}

impl CameraMount {
    /// A camera at the body reference point that looks forward, with the
    /// image top up.
    #[must_use]
    pub fn forward() -> Self {
        // Camera +X = body right (+Y), +Y = body down (+Z), +Z = body forward (+X).
        Self::from_axes(Vector3::y(), Vector3::z(), Vector3::x())
    }

    /// A camera at the body reference point that looks down, with the
    /// image top toward the nose.
    #[must_use]
    pub fn nadir() -> Self {
        // Camera +X = body right, +Y = body aft (-X), +Z = body down.
        Self::from_axes(Vector3::y(), -Vector3::x(), Vector3::z())
    }

    /// A camera at the body reference point that looks forward and down by
    /// `depression_rad` below the body's forward axis, with the image right
    /// along the body's right. Zero is [`Self::forward`], and a right angle
    /// is [`Self::nadir`].
    #[must_use]
    pub fn depressed(depression_rad: f64) -> Self {
        let (sin, cos) = (sin(depression_rad), cos(depression_rad));
        // Camera +Z = forward and down; +Y = +Z × +X.
        Self::from_axes(
            Vector3::y(),
            Vector3::new(-sin, 0.0, cos),
            Vector3::new(cos, 0.0, sin),
        )
    }

    /// A camera at the body reference point: the forward camera of
    /// [`Self::forward`] turned as a rigid part by `yaw_rad` about the body
    /// down axis, then by `pitch_rad` about its own image-right axis, then
    /// by `roll_rad` about its own optical axis. This is the aerospace
    /// yaw, pitch, roll order. A positive yaw turns the optical axis toward
    /// the right wing. A positive pitch lifts the optical axis, so a pitch
    /// of a negative right angle is [`Self::nadir`] and a pitch of `-d` is
    /// [`Self::depressed`] with `d`. A positive roll lowers the right side
    /// of the image.
    #[must_use]
    pub fn rotated_forward(roll_rad: f64, pitch_rad: f64, yaw_rad: f64) -> Self {
        let (sr, cr) = (sin(roll_rad), cos(roll_rad));
        let (sp, cp) = (sin(pitch_rad), cos(pitch_rad));
        let (sy, cy) = (sin(yaw_rad), cos(yaw_rad));
        // The columns of Rz(yaw)·Ry(pitch)·Rx(roll): the body-axis
        // directions of the optical axis, the image right, and the image
        // down of the turned camera.
        let optical = Vector3::new(cy * cp, sy * cp, -sp);
        let right = Vector3::new(cy * sp * sr - sy * cr, sy * sp * sr + cy * cr, cp * sr);
        let down = Vector3::new(cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr);
        Self::from_axes(right, down, optical)
    }

    /// A camera at the body reference point that looks up, with the image
    /// top toward the nose.
    #[must_use]
    pub fn zenith() -> Self {
        // Camera +X = body left (-Y), +Y = body aft (-X), +Z = body up (-Z).
        Self::from_axes(-Vector3::y(), -Vector3::x(), -Vector3::z())
    }

    /// The same mount with a lever arm.
    #[must_use]
    pub fn with_lever_arm(mut self, lever_arm_body_m: Vector3<f64>) -> Self {
        self.lever_arm_body_m = lever_arm_body_m;
        self
    }

    /// The camera pose from the body position in the world frame and the
    /// rotation from the body axes into the world axes.
    #[must_use]
    pub fn camera_pose(
        &self,
        body_position: &Vector3<f64>,
        world_from_body: &UnitQuaternion<f64>,
    ) -> OpticalPose {
        let eye = body_position + world_from_body * self.lever_arm_body_m;
        let cam_from_world = self.cam_from_body * world_from_body.inverse();
        OpticalPose::new(eye, cam_from_world.to_rotation_matrix())
    }

    /// A direction in the camera axes, in the body axes.
    #[must_use]
    pub fn camera_to_body(&self, camera_direction: &Vector3<f64>) -> Vector3<f64> {
        self.cam_from_body.inverse() * camera_direction
    }

    /// A mount from the body-axis directions of the camera `+X`, `+Y`, and
    /// `+Z` axes.
    fn from_axes(x: Vector3<f64>, y: Vector3<f64>, z: Vector3<f64>) -> Self {
        let body_from_cam = Rotation3::from_basis_unchecked(&[x, y, z]);
        Self {
            lever_arm_body_m: Vector3::zeros(),
            cam_from_body: UnitQuaternion::from_rotation_matrix(&body_from_cam.inverse()),
        }
    }
}

#[cfg(test)]
mod tests;
