//! A camera on a vehicle body.

use nalgebra::{Rotation3, UnitQuaternion, Vector3};

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
