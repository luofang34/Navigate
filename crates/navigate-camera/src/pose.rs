//! The camera pose in the optical axes.

use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector2, Vector3};

use crate::{CameraError, Intrinsics};

/// The rotation from the eye axes of a GL camera (right, up, back) into the
/// optical axes (right, down, forward). It is its own inverse.
pub const OPTICAL_FROM_GL: [f64; 9] = [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0];

/// Largest deviation of `R·Rᵀ` from the identity that a rotation can have.
const ORTHONORMAL_TOLERANCE: f64 = 1e-6;

/// The pose of a camera in one world frame that the caller declares.
///
/// The camera axes are `+X` right, `+Y` down, and `+Z` forward.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OpticalPose {
    /// Optical centre in the world frame, in metres.
    pub eye: Vector3<f64>,
    /// Rotation from the world axes into the camera axes.
    pub world_to_cam: Rotation3<f64>,
}

impl OpticalPose {
    /// A pose from its parts.
    #[must_use]
    pub const fn new(eye: Vector3<f64>, world_to_cam: Rotation3<f64>) -> Self {
        Self { eye, world_to_cam }
    }

    /// A pose from an optical centre and a row-major rotation matrix, as
    /// `navigate_match::VirtualViewCamera` holds them.
    ///
    /// # Errors
    /// Refuses a matrix that is not a proper rotation, and a value that is
    /// not finite.
    pub fn from_row_major(eye: [f64; 3], world_to_cam: [f64; 9]) -> Result<Self, CameraError> {
        let m = Matrix3::from_row_slice(&world_to_cam);
        let eye = Vector3::from(eye);
        if !eye.iter().chain(m.iter()).all(|v| v.is_finite()) {
            return Err(CameraError::Invalid {
                field: "pose value",
            });
        }
        let orthonormal = (m * m.transpose() - Matrix3::identity()).amax() < ORTHONORMAL_TOLERANCE;
        if !orthonormal || m.determinant() <= 0.0 {
            return Err(CameraError::Invalid {
                field: "pose rotation",
            });
        }
        Ok(Self::new(eye, Rotation3::from_matrix_unchecked(m)))
    }

    /// The optical centre and the row-major rotation matrix.
    #[must_use]
    pub fn to_row_major(&self) -> ([f64; 3], [f64; 9]) {
        let m = self.world_to_cam.matrix();
        let mut rows = [0.0; 9];
        for (i, slot) in rows.iter_mut().enumerate() {
            *slot = m[(i / 3, i % 3)];
        }
        (self.eye.into(), rows)
    }

    /// A pose from a GL camera: the position and the rotation from the eye
    /// axes (right, up, back) into the world axes, as
    /// `navigate_visual::CameraPose` holds them.
    #[must_use]
    pub fn from_gl(position: Vector3<f64>, world_from_eye: UnitQuaternion<f64>) -> Self {
        let gl_to_optical = Self::optical_from_gl();
        let world_to_cam = gl_to_optical * world_from_eye.inverse().to_rotation_matrix();
        Self::new(position, world_to_cam)
    }

    /// The rotation from the GL eye axes into the world axes.
    #[must_use]
    pub fn world_from_gl_eye(&self) -> UnitQuaternion<f64> {
        let world_from_eye = self.world_to_cam.inverse() * Self::optical_from_gl();
        UnitQuaternion::from_rotation_matrix(&world_from_eye)
    }

    /// A world point in the camera axes.
    #[must_use]
    pub fn to_camera(&self, world: &Vector3<f64>) -> Vector3<f64> {
        self.world_to_cam * (world - self.eye)
    }

    /// The pixel of a world point. A point at or behind the camera gives
    /// `None`.
    #[must_use]
    pub fn project(&self, intrinsics: &Intrinsics, world: &Vector3<f64>) -> Option<Vector2<f64>> {
        intrinsics.project(&self.to_camera(world))
    }

    /// The unit ray of a pixel in the world axes.
    #[must_use]
    pub fn ray_world(&self, intrinsics: &Intrinsics, pixel: &Vector2<f64>) -> Vector3<f64> {
        self.world_to_cam.inverse() * intrinsics.ray(pixel)
    }

    /// The world point at depth `depth` along `+Z` through a pixel.
    #[must_use]
    pub fn unproject(
        &self,
        intrinsics: &Intrinsics,
        pixel: &Vector2<f64>,
        depth: f64,
    ) -> Vector3<f64> {
        let n = intrinsics.normalize(pixel);
        self.eye + self.world_to_cam.inverse() * Vector3::new(n.x * depth, n.y * depth, depth)
    }

    fn optical_from_gl() -> Rotation3<f64> {
        Rotation3::from_matrix_unchecked(Matrix3::from_row_slice(&OPTICAL_FROM_GL))
    }
}

#[cfg(test)]
mod tests;
