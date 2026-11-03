//! A tangent plane fixed to a rotating body.

use nalgebra::{Matrix3, Vector3};
use navigate_geodesy::LocalTangentPlane;

use super::GravityField;
use crate::body::PlanetaryBody;

/// A tangent plane on a rotating body.
///
/// The plane is a rigid Cartesian frame fixed to the body:
/// `p_bf = r_origin + Rᵀ·p_plane`, with `R` the rotation from the
/// body-fixed axes to the plane axes (north, east, down at the origin).
/// The plane has no flat-body approximation, because the gravity reads
/// the body-centered position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceFrame {
    plane: LocalTangentPlane,
    body: PlanetaryBody,
    origin_body_fixed: Vector3<f64>,
    body_fixed_to_plane: Matrix3<f64>,
}

impl SurfaceFrame {
    /// The frame of `plane` on `body`. The plane must be on the
    /// ellipsoid of `body`.
    #[must_use]
    pub fn new(plane: LocalTangentPlane, body: PlanetaryBody) -> Self {
        let origin = plane.origin_body_fixed();
        let rows = plane.body_fixed_to_ned_rows();
        Self {
            plane,
            body,
            origin_body_fixed: Vector3::new(origin.x_m, origin.y_m, origin.z_m),
            body_fixed_to_plane: Matrix3::from_fn(|i, j| rows[i][j]),
        }
    }

    /// The tangent plane of the frame.
    #[must_use]
    pub const fn plane(&self) -> &LocalTangentPlane {
        &self.plane
    }

    /// The body of the frame.
    #[must_use]
    pub const fn body(&self) -> &PlanetaryBody {
        &self.body
    }

    /// The body-centered body-fixed position in meters of the plane
    /// offset `p` in meters.
    #[must_use]
    pub fn body_fixed_position(&self, p: &Vector3<f64>) -> Vector3<f64> {
        self.origin_body_fixed + self.body_fixed_to_plane.transpose() * p
    }
}

impl GravityField for SurfaceFrame {
    fn gravity(&self, p: &Vector3<f64>) -> (Vector3<f64>, Matrix3<f64>) {
        let rotation = self.body_fixed_to_plane;
        let (g_bf, gradient_bf) = self
            .body
            .apparent_gravity_body_fixed(&self.body_fixed_position(p));
        (
            rotation * g_bf,
            rotation * gradient_bf * rotation.transpose(),
        )
    }

    fn frame_rotation(&self) -> Vector3<f64> {
        self.body_fixed_to_plane * self.body.rotation_body_fixed()
    }
}
