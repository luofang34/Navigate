//! Gravity and frame rotation of the inertial navigation regimes
//! (ADR-0014).
//!
//! [`PlanetaryBody`] gives the apparent gravity on a rotating body: the
//! point mass, the `J2` zonal harmonic, and the centrifugal term. The
//! inertial filter evaluates it at the body-centered position of the
//! vehicle, never at the tangent-plane offset alone.
//! [`SpaceEnvironment`] gives the gravity in a Cartesian frame centered
//! on one body, with optional third bodies and an optional frame
//! rotation. Each model also gives its exact gravity gradient, which the
//! error-state dynamics use.

use nalgebra::{Matrix3, Vector3};
use navigate_geodesy::{BodyEllipsoid, LocalTangentPlane};

mod space;

pub use space::{CentralBodyGravity, PointMassBody, SpaceEnvironment, SpaceGravityModel};

/// Largest magnitude of `J2` that [`PlanetaryBody::is_plausible`] and
/// [`CentralBodyGravity::is_plausible`] accept.
const MAX_ABS_J2: f64 = 0.1;

/// A rotating body with a reference ellipsoid, a gravitational
/// parameter, a `J2` zonal harmonic, and a rotation rate about the
/// body-fixed `z` axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanetaryBody {
    /// The reference ellipsoid. Its frame is the body-fixed frame.
    pub ellipsoid: BodyEllipsoid,
    /// Gravitational parameter `μ` in m³/s².
    pub gm_m3_per_s2: f64,
    /// Unnormalized `J2` zonal harmonic. It has no unit.
    pub j2: f64,
    /// Rotation rate about the body-fixed `z` axis in rad/s.
    pub rotation_rate_rad_per_s: f64,
}

impl PlanetaryBody {
    /// The Earth on WGS84 with the WGS84 gravity constants.
    #[must_use]
    pub const fn earth_wgs84() -> Self {
        Self {
            ellipsoid: BodyEllipsoid::WGS84,
            gm_m3_per_s2: 3.986_004_418e14,
            j2: 1.082_626_68e-3,
            rotation_rate_rad_per_s: 7.292_115e-5,
        }
    }

    /// The Moon in the mean Earth frame.
    #[must_use]
    pub const fn moon_me() -> Self {
        Self {
            ellipsoid: BodyEllipsoid::MOON_ME,
            ..Self::moon_pa()
        }
    }

    /// The Moon in the principal axis frame.
    #[must_use]
    pub const fn moon_pa() -> Self {
        Self {
            ellipsoid: BodyEllipsoid::MOON_PA,
            gm_m3_per_s2: 4.904_869_5e12,
            j2: 2.032_3e-4,
            rotation_rate_rad_per_s: 2.661_699_5e-6,
        }
    }

    /// Mars in the IAU body-fixed frame.
    #[must_use]
    pub const fn mars_iau() -> Self {
        Self {
            ellipsoid: BodyEllipsoid::MARS_IAU,
            gm_m3_per_s2: 4.282_837_2e13,
            j2: 1.960_45e-3,
            rotation_rate_rad_per_s: 7.088_218_1e-5,
        }
    }

    /// Whether `μ` is finite and positive, `J2` is finite and small, and
    /// the rotation rate is finite.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        self.gm_m3_per_s2.is_finite()
            && self.gm_m3_per_s2 > 0.0
            && self.j2.is_finite()
            && self.j2.abs() < MAX_ABS_J2
            && self.rotation_rate_rad_per_s.is_finite()
    }

    /// The point mass and the `J2` term about the body-fixed `z` axis.
    pub(crate) const fn central_gravity(&self) -> CentralBodyGravity {
        CentralBodyGravity {
            gm_m3_per_s2: self.gm_m3_per_s2,
            j2: self.j2,
            equatorial_radius_m: self.ellipsoid.semi_major_axis_m(),
            pole_unit_in_frame: [0.0, 0.0, 1.0],
        }
    }

    /// The rotation vector of the body in its body-fixed axes.
    pub(crate) const fn rotation_body_fixed(&self) -> Vector3<f64> {
        Vector3::new(0.0, 0.0, self.rotation_rate_rad_per_s)
    }

    /// The apparent gravity at the body-fixed position `r` and its
    /// gradient `∂g/∂r`, both in the body-fixed axes.
    pub(crate) fn apparent_gravity_body_fixed(
        &self,
        r: &Vector3<f64>,
    ) -> (Vector3<f64>, Matrix3<f64>) {
        let central = self.central_gravity();
        let omega = self.rotation_body_fixed();
        (
            central.acceleration(r) + centrifugal(&omega, r),
            central.gradient(r) + centrifugal_gradient(&omega),
        )
    }
}

/// The centrifugal acceleration `−ω × (ω × r)`.
pub(crate) fn centrifugal(omega: &Vector3<f64>, r: &Vector3<f64>) -> Vector3<f64> {
    -omega.cross(&omega.cross(r))
}

/// The gradient `−[ω×]²` of the centrifugal acceleration.
pub(crate) fn centrifugal_gradient(omega: &Vector3<f64>) -> Matrix3<f64> {
    let skew = omega.cross_matrix();
    -(skew * skew)
}

/// The gravity and the frame rotation of one navigation frame. The
/// position, the gravity, and the rotation are in the axes of the frame.
pub(crate) trait GravityField {
    /// The apparent gravity at `p` and its gradient `∂g/∂p`.
    fn gravity(&self, p: &Vector3<f64>) -> (Vector3<f64>, Matrix3<f64>);
    /// The rotation vector of the frame relative to inertial space.
    fn frame_rotation(&self) -> Vector3<f64>;
}

/// A tangent plane on a rotating body. The plane is a rigid Cartesian
/// frame fixed to the body: `p_bf = r_origin + Rᵀ·p_plane`, with `R` the
/// rotation from the body-fixed axes to the plane axes. The plane has no
/// flat-earth approximation, because the gravity reads the body-centered
/// position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SurfaceFrame {
    pub(crate) plane: LocalTangentPlane,
    pub(crate) body: PlanetaryBody,
    origin_body_fixed: Vector3<f64>,
    body_fixed_to_plane: Matrix3<f64>,
}

impl SurfaceFrame {
    pub(crate) fn new(plane: LocalTangentPlane, body: PlanetaryBody) -> Self {
        let origin = plane.origin_body_fixed();
        let rows = plane.body_fixed_to_ned_rows();
        Self {
            plane,
            body,
            origin_body_fixed: Vector3::new(origin.x_m, origin.y_m, origin.z_m),
            body_fixed_to_plane: Matrix3::from_fn(|i, j| rows[i][j]),
        }
    }

    /// The body-centered body-fixed position of the plane offset `p`.
    pub(crate) fn body_fixed_position(&self, p: &Vector3<f64>) -> Vector3<f64> {
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

#[cfg(test)]
mod tests;
