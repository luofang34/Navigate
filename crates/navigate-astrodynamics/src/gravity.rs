//! Gravity fields and frame rotation of the navigation frames (ADR-0014,
//! ADR-0016).
//!
//! A [`GravityField`] gives the apparent gravity in one navigation frame
//! and the rotation of that frame relative to inertial space. Each model
//! also gives its exact gravity gradient, which the error-state dynamics
//! of a filter use. [`SurfaceFrame`] is a tangent plane fixed to a
//! rotating [`crate::PlanetaryBody`]. [`SpaceEnvironment`] is a
//! Cartesian frame centered on one body, with a fixed number of third
//! bodies and an optional frame rotation. [`SphericalHarmonicField`]
//! gives the gravity of a body from borrowed Stokes coefficients. Every
//! type is `Copy` and does not allocate.

use nalgebra::{Matrix3, Vector3};

mod central;
mod point_mass;
mod space;
mod spherical_harmonics;
mod surface;

#[cfg(test)]
mod tests;

pub use central::CentralBodyGravity;
pub use point_mass::{MAX_THIRD_BODIES, PointMassBody, ThirdBodyCapacityError, ThirdBodySet};
pub use space::{SpaceEnvironment, SpaceGravityModel};
pub use spherical_harmonics::{
    HarmonicFieldError, HarmonicNormalization, MAX_INLINE_HARMONIC_DEGREE, SphericalHarmonicField,
    harmonic_coefficient_count, harmonic_index,
};
pub use surface::SurfaceFrame;

/// Largest magnitude of `J2` that [`crate::PlanetaryBody::is_plausible`]
/// and [`CentralBodyGravity::is_plausible`] accept.
pub(crate) const MAX_ABS_J2: f64 = 0.1;

/// Below this distance from a point mass in meters, the gravity of that
/// mass and its gradient are zero. The point-mass field is singular
/// there.
pub(crate) const MIN_CENTRAL_DISTANCE_M: f64 = 1.0;

/// The gravity and the frame rotation of one navigation frame. The
/// position, the gravity, and the rotation are in the axes of the frame.
pub trait GravityField {
    /// The apparent gravity in m/s² at the position `p` in meters, and
    /// its gradient `∂g/∂p` in 1/s².
    fn gravity(&self, p: &Vector3<f64>) -> (Vector3<f64>, Matrix3<f64>);
    /// The rotation vector of the frame relative to inertial space in
    /// rad/s.
    fn frame_rotation(&self) -> Vector3<f64>;
}

impl<T: GravityField + ?Sized> GravityField for &T {
    fn gravity(&self, p: &Vector3<f64>) -> (Vector3<f64>, Matrix3<f64>) {
        (**self).gravity(p)
    }

    fn frame_rotation(&self) -> Vector3<f64> {
        (**self).frame_rotation()
    }
}

/// The centrifugal acceleration `−ω × (ω × r)` in m/s² for the rotation
/// vector `omega` in rad/s and the position `r` in meters.
#[must_use]
pub fn centrifugal(omega: &Vector3<f64>, r: &Vector3<f64>) -> Vector3<f64> {
    -omega.cross(&omega.cross(r))
}

/// The gradient `−[ω×]²` of the centrifugal acceleration in 1/s².
#[must_use]
pub fn centrifugal_gradient(omega: &Vector3<f64>) -> Matrix3<f64> {
    let skew = omega.cross_matrix();
    -(skew * skew)
}
