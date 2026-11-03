//! Surface shapes and the intersection of a ray with them (ADR-0015).
//!
//! A [`SurfaceShape`] is the surface of one body in its body-fixed
//! frame. A ray from a sensor intersects it at one [`SurfaceHit`]. The
//! trait has no allocation in its signature, so a polyhedron of a small
//! body (for example Phobos or an asteroid) can implement it from a
//! static mesh. [`BodyEllipsoid`] implements it in closed form.

use navigate_contract::FrameId;

use crate::ellipsoid::BodyEllipsoid;
use crate::float_math::sqrt;

/// The largest deviation of the norm of a ray direction from 1 that
/// [`BodyEllipsoid`] accepts. A larger deviation is a caller error, not
/// rounding.
const UNIT_DIRECTION_TOLERANCE: f64 = 1e-6;

/// The first intersection of a ray with a surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceHit {
    /// The distance along the ray from its origin to the surface, in
    /// meters, more than 0.
    pub range_m: f64,
    /// The intersection point in the body-fixed frame, in meters.
    pub point_body_fixed_m: [f64; 3],
    /// The unit normal of the surface at the point, away from the body.
    pub outward_normal_unit: [f64; 3],
}

/// The surface of one body in its body-fixed frame.
pub trait SurfaceShape {
    /// The body-fixed frame of every point, direction, and normal.
    fn frame(&self) -> FrameId;

    /// The first intersection of the ray from `origin_body_fixed_m` in
    /// the direction `direction_body_fixed_unit`, both in
    /// [`Self::frame`]. The result is `None` when the ray misses the
    /// surface, when the surface is behind the origin, or when the
    /// origin is not outside the surface.
    fn intersect_ray(
        &self,
        origin_body_fixed_m: [f64; 3],
        direction_body_fixed_unit: [f64; 3],
    ) -> Option<SurfaceHit>;
}

impl SurfaceShape for BodyEllipsoid {
    fn frame(&self) -> FrameId {
        BodyEllipsoid::frame(self)
    }

    /// Exact ray–ellipsoid intersection. A scale of the polar axis by
    /// `a/b` turns the ellipsoid of revolution into a sphere of radius
    /// `a` and keeps the ray parameter. The quadratic uses the stable
    /// root form, so a ray far from the body keeps its precision. The
    /// normal is the normalized gradient `(x/a², y/a², z/b²)`.
    ///
    /// The direction must be a unit vector within 10^-6. The function
    /// normalizes it, so the range is exact.
    fn intersect_ray(
        &self,
        origin_body_fixed_m: [f64; 3],
        direction_body_fixed_unit: [f64; 3],
    ) -> Option<SurfaceHit> {
        let direction = unit_direction(direction_body_fixed_unit)?;
        if !origin_body_fixed_m.iter().all(|c| c.is_finite()) {
            return None;
        }
        let a = self.semi_major_axis_m();
        let b = self.semi_minor_axis_m();
        let polar_scale = a / b;
        let [ox, oy, oz] = origin_body_fixed_m;
        let [dx, dy, dz] = direction;
        let (sz, sdz) = (oz * polar_scale, dz * polar_scale);
        let qa = dx * dx + dy * dy + sdz * sdz;
        let half_qb = ox * dx + oy * dy + sz * sdz;
        let qc = ox * ox + oy * oy + sz * sz - a * a;
        // An origin on or inside the surface sees the surface from
        // within, which is not a surface intersection of a sensor.
        if qc <= 0.0 {
            return None;
        }
        let discriminant = half_qb * half_qb - qa * qc;
        // A ray from outside that points away from the body cannot meet
        // it: both roots are then negative.
        if discriminant < 0.0 || half_qb >= 0.0 {
            return None;
        }
        // Both roots are positive. The stable form gives the near root
        // without the cancellation of `-B - sqrt(D)`.
        let q = -half_qb + sqrt(discriminant);
        let range_m = qc / q;
        if !range_m.is_finite() || range_m <= 0.0 {
            return None;
        }
        let point = [ox + range_m * dx, oy + range_m * dy, oz + range_m * dz];
        let gradient = [point[0] / (a * a), point[1] / (a * a), point[2] / (b * b)];
        Some(SurfaceHit {
            range_m,
            point_body_fixed_m: point,
            outward_normal_unit: normalized(gradient)?,
        })
    }
}

/// `direction` normalized, when it is finite and a unit vector within
/// [`UNIT_DIRECTION_TOLERANCE`].
fn unit_direction(direction: [f64; 3]) -> Option<[f64; 3]> {
    let norm = norm(direction);
    if !norm.is_finite() || (norm - 1.0).abs() > UNIT_DIRECTION_TOLERANCE {
        return None;
    }
    normalized(direction)
}

fn norm(v: [f64; 3]) -> f64 {
    sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
}

fn normalized(v: [f64; 3]) -> Option<[f64; 3]> {
    let length = norm(v);
    (length.is_finite() && length > 0.0).then(|| [v[0] / length, v[1] / length, v[2] / length])
}

#[cfg(test)]
mod tests;
