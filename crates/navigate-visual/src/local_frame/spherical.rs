use super::MERCATOR_SPHERE_RADIUS_M as RADIUS;
use nalgebra::{Matrix3, Vector3};

pub(super) fn basis([lat, lon]: [f64; 2]) -> Matrix3<f64> {
    let (s, c) = lat.to_radians().sin_cos();
    let (sl, cl) = lon.to_radians().sin_cos();
    Matrix3::from_columns(&[
        Vector3::new(-sl, cl, 0.0),
        Vector3::new(-s * cl, -s * sl, c),
        Vector3::new(c * cl, c * sl, s),
    ])
}

pub(super) fn local(anchor: [f64; 2], [lat, lon, alt]: [f64; 3]) -> Vector3<f64> {
    let axes = basis(anchor);
    let origin = axes.column(2) * RADIUS;
    axes.transpose() * (basis([lat, lon]).column(2) * (RADIUS + alt) - origin)
}

pub(super) fn geodetic(anchor: [f64; 2], position: Vector3<f64>) -> [f64; 3] {
    let axes = basis(anchor);
    let earth = axes * position + axes.column(2) * RADIUS;
    [
        earth.z.atan2(earth.x.hypot(earth.y)).to_degrees(),
        earth.y.atan2(earth.x).to_degrees(),
        earth.norm() - RADIUS,
    ]
}

pub(super) fn ellipsoid_displacement(anchor: [f64; 2], position: Vector3<f64>) -> f64 {
    let point = geodetic(anchor, position);
    let reference = ellipsoid([anchor[0], anchor[1], 0.0]);
    let ellipsoid_position = basis(anchor).transpose() * (ellipsoid(point) - reference);
    (ellipsoid_position - position).norm()
}

fn ellipsoid([lat, lon, height]: [f64; 3]) -> Vector3<f64> {
    const A: f64 = 6_378_137.0;
    const F: f64 = 1.0 / 298.257_223_563;
    const E2: f64 = F * (2.0 - F);
    let (s, c) = lat.to_radians().sin_cos();
    let (sl, cl) = lon.to_radians().sin_cos();
    let n = A / (1.0 - E2 * s * s).sqrt();
    Vector3::new(
        (n + height) * c * cl,
        (n + height) * c * sl,
        (n * (1.0 - E2) + height) * s,
    )
}
