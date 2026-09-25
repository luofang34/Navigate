//! Calibrated rays and two-view triangulation in camera coordinates.
use super::*;
use nalgebra::Matrix3;
#[derive(Clone)]
pub(super) struct Rays {
    pub a: Vector3<f64>,
    pub b: Vector3<f64>,
}
pub(super) fn ray(c: &CameraModel, p: Vector2<f64>) -> Vector3<f64> {
    Vector3::new((p.x - c.cx) / c.fx, (p.y - c.cy) / c.fy, 1.0)
}
pub(super) fn positive_depth(
    r: Matrix3<f64>,
    t: Vector3<f64>,
    p: &Rays,
) -> Option<(Vector3<f64>, f64)> {
    let a = r * p.a;
    let b = p.b;
    let normal = nalgebra::Matrix2::new(a.dot(&a), -a.dot(&b), -a.dot(&b), b.dot(&b));
    let rhs = nalgebra::Vector2::new(-a.dot(&t), b.dot(&t));
    let depth = normal.try_inverse()? * rhs;
    if depth.x <= 0.0 || depth.y <= 0.0 {
        return None;
    }
    let angle = (a.normalize().dot(&b.normalize()))
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees();
    Some((p.a * depth.x, angle))
}
pub(super) fn essential_poses(e: Matrix3<f64>) -> Option<Vec<(Matrix3<f64>, Vector3<f64>)>> {
    let svd = e.svd(true, true);
    let mut u = svd.u?;
    let mut v = svd.v_t?;
    if u.determinant() < 0.0 {
        u.column_mut(2).neg_mut()
    }
    if v.determinant() < 0.0 {
        v.row_mut(2).neg_mut()
    }
    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let t = u.column(2).into_owned();
    Some(vec![
        (u * w * v, t),
        (u * w * v, -t),
        (u * w.transpose() * v, t),
        (u * w.transpose() * v, -t),
    ])
}
