//! Robust six-degree camera fitting in arbitrary scene units.
use super::{CameraModel, Pose};
use nalgebra::{Matrix3, SMatrix, SVector, Vector2, Vector3};
#[derive(Clone)]
pub(super) struct Point {
    pub world: Vector3<f64>,
    pub pixel: Vector2<f64>,
}
pub(super) fn residual(c: &CameraModel, pose: Pose, point: &Point) -> f64 {
    pose.project(c, point.world)
        .map_or(f64::INFINITY, |p| (p - point.pixel).norm())
}
fn cost(c: &CameraModel, pose: Pose, points: &[Point]) -> f64 {
    points
        .iter()
        .map(|p| {
            let r = residual(c, pose, p).min(1000.0);
            if r < 3.0 {
                r * r * 0.5
            } else {
                3.0 * (r - 1.5)
            }
        })
        .sum()
}
fn optimize(c: &CameraModel, points: &[Point], mut pose: Pose) -> Option<Pose> {
    let mut damping = 0.001;
    for _ in 0..35 {
        let mut h = SMatrix::<f64, 6, 6>::zeros();
        let mut b = SVector::<f64, 6>::zeros();
        for point in points {
            let p = pose.r * point.world + pose.t;
            let Some(pixel) = pose.project(c, point.world) else {
                continue;
            };
            let error = point.pixel - pixel;
            let projection = SMatrix::<f64, 2, 3>::new(
                c.fx / p.z,
                0.0,
                -c.fx * p.x / p.z.powi(2),
                0.0,
                c.fy / p.z,
                -c.fy * p.y / p.z.powi(2),
            );
            let mut motion = SMatrix::<f64, 3, 6>::zeros();
            motion
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&Matrix3::identity());
            motion
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&(-p.cross_matrix() * 0.001));
            let jacobian = projection * motion;
            let weight = 3.0 / error.norm().max(3.0);
            h += jacobian.transpose() * jacobian * weight;
            b += jacobian.transpose() * error * weight;
        }
        let regularized =
            h + SMatrix::<f64, 6, 6>::from_diagonal(&h.diagonal().map(|x| damping * x.max(1e-9)));
        let delta = regularized.cholesky()?.solve(&b);
        if !delta.iter().all(|x| x.is_finite()) {
            return None;
        }
        let next = pose.increment(delta);
        if cost(c, next, points) < cost(c, pose, points) {
            pose = next;
            damping = (damping * 0.3).max(1e-8);
            if delta.norm() < 1e-5 {
                break;
            }
        } else {
            damping *= 10.0;
            if damping > 1e10 {
                break;
            }
        }
    }
    Some(pose)
}
pub(super) fn fit(c: &CameraModel, points: &[Point], initial: Pose) -> Option<(Pose, Vec<usize>)> {
    if points.len() < 20 {
        return None;
    }
    let support = |pose| {
        points
            .iter()
            .enumerate()
            .filter_map(|(i, p)| (residual(c, pose, p) <= 2.5).then_some(i))
            .collect::<Vec<_>>()
    };
    let mut best = optimize(c, points, initial)?;
    let mut ids = support(best);
    let mut seed = 0x873919_u64;
    for _ in 0..256 {
        if ids.len() as f64 >= points.len() as f64 * 0.9 {
            break;
        }
        let mut indices = Vec::new();
        while indices.len() < 6 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let index = (seed >> 32) as usize % points.len();
            if !indices.contains(&index) {
                indices.push(index)
            }
        }
        let subset: Vec<_> = indices.iter().map(|&i| points[i].clone()).collect();
        let Some(pose) = optimize(c, &subset, initial) else {
            continue;
        };
        let found = support(pose);
        if found.len() > ids.len() {
            best = pose;
            ids = found;
        }
    }
    let subset: Vec<_> = ids.iter().map(|&i| points[i].clone()).collect();
    if let Some(next) = optimize(c, &subset, best) {
        let found = support(next);
        if found.len() >= ids.len() {
            best = next;
            ids = found
        }
    }
    (ids.len() >= 20).then_some((best, ids))
}
pub(super) fn triangulate(
    a: Pose,
    b: Pose,
    x: Vector3<f64>,
    y: Vector3<f64>,
) -> Option<(Vector3<f64>, f64)> {
    let r = b.r * a.r.transpose();
    let t = b.t - r * a.t;
    let (point, angle) =
        super::triangulation::positive_depth(r, t, &super::triangulation::Rays { a: x, b: y })?;
    Some((a.r.transpose() * (point - a.t), angle))
}

#[cfg(test)]
mod tests;
