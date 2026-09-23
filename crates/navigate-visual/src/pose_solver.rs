//! Robust reprojection minimization from a bounded pose prior.

use crate::{CameraModel, CameraPose, VisualError};
use nalgebra::{SMatrix, SVector, Vector2, Vector3};

pub(crate) type Matrix6 = SMatrix<f64, 6, 6>;
type Vector6 = SVector<f64, 6>;

#[derive(Clone)]
pub(crate) struct Correspondence {
    pub world: Vector3<f64>,
    pub pixel: Vector2<f64>,
}

mod consensus;
pub(crate) fn initialize(
    camera: &CameraModel,
    points: &[Correspondence],
    initial: CameraPose,
    threshold: f64,
) -> CameraPose {
    consensus::initialize(camera, points, initial, threshold)
}

pub(crate) fn optimize(
    camera: &CameraModel,
    points: &[Correspondence],
    initial: CameraPose,
) -> Result<CameraPose, VisualError> {
    let mut pose = initial;
    let mut damping = 0.001;
    for _ in 0..40 {
        let (h, b) = normal_equations(camera, points, &pose);
        let regularized = h + Matrix6::from_diagonal(&h.diagonal().map(|v| damping * v.max(1e-9)));
        let delta = regularized
            .cholesky()
            .ok_or(VisualError::DegenerateGeometry)?
            .solve(&b);
        if !delta.iter().all(|v| v.is_finite()) {
            return Err(VisualError::DegenerateGeometry);
        }
        let candidate = pose.increment(&delta);
        if cost(camera, points, &candidate) < cost(camera, points, &pose) {
            pose = candidate;
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
    Ok(pose)
}

fn cost(camera: &CameraModel, points: &[Correspondence], pose: &CameraPose) -> f64 {
    points
        .iter()
        .map(|point| {
            let Some(pixel) = camera.project(pose, point.world) else {
                return 1e6;
            };
            let e = (pixel - point.pixel).norm();
            if e < 3.0 {
                0.5 * e * e
            } else {
                3.0 * (e - 1.5)
            }
        })
        .sum()
}

pub(crate) fn normal_equations(
    camera: &CameraModel,
    points: &[Correspondence],
    pose: &CameraPose,
) -> (Matrix6, Vector6) {
    let mut h = Matrix6::zeros();
    let mut b = Vector6::zeros();
    for point in points {
        let Some(projected) = camera.project(pose, point.world) else {
            continue;
        };
        let error = point.pixel - projected;
        let mut jacobian = SMatrix::<f64, 2, 6>::zeros();
        for axis in 0..6 {
            let mut delta = Vector6::zeros();
            delta[axis] = 0.01;
            if let Some(shifted) = camera.project(&pose.increment(&delta), point.world) {
                jacobian.set_column(axis, &((shifted - projected) / 0.01));
            }
        }
        let weight = 3.0 / error.norm().max(3.0);
        h += jacobian.transpose() * jacobian * weight;
        b += jacobian.transpose() * error * weight;
    }
    (h, b)
}

pub(crate) fn residual(camera: &CameraModel, pose: &CameraPose, point: &Correspondence) -> f64 {
    camera
        .project(pose, point.world)
        .map_or(f64::INFINITY, |p| (p - point.pixel).norm())
}

#[cfg(test)]
mod tests;
