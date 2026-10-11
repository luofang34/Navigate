//! Bounded consensus initialization for surface pose fitting.
use super::{Correspondence, Motion, optimize_motion};
use crate::{CameraModel, CameraPose};
use nalgebra::{UnitQuaternion, Vector2, Vector3};

pub(super) fn initialize(
    camera: &CameraModel,
    points: &[Correspondence],
    initial: CameraPose,
    threshold: f64,
    motion: Motion,
) -> CameraPose {
    if points.len() < 6 {
        return initial;
    }
    let mut best = initial;
    let mut support = inliers(camera, points, &best, threshold);
    let mut seed = 0x7812_531a_u64;
    let mut budget = 4096;
    let mut trial = 0_usize;
    while trial < budget && support.len() < points.len() {
        trial = trial.wrapping_add(1);
        let mut indices = [usize::MAX; 4];
        for slot in 0..indices.len() {
            loop {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let index = (seed >> 32) as usize % points.len();
                if !indices[..slot].contains(&index) {
                    indices[slot] = index;
                    break;
                }
            }
        }
        let sample = indices.map(|i| points[i].clone());
        let Ok(pose) = optimize_motion(camera, &sample, initial, motion) else {
            continue;
        };
        if better_support(camera, points, &pose, threshold, support.len()).is_none() {
            continue;
        }
        best = pose;
        support = inliers(camera, points, &best, threshold);
        // The sampling budget bounds work. It is not an accuracy probability.
        let fraction = support.len() as f64 / points.len() as f64;
        budget = budget.min((0.001_f64.ln() / (1.0 - fraction.powi(4)).ln()).ceil() as usize);
        budget = budget.max(64);
    }
    if support.len() >= 6 {
        let selected: Vec<_> = support.iter().map(|&i| points[i].clone()).collect();
        if let Ok(refined) = optimize_motion(camera, &selected, best, motion)
            && better_support(camera, points, &refined, threshold, support.len() - 1).is_some()
        {
            best = refined;
        }
    }
    best
}

struct Projection<'a> {
    camera: &'a CameraModel,
    position: Vector3<f64>,
    rotation: UnitQuaternion<f64>,
}
impl<'a> Projection<'a> {
    fn new(camera: &'a CameraModel, pose: &CameraPose) -> Self {
        Self {
            camera,
            position: pose.position,
            rotation: pose.orientation.inverse(),
        }
    }

    fn supports(&self, point: &Correspondence, threshold: f64) -> bool {
        let eye = self.rotation * (point.world - self.position);
        let depth = -eye.z;
        if depth <= 0.01 || !depth.is_finite() {
            return false;
        }
        let pixel = Vector2::new(
            self.camera.fx * eye.x / depth + self.camera.cx,
            self.camera.cy - self.camera.fy * eye.y / depth,
        );
        (pixel - point.pixel).norm() <= threshold
    }
}

fn better_support(
    camera: &CameraModel,
    points: &[Correspondence],
    pose: &CameraPose,
    threshold: f64,
    minimum: usize,
) -> Option<usize> {
    let projection = Projection::new(camera, pose);
    let mut count = 0_usize;
    for (i, point) in points.iter().enumerate() {
        // No untested point can make this hypothesis beat the retained one.
        if count + points.len() - i <= minimum {
            return None;
        }
        if projection.supports(point, threshold) {
            count = count.wrapping_add(1);
        }
    }
    (count > minimum).then_some(count)
}

fn inliers(
    camera: &CameraModel,
    points: &[Correspondence],
    pose: &CameraPose,
    threshold: f64,
) -> Vec<usize> {
    let projection = Projection::new(camera, pose);
    points
        .iter()
        .enumerate()
        .filter_map(|(i, point)| projection.supports(point, threshold).then_some(i))
        .collect()
}

#[cfg(test)]
mod tests;
