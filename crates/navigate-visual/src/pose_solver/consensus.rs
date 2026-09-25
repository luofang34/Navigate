//! Bounded consensus initialization for surface pose fitting.
use super::{Correspondence, Motion, optimize_motion, residual};
use crate::{CameraModel, CameraPose};

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
        let sample: Vec<_> = indices.iter().map(|&i| points[i].clone()).collect();
        let Ok(pose) = optimize_motion(camera, &sample, initial, motion) else {
            continue;
        };
        let found = inliers(camera, points, &pose, threshold);
        if found.len() <= support.len() {
            continue;
        }
        best = pose;
        support = found;
        // The sampling budget bounds work. It is not an accuracy probability.
        let fraction = support.len() as f64 / points.len() as f64;
        budget = budget.min((0.001_f64.ln() / (1.0 - fraction.powi(4)).ln()).ceil() as usize);
        budget = budget.max(64);
    }
    if support.len() >= 6 {
        let selected: Vec<_> = support.iter().map(|&i| points[i].clone()).collect();
        if let Ok(refined) = optimize_motion(camera, &selected, best, motion)
            && inliers(camera, points, &refined, threshold).len() >= support.len()
        {
            best = refined;
        }
    }
    best
}

fn inliers(
    camera: &CameraModel,
    points: &[Correspondence],
    pose: &CameraPose,
    threshold: f64,
) -> Vec<usize> {
    points
        .iter()
        .enumerate()
        .filter_map(|(i, point)| (residual(camera, pose, point) <= threshold).then_some(i))
        .collect()
}
