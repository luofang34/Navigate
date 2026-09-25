//! Pose proposals from three local points, independent of a previous pose guess.
use super::reprojection::Point;
use super::*;
use cv_core::nalgebra::{Point3, Unit, Vector3 as CvVector};
use cv_core::sample_consensus::Estimator;
use cv_core::{FeatureWorldMatch, Pose as CvPose, Projective, WorldPoint};
use lambda_twist::LambdaTwist;
pub(super) fn estimate(camera: &CameraModel, points: &[Point]) -> Option<Pose> {
    if points.len() < 20 {
        return None;
    }
    let data: Vec<_> = points
        .iter()
        .map(|p| {
            FeatureWorldMatch(
                Unit::new_normalize(CvVector::new(
                    (p.pixel.x - camera.cx) / camera.fx,
                    (p.pixel.y - camera.cy) / camera.fy,
                    1.0,
                )),
                WorldPoint::from_point(Point3::new(p.world.x, p.world.y, p.world.z)),
            )
        })
        .collect();
    let mut seed = 0x198122_u64;
    let mut best = None;
    let mut count = 0;
    for _ in 0..512 {
        let mut indices = Vec::new();
        while indices.len() < 3 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let i = (seed >> 32) as usize % points.len();
            if !indices.contains(&i) {
                indices.push(i)
            }
        }
        for pose in LambdaTwist::new().estimate(indices.iter().map(|&i| data[i])) {
            let iso = pose.isometry();
            let r = nalgebra::Matrix3::from_column_slice(iso.rotation.matrix().as_slice());
            let t = Vector3::from_column_slice(iso.translation.vector.as_slice());
            let pose = Pose { r, t };
            let support = points
                .iter()
                .filter(|p| reprojection::residual(camera, pose, p) < 2.5)
                .count();
            if support > count {
                count = support;
                best = Some(pose)
            }
        }
        if count * 10 >= points.len() * 9 {
            break;
        }
    }
    best.filter(|_| count >= 20)
}

#[cfg(test)]
mod tests;
