//! Robust camera-fitting controls.
#![allow(clippy::expect_used, clippy::panic)]

use super::*;
use nalgebra::{UnitQuaternion, Vector2};
#[test]
fn recovers_large_rotation_without_a_previous_pose() {
    let camera = CameraModel {
        width: 960,
        height: 544,
        fx: 700.0,
        fy: 690.0,
        cx: 479.5,
        cy: 271.5,
    };
    let expected = Pose {
        r: UnitQuaternion::from_euler_angles(0.4, -0.8, 1.2)
            .to_rotation_matrix()
            .into_inner(),
        t: Vector3::new(0.2, -0.1, 10.0),
    };
    let points: Vec<_> = (0..150)
        .map(|i| {
            let world = Vector3::new(
                (i % 15) as f64 * 0.2 - 1.0,
                (i / 15) as f64 * 0.2 - 0.8,
                2.0 + (i % 7) as f64 * 0.3,
            );
            let mut pixel = expected
                .project(&camera, world)
                .expect("in front of camera");
            if i % 4 == 0 {
                pixel += Vector2::new(90.0, -45.0)
            }
            Point { world, pixel }
        })
        .collect();
    let proposal = estimate(&camera, &points).expect("P3P hypothesis");
    let (pose, inliers) = reprojection::fit(&camera, &points, proposal).expect("geometric support");
    assert!(inliers.len() >= 110);
    assert!(
        UnitQuaternion::from_matrix(&pose.r).angle_to(&UnitQuaternion::from_matrix(&expected.r))
            < 1e-6
    );
    assert!((pose.t - expected.t).norm() < 1e-6);
    assert!(estimate(&camera, &points[..19]).is_none());
}
