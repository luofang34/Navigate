//! Robust camera-fitting controls.
#![allow(clippy::expect_used, clippy::panic)]

use super::*;
use nalgebra::{UnitQuaternion, Vector2};
#[test]
fn recovers_free_pose_from_noisy_outlier_points() {
    let camera = CameraModel {
        width: 960,
        height: 544,
        fx: 700.0,
        fy: 690.0,
        cx: 479.5,
        cy: 271.5,
    };
    let expected = Pose {
        r: UnitQuaternion::from_euler_angles(0.12, -0.07, 0.08)
            .to_rotation_matrix()
            .into_inner(),
        t: Vector3::new(0.2, -0.1, 0.3),
    };
    let points: Vec<_> = (0..150)
        .map(|i| {
            let world = Vector3::new(
                (i % 15) as f64 / 9.0 - 0.7,
                (i / 15) as f64 / 8.0 - 0.5,
                3.0 + (i % 7) as f64 * 0.2,
            );
            let pixel = expected.project(&camera, world).unwrap_or_default()
                + if i % 5 == 0 {
                    Vector2::new(25.0, -19.0)
                } else {
                    Vector2::new(0.05, -0.08)
                };
            Point { world, pixel }
        })
        .collect();
    let (found, ids) = fit(&camera, &points, crate::reconstruction::seeds::identity())
        .unwrap_or_else(|| panic!("pose fit"));
    assert!(ids.len() >= 120);
    assert!((found.center() - expected.center()).norm() < 0.005);
    assert!(
        UnitQuaternion::from_matrix(&found.r).angle_to(&UnitQuaternion::from_matrix(&expected.r))
            < 0.001
    );
}
