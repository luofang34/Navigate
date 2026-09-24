#![allow(clippy::expect_used)]

use super::*;
use nalgebra::UnitQuaternion;

#[test]
fn six_axis_pose_converges_from_offset_prior() {
    let camera = CameraModel {
        width: 640,
        height: 480,
        fx: 550.0,
        fy: 550.0,
        cx: 319.5,
        cy: 239.5,
    };
    for pitch in [0.0, 0.7, 1.3] {
        let truth = CameraPose {
            position: Vector3::new(71.0, -123.0, 1600.0),
            orientation: UnitQuaternion::from_euler_angles(pitch, 0.02, 0.2),
        };
        let mut points = Vec::new();
        for y in 0..8 {
            for x in 0..10 {
                let pixel = Vector2::new(50.0 + f64::from(x) * 55.0, 40.0 + f64::from(y) * 50.0);
                let depth = 1300.0 + 90.0 * f64::from(x + y).sin();
                points.push(Correspondence {
                    world: camera.unproject(&truth, pixel, depth),
                    pixel,
                });
            }
        }
        let initial = CameraPose {
            position: truth.position + Vector3::new(60.0, -40.0, 30.0),
            orientation: truth.orientation * UnitQuaternion::from_euler_angles(0.02, -0.01, 0.015),
        };
        let estimated = optimize(&camera, &points, initial).expect("usable geometry");
        assert!((estimated.position - truth.position).norm() < 0.001);
        assert!(estimated.orientation.angle_to(&truth.orientation) < 1e-6);
    }
}

#[test]
fn analytic_normal_equations_match_central_differences() {
    let camera = CameraModel {
        width: 960,
        height: 540,
        fx: 812.0,
        fy: 798.0,
        cx: 479.5,
        cy: 269.5,
    };
    for pitch in [0.0, 0.7, 1.5, 2.8] {
        let pose = CameraPose {
            position: Vector3::new(21.0, -30.0, 115.0),
            orientation: UnitQuaternion::from_euler_angles(pitch, -0.3, 1.2),
        };
        for depth in [3.0, 90.0, 1500.0] {
            let pixel = Vector2::new(721.0, 102.0);
            let point = Correspondence {
                world: camera.unproject(&pose, pixel, depth),
                pixel: pixel + Vector2::new(1.5, -0.75),
            };
            let (h, b) = normal_equations(&camera, std::slice::from_ref(&point), &pose);
            let mut numerical = SMatrix::<f64, 2, 6>::zeros();
            for axis in 0..6 {
                let mut delta = Vector6::zeros();
                delta[axis] = 1e-4;
                let plus = camera
                    .project(&pose.increment(&delta), point.world)
                    .expect("visible");
                let minus = camera
                    .project(&pose.increment(&(-delta)), point.world)
                    .expect("visible");
                numerical.set_column(axis, &((plus - minus) / 2e-4));
            }
            let expected = numerical.transpose() * numerical;
            assert!((h - expected).norm() / expected.norm() < 1e-7);
            assert!((b - numerical.transpose() * Vector2::new(1.5, -0.75)).norm() < 1e-5);
        }
    }
}
