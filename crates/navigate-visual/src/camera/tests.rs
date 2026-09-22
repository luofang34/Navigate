#![allow(clippy::expect_used)]

use super::*;

#[test]
fn project_unproject_round_trip_for_all_views() {
    let camera = CameraModel {
        width: 640,
        height: 480,
        fx: 510.0,
        fy: 530.0,
        cx: 311.2,
        cy: 241.7,
    };
    for pitch in [0.0_f64, 0.7, 1.4] {
        let pose = CameraPose {
            position: Vector3::new(23.0, -41.0, 1300.0),
            orientation: UnitQuaternion::from_euler_angles(pitch, 0.08, -0.3),
        };
        for pixel in [Vector2::new(20.0, 40.0), Vector2::new(410.2, 312.1)] {
            let world = camera.unproject(&pose, pixel, 1732.0);
            let projected = camera
                .project(&pose, world)
                .expect("point is in front of the eye");
            assert!((projected - pixel).norm() < 1e-9);
        }
        assert!(
            camera
                .project(&pose, pose.position + pose.orientation * Vector3::z())
                .is_none()
        );
    }
}

#[test]
fn invalid_calibration_and_prior_are_rejected() {
    let camera = CameraModel {
        width: 640,
        height: 480,
        fx: f64::NAN,
        fy: 530.0,
        cx: 311.2,
        cy: 241.7,
    };
    assert!(camera.validate().is_err());
    let prior = PosePrior {
        pose: CameraPose {
            position: Vector3::zeros(),
            orientation: UnitQuaternion::identity(),
        },
        position_radius_m: -1.0,
        attitude_radius_rad: 0.1,
    };
    assert!(prior.validate().is_err());
}
