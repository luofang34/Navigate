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

#[test]
fn crop_resize_preserves_rays_with_nonuniform_scale_and_offcentre_principal_point() {
    let source = CameraModel {
        width: 1920,
        height: 1440,
        fx: 1500.0,
        fy: 1490.0,
        cx: 960.3,
        cy: 718.7,
    };
    let target = source
        .crop_and_resize([88, 66, 1744, 1308], [640, 400])
        .unwrap();
    let pose = CameraPose {
        position: Vector3::new(10.0, -30.0, 100.0),
        orientation: UnitQuaternion::from_euler_angles(0.3, -0.2, 0.7),
    };
    for pixel in [Vector2::new(88.0, 66.0), Vector2::new(1500.25, 1230.75)] {
        let world = source.unproject(&pose, pixel, 450.0);
        let expected = Vector2::new(
            (pixel.x - 88.0 + 0.5) * 640.0 / 1744.0 - 0.5,
            (pixel.y - 66.0 + 0.5) * 400.0 / 1308.0 - 0.5,
        );
        assert!((target.project(&pose, world).unwrap() - expected).norm() < 1e-9);
        assert!((target.unproject(&pose, expected, 450.0) - world).norm() < 1e-9);
    }
    assert_eq!(
        source
            .crop_and_resize([0, 0, 1920, 1440], [1920, 1440])
            .unwrap(),
        source
    );
    for crop in [
        [0, 0, 0, 100],
        [1910, 0, 20, 100],
        [0, 1430, 100, 20],
        [u32::MAX, 0, 100, 100],
    ] {
        assert!(source.crop_and_resize(crop, [640, 480]).is_err());
    }
    assert!(
        source
            .crop_and_resize([0, 0, 1920, 1440], [0, 480])
            .is_err()
    );
}
