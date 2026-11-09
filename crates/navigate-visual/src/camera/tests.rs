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

/// A small deterministic generator, so the agreement tests replay.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn symmetric(&mut self, half: f64) -> f64 {
        (self.unit() * 2.0 - 1.0) * half
    }

    fn pose(&mut self) -> CameraPose {
        let axis = Vector3::new(
            self.symmetric(3.0),
            self.symmetric(3.0),
            self.symmetric(3.0),
        );
        CameraPose {
            position: Vector3::new(
                self.symmetric(5000.0),
                self.symmetric(5000.0),
                self.symmetric(2000.0),
            ),
            orientation: UnitQuaternion::from_scaled_axis(axis),
        }
    }
}

fn agreement_camera() -> CameraModel {
    CameraModel {
        width: 800,
        height: 600,
        fx: 610.0,
        fy: 640.0,
        cx: 401.3,
        cy: 297.8,
    }
}

#[test]
fn intrinsics_round_trip_is_lossless() {
    let camera = agreement_camera();
    let intrinsics = navigate_camera::Intrinsics::from(camera);
    assert_eq!(CameraModel::from(intrinsics), camera);
    assert_eq!(intrinsics.matrix()[(0, 2)], camera.cx);
}

#[test]
fn pose_round_trip_keeps_position_and_rotation() {
    let mut rng = Lcg(7);
    for _ in 0..200 {
        let pose = rng.pose();
        let back = CameraPose::from(navigate_camera::OpticalPose::from(pose));
        assert_eq!(back.position, pose.position);
        assert!(back.orientation.angle_to(&pose.orientation) < 1e-12);
    }
}

#[test]
fn optical_projection_agrees_with_gl_projection() {
    let camera = agreement_camera();
    let intrinsics = navigate_camera::Intrinsics::from(camera);
    let mut rng = Lcg(42);
    for _ in 0..500 {
        let pose = rng.pose();
        let optical = navigate_camera::OpticalPose::from(pose);
        let pixel = Vector2::new(rng.symmetric(600.0) + 400.0, rng.symmetric(450.0) + 300.0);
        let depth = 1.0 + rng.unit() * 8000.0;
        let front = camera.unproject(&pose, pixel, depth);
        let optical_front = optical.unproject(&intrinsics, &pixel, depth);
        assert!((front - optical_front).norm() < 1e-9 * depth.max(1.0) * 10.0);
        let gl = camera.project(&pose, front).expect("point is in front");
        let ours = optical
            .project(&intrinsics, &front)
            .expect("point is in front");
        assert!((gl - ours).norm() < 1e-6, "{gl} != {ours}");
        assert!((ours - pixel).norm() < 1e-6);
        let behind = camera.unproject(&pose, pixel, -depth);
        assert!(camera.project(&pose, behind).is_none());
        assert!(optical.project(&intrinsics, &behind).is_none());
    }
}
