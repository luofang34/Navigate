use nalgebra::{UnitQuaternion, Vector2, Vector3};

use super::*;
use crate::{CameraMount, Intrinsics};

fn sample_pose() -> OpticalPose {
    let q = UnitQuaternion::from_euler_angles(0.3, -0.7, 2.1);
    OpticalPose::new(Vector3::new(10.0, -4.0, 120.0), q.to_rotation_matrix())
}

#[test]
fn row_major_round_trips() {
    let pose = sample_pose();
    let (eye, rows) = pose.to_row_major();
    let back = OpticalPose::from_row_major(eye, rows).unwrap();
    assert!((back.world_to_cam.matrix() - pose.world_to_cam.matrix()).amax() < 1e-15);
    assert_eq!(back.eye, pose.eye);
}

#[test]
fn a_reflection_is_refused() {
    let rows = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0];
    assert!(OpticalPose::from_row_major([0.0; 3], rows).is_err());
}

#[test]
fn gl_round_trip_keeps_the_projection() {
    let pose = sample_pose();
    let gl = pose.world_from_gl_eye();
    let back = OpticalPose::from_gl(pose.eye, gl);
    assert!((back.world_to_cam.matrix() - pose.world_to_cam.matrix()).amax() < 1e-12);
}

#[test]
fn gl_camera_looks_along_minus_z_of_its_eye() {
    // A GL eye with identity orientation looks along world -Z, image top +Y.
    let pose = OpticalPose::from_gl(Vector3::zeros(), UnitQuaternion::identity());
    let k = Intrinsics::centred(101, 101, 1.0).unwrap();
    let centre = pose.project(&k, &Vector3::new(0.0, 0.0, -5.0)).unwrap();
    assert!((centre - Vector2::new(50.0, 50.0)).norm() < 1e-12);
    let above = pose.project(&k, &Vector3::new(0.0, 1.0, -5.0)).unwrap();
    assert!(above.y < 50.0, "world +Y is up in the image");
}

#[test]
fn unproject_inverts_project() {
    let pose = sample_pose();
    let k = Intrinsics::centred(640, 512, 0.9).unwrap();
    let world = pose.unproject(&k, &Vector2::new(100.0, 400.0), 250.0);
    let pixel = pose.project(&k, &world).unwrap();
    assert!((pixel - Vector2::new(100.0, 400.0)).norm() < 1e-9);
}

#[test]
fn nadir_mount_on_a_level_north_body_looks_down_with_the_nose_up_the_image() {
    // World axes NED; body level and heading north: world_from_body = identity.
    let mount = CameraMount::nadir();
    let pose = mount.camera_pose(&Vector3::zeros(), &UnitQuaternion::identity());
    let k = Intrinsics::centred(101, 101, 1.0).unwrap();
    let below = pose.project(&k, &Vector3::new(0.0, 0.0, 100.0)).unwrap();
    assert!((below - Vector2::new(50.0, 50.0)).norm() < 1e-9);
    let ahead = pose.project(&k, &Vector3::new(10.0, 0.0, 100.0)).unwrap();
    assert!(ahead.y < 50.0, "the nose is toward the image top");
    let right = pose.project(&k, &Vector3::new(0.0, 10.0, 100.0)).unwrap();
    assert!(right.x > 50.0, "body right is image right");
}

#[test]
fn forward_and_zenith_mounts_look_along_their_axes() {
    let fwd = CameraMount::forward();
    assert!((fwd.camera_to_body(&Vector3::z()) - Vector3::x()).norm() < 1e-12);
    let up = CameraMount::zenith();
    assert!((up.camera_to_body(&Vector3::z()) + Vector3::z()).norm() < 1e-12);
}

#[test]
fn a_depressed_mount_spans_forward_to_nadir() {
    let same =
        |a: &CameraMount, b: &CameraMount| a.cam_from_body.angle_to(&b.cam_from_body) < 1e-12;
    assert!(same(&CameraMount::depressed(0.0), &CameraMount::forward()));
    assert!(same(
        &CameraMount::depressed(core::f64::consts::FRAC_PI_2),
        &CameraMount::nadir()
    ));
    let thirty = 30.0_f64.to_radians();
    let axis = CameraMount::depressed(thirty).camera_to_body(&Vector3::z());
    let expected = Vector3::new(thirty.cos(), 0.0, thirty.sin());
    assert!((axis - expected).norm() < 1e-12);
}

#[test]
fn lever_arm_moves_the_eye_in_world_axes() {
    let yaw = UnitQuaternion::from_euler_angles(0.0, 0.0, core::f64::consts::FRAC_PI_2);
    let mount = CameraMount::forward().with_lever_arm(Vector3::new(2.0, 0.0, 0.0));
    let pose = mount.camera_pose(&Vector3::new(1.0, 1.0, 1.0), &yaw);
    assert!((pose.eye - Vector3::new(1.0, 3.0, 1.0)).norm() < 1e-12);
}
