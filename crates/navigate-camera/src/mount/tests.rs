use super::*;
use core::f64::consts::FRAC_PI_2;

#[cfg(test)]
fn same(a: &CameraMount, b: &CameraMount) -> bool {
    a.cam_from_body.angle_to(&b.cam_from_body) < 1e-12
}

#[test]
fn a_turned_forward_camera_gives_the_named_mounts() {
    assert!(same(
        &CameraMount::rotated_forward(0.0, 0.0, 0.0),
        &CameraMount::forward()
    ));
    assert!(same(
        &CameraMount::rotated_forward(0.0, -FRAC_PI_2, 0.0),
        &CameraMount::nadir()
    ));
    assert!(same(
        &CameraMount::rotated_forward(0.0, -0.6, 0.0),
        &CameraMount::depressed(0.6)
    ));
}

#[test]
fn yaw_pitch_and_roll_turn_the_camera_in_the_aerospace_sense() {
    let optical = Vector3::z();
    let image_right = Vector3::x();
    // Yaw to the right wing: the optical axis is the body right, and the
    // image right is the body aft.
    let yawed = CameraMount::rotated_forward(0.0, 0.0, FRAC_PI_2);
    assert!((yawed.camera_to_body(&optical) - Vector3::y()).norm() < 1e-12);
    assert!((yawed.camera_to_body(&image_right) + Vector3::x()).norm() < 1e-12);
    // Pitch up: the optical axis rises, which is the negative body down.
    let pitched = CameraMount::rotated_forward(0.0, 0.3, 0.0);
    assert!(pitched.camera_to_body(&optical).z < 0.0);
    // Roll: the optical axis stays, and the image right goes down.
    let rolled = CameraMount::rotated_forward(0.2, 0.0, 0.0);
    assert!((rolled.camera_to_body(&optical) - Vector3::x()).norm() < 1e-12);
    assert!(rolled.camera_to_body(&image_right).z > 0.0);
    // The order is yaw, then pitch, then roll about the turned axes: a
    // side-looking camera pitched down looks down and to the right.
    let side = CameraMount::rotated_forward(0.0, -0.5, FRAC_PI_2);
    let axis = side.camera_to_body(&optical);
    assert!(
        axis.x.abs() < 1e-12 && axis.y > 0.0 && axis.z > 0.0,
        "{axis}"
    );
}

#[test]
fn the_lever_arm_moves_the_optical_centre_in_body_axes() {
    let mount = CameraMount::nadir().with_lever_arm(Vector3::new(-1.0, 0.0, 0.5));
    let yaw_east = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), FRAC_PI_2);
    let pose = mount.camera_pose(&Vector3::new(10.0, 20.0, 30.0), &yaw_east);
    // Body forward is the world +Y after the yaw, so one metre aft is -Y.
    assert!(
        (pose.eye - Vector3::new(10.0, 19.0, 30.5)).norm() < 1e-12,
        "{}",
        pose.eye
    );
}
