use super::*;
use crate::Navigator;
use crate::reference::terrain::tests::flight::{CLOCK, at, imager, suite};
use crate::{Alignment, ReferenceNavigator};
use navigate_contract::{AttitudeQuaternion, GeodeticPosition, NedVelocity, SymmetricCov3};

/// The solution of a navigator that was aligned with `attitude` and
/// `velocity` at 41.7° north, 1° west, 1 000 m.
#[cfg(test)]
fn solution(attitude: UnitQuaternion<f64>, velocity: NedVelocity) -> NavigationSolution {
    let q = attitude.quaternion();
    let suite = suite(imager(), CameraMount::nadir());
    let mut navigator = ReferenceNavigator::new(suite, CLOCK, None).unwrap();
    navigator
        .align(&Alignment {
            position: GeodeticPosition::new(41.7_f64.to_radians(), -1.0_f64.to_radians(), 1_000.0),
            velocity,
            attitude: AttitudeQuaternion::new(q.w, q.i, q.j, q.k),
            position_cov: SymmetricCov3::from_diagonal(9.0, 16.0, 4.0),
            velocity_cov: SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            attitude_cov: SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-5),
            at: at(0),
            imu_bias_prior: None,
        })
        .unwrap();
    navigator.solution(at(0)).unwrap()
}

#[test]
fn a_level_vehicle_with_a_nadir_camera_looks_down_with_the_nose_up_the_image() {
    let north = UnitQuaternion::identity();
    let belief = Belief::before(&solution(north, NedVelocity::new(0.0, 0.0, 0.0)), 0.0).unwrap();
    assert!((belief.sigma_h_m - 5.0).abs() < 1e-12);
    // The camera is one metre aft and half a metre below the reference.
    let mount = CameraMount::nadir().with_lever_arm(Vector3::new(-1.0, 0.0, 0.5));
    let (eye, pose) = belief.eye(&mount).unwrap();
    assert!(
        (pose.position - Vector3::new(0.0, -1.0, 999.5)).norm() < 1e-6,
        "{}",
        pose.position
    );
    assert!((eye[2] - 999.5).abs() < 1e-6);
    // Eye axes are right, up, back: back is the frame up, up is north.
    assert!((pose.orientation * Vector3::z() - Vector3::z()).norm() < 1e-9);
    assert!((pose.orientation * Vector3::y() - Vector3::y()).norm() < 1e-9);
    assert!((pose.orientation * Vector3::x() - Vector3::x()).norm() < 1e-9);
}

#[test]
fn the_lever_arm_comes_off_a_solved_pose_in_any_attitude_and_frame() {
    let attitude = UnitQuaternion::from_euler_angles(0.3, -0.15, 2.1);
    let belief = Belief::before(&solution(attitude, NedVelocity::new(0.0, 0.0, 0.0)), 0.0).unwrap();
    let mount =
        CameraMount::rotated_forward(0.1, -1.0, 0.4).with_lever_arm(Vector3::new(2.0, -0.7, 0.9));
    // A frame anchored 1.5 km away, as a keyframe or a resident window is.
    let frame = LocalFrame::wgs84_enu(41.71, -1.012).unwrap();
    let camera = belief.camera_in(frame, &mount).unwrap();
    let body = frame.local(belief.position).unwrap();
    assert!((camera.position - body).norm() > 2.0, "the lever arm is in");
    assert!((body_position(&camera, &mount) - body).norm() < 1e-6);
    // The optical axis is the mount axis turned by the attitude.
    let axis_ned = attitude * mount.camera_to_body(&Vector3::z());
    let axis_frame = camera.orientation * -Vector3::z();
    let expected = frame
        .local_rotation([belief.position[0], belief.position[1]])
        .unwrap()
        * Vector3::new(axis_ned.y, axis_ned.x, -axis_ned.z);
    assert!((axis_frame - expected).norm() < 1e-9);
}

#[test]
fn the_belief_goes_back_along_the_velocity() {
    let level = UnitQuaternion::identity();
    let moving = solution(level, NedVelocity::new(60.0, -20.0, 3.0));
    let now = Belief::before(&moving, 0.0).unwrap();
    let earlier = Belief::before(&moving, 0.1).unwrap();
    let frame = LocalFrame::wgs84_enu(now.position[0], now.position[1]).unwrap();
    let step = frame.local(earlier.position).unwrap() - frame.local(now.position).unwrap();
    // East, north, up: against the velocity, and up for a descent.
    assert!(
        (step - Vector3::new(2.0, -6.0, 0.3)).norm() < 1e-6,
        "{step}"
    );
    let mut without_attitude = moving;
    without_attitude.attitude = None;
    assert!(Belief::before(&without_attitude, 0.0).is_none());
}
