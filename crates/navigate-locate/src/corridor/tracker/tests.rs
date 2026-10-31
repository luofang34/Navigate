use super::*;
use nalgebra::Rotation3;

#[test]
fn propagate_pose_from_accumulates_interframe_steps_after_sent_frame() {
    let mut state = CorridorTracker::new();
    let step_rot =
        Rotation3::from_axis_angle(&Vector3::z_axis(), 2.0_f64.to_radians()).into_inner();
    state.frame_seq = 1;
    state.pose_steps.push_back(OdometryStep {
        frame_seq: 1,
        rot_step: step_rot,
        eye_delta: [100.0, 0.0, 0.0],
    });
    state.frame_seq = 2;
    state.pose_steps.push_back(OdometryStep {
        frame_seq: 2,
        rot_step: step_rot,
        eye_delta: [4.0, -1.5, 0.25],
    });
    state.frame_seq = 3;
    state.pose_steps.push_back(OdometryStep {
        frame_seq: 3,
        rot_step: step_rot,
        eye_delta: [5.0, -2.5, 0.75],
    });

    let (eye, rot) = state.propagate_pose_from(1, [10.0, 20.0, 500.0], Matrix3::identity());
    assert!((eye[0] - 19.0).abs() < 1e-9);
    assert!((eye[1] - 16.0).abs() < 1e-9);
    assert!((eye[2] - 501.0).abs() < 1e-9);
    let expected_rot =
        Rotation3::from_axis_angle(&Vector3::z_axis(), 4.0_f64.to_radians()).into_inner();
    assert!((rot - expected_rot).norm() < 1e-9);
}
