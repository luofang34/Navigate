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
        derotated_flow_px: None,
        dt_s: 1.0 / 30.0,
        has_world_pose: false,
    });
    state.frame_seq = 2;
    state.pose_steps.push_back(OdometryStep {
        frame_seq: 2,
        rot_step: step_rot,
        eye_delta: [4.0, -1.5, 0.25],
        derotated_flow_px: None,
        dt_s: 1.0 / 30.0,
        has_world_pose: false,
    });
    state.frame_seq = 3;
    state.pose_steps.push_back(OdometryStep {
        frame_seq: 3,
        rot_step: step_rot,
        eye_delta: [5.0, -2.5, 0.75],
        derotated_flow_px: None,
        dt_s: 1.0 / 30.0,
        has_world_pose: false,
    });

    let (eye, rot) = state.propagate_pose_from(1, [10.0, 20.0, 500.0], Matrix3::identity());
    assert!((eye[0] - 19.0).abs() < 1e-9);
    assert!((eye[1] - 16.0).abs() < 1e-9);
    assert!((eye[2] - 501.0).abs() < 1e-9);
    let expected_rot =
        Rotation3::from_axis_angle(&Vector3::z_axis(), 4.0_f64.to_radians()).into_inner();
    assert!((rot - expected_rot).norm() < 1e-9);
}

#[test]
fn propagate_pose_from_calibrated_projects_derotated_flow_with_solved_yaw_and_focal() {
    let mut state = CorridorTracker::new();
    state.frame_seq = 2;
    state.pose_steps.push_back(OdometryStep {
        frame_seq: 2,
        rot_step: Matrix3::identity(),
        eye_delta: [0.0, 0.0, 0.0],
        derotated_flow_px: Some([0.0, 20.0]),
        dt_s: 0.1,
        has_world_pose: false,
    });
    // West-heading (270 deg) nadir camera: top of image (-Y_cam) is West (-X_enu), +X_cam is North (+Y_enu), +Z_cam is Down (-Z_enu).
    let west_nadir = Matrix3::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0);
    let k_cal = Matrix3::new(1000.0, 0.0, 400.0, 0.0, 1000.0, 225.0, 0.0, 0.0, 1.0);
    let (eye, _) = state.propagate_pose_from_calibrated(
        1,
        [50.0, 80.0, 120.0],
        west_nadir,
        &k_cal,
        100.0,
        &VehicleDynamicsProfile::rotary_survey(),
    );
    // Ground features move +20 px downward (+Y_cam = East) -> drone moved 2.0 m West (-X_enu).
    assert!((eye[0] - 48.0).abs() < 1e-9);
    assert!((eye[1] - 80.0).abs() < 1e-9);
    assert!((eye[2] - 120.0).abs() < 1e-9);
}
