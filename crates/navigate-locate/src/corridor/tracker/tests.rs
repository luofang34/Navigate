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

#[test]
fn cold_dead_reckon_prior_propagates_drift_and_grows_search_sigma() {
    use navigate_match::{DEFAULT_KASHIN_SEED, HadamardProjector, SparseFeatures};

    let projector = HadamardProjector::new(DEFAULT_KASHIN_SEED);
    let mut state = CorridorTracker::new();
    state.set_cold_prior(ColdDeadReckonPrior::new(
        [100.0, 200.0, 120.0],
        25.0,
        250.0,
        None,
    ));
    let west_nadir = Matrix3::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0);
    let k_cal = Matrix3::new(1000.0, 0.0, 400.0, 0.0, 1000.0, 225.0, 0.0, 0.0, 1.0);
    let profile = VehicleDynamicsProfile::rotary_survey();

    let make_frame = |shift_v_px: f64| {
        let mut f = SparseFeatures::empty();
        for i in 0..24_usize {
            let u = 120.0 + (i % 6) as f64 * 90.0;
            let v = 60.0 + (i / 6) as f64 * 80.0 + shift_v_px;
            f.pixels.push([u, v]);
            f.model_pixels.push([u as f32, v as f32]);
            f.scores.push(0.95);
            let mut d = [0.0_f32; 64];
            for (k, slot) in d.iter_mut().enumerate() {
                let h = ((i as u64)
                    .wrapping_add(1)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15))
                    ^ ((k as u64)
                        .wrapping_add(1)
                        .wrapping_mul(0xBF58_476D_1CE4_E5B9));
                let mixed = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                *slot = if (mixed >> 63) == 0 { 1.0 } else { -1.0 };
            }
            f.descriptors.extend_from_slice(&d);
        }
        PreparedKashinQuery::from_features(&projector, &f).expect("valid query")
    };

    // Frame 1 establishes previous features; Frames 2..=6 each shift +15 px downward (+1.5 m West at 100 m AGL).
    let q0 = make_frame(0.0);
    state.record_odometry_step(&q0, &k_cal, Some(west_nadir), 100.0, 0.1, &profile);
    state.advance_unsolved_prepared(q0, 0.1, &profile);
    for step in 1..=5 {
        let q = make_frame(step as f64 * 15.0);
        state.record_odometry_step(&q, &k_cal, Some(west_nadir), 100.0, 0.1, &profile);
        state.advance_unsolved_prepared(q, 0.1, &profile);
    }

    let cp = state.cold_prior.expect("cold prior active");
    assert!(
        (cp.center_enu[0] - 92.5).abs() < 0.05,
        "expected center east ~92.5 m after 7.5 m West flight, got {}",
        cp.center_enu[0]
    );
    assert!((cp.center_enu[1] - 200.0).abs() < 0.05);
    assert!((cp.cumulative_path_m - 7.5).abs() < 0.05);
    assert!((cp.sigma_horiz_m - (25.0 + 0.20 * 7.5)).abs() < 0.02);
    assert!(cp.yaw_est_rad.is_some());

    // A cold anchor captured on Frame 1 at [100.0, 200.0, 120.0] propagates forward across Frames 2..=6 to ~92.5 m!
    let (prop_eye, _) = state.propagate_pose_from_calibrated(
        1,
        [100.0, 200.0, 120.0],
        west_nadir,
        &k_cal,
        100.0,
        &profile,
    );
    let (q_vo, path_m, _) =
        state.compute_vo_propagation_covariance(1, west_nadir, &k_cal, 100.0, &profile);
    assert!((prop_eye[0] - 92.5).abs() < 0.05);
    assert!((path_m - 7.5).abs() < 0.05);
    assert!(q_vo[(0, 0)] > 0.8);
}
