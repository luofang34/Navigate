//! Empirical stress and verification suite for pixel uncertainty projection,
//! dynamic search radius clamping, and multi-criteria interval triggers (Milestone 1).

use nalgebra::{Matrix3, Rotation3, Vector3};
use navigate_visual_pipeline::corridor::kinematics::{
    KinematicState, TrackingSearchStrategy, determine_search_strategy, should_force_anchor,
};
use navigate_visual_session::MapCell;
use std::collections::BTreeMap;

#[test]
fn test_sigma_px_focal_length_linear_scaling() {
    let mut state = KinematicState::new([0.0, 0.0, 100.0], Matrix3::identity());
    state.pos_cov_enu_m2 = Matrix3::from_diagonal(&Vector3::new(4.0, 4.0, 1.0));
    state.att_sigma_rad = 0.005;

    let depth = 50.0;
    let base_fx = 600.0;
    let base_fy = 600.0;
    let base_sigma = state.project_pixel_uncertainty(base_fx, base_fy, depth);
    assert!(base_sigma > 0.0);

    for multiplier in [0.5, 1.5, 2.0, 4.0, 8.0] {
        let test_fx = base_fx * multiplier;
        let test_fy = base_fy * multiplier;
        let sigma = state.project_pixel_uncertainty(test_fx, test_fy, depth);
        let expected = base_sigma * multiplier;
        let diff = (sigma - expected).abs();
        assert!(
            diff < 1e-9,
            "Linear scaling failed for mult={multiplier}: got {sigma}, expected {expected}"
        );
    }
}

#[test]
fn test_sigma_px_depth_attenuation_and_attitude_floor() {
    let mut state = KinematicState::new([0.0, 0.0, 100.0], Matrix3::identity());
    state.pos_cov_enu_m2 = Matrix3::from_diagonal(&Vector3::new(16.0, 16.0, 4.0));
    state.att_sigma_rad = 0.01;
    let (fx, fy) = (800.0, 800.0);

    let shallow_sigma = state.project_pixel_uncertainty(fx, fy, 10.0);
    let deep_sigma = state.project_pixel_uncertainty(fx, fy, 1000.0);
    assert!(
        shallow_sigma > deep_sigma,
        "Shallow depth must produce larger pixel uncertainty than deep"
    );

    let asymptotic_sigma = state.project_pixel_uncertainty(fx, fy, 1e7);
    let expected_att_floor = fx * state.att_sigma_rad;
    assert!(
        (asymptotic_sigma - expected_att_floor).abs() < 1e-4,
        "Deep limit must converge to attitude uncertainty floor"
    );

    let clamp_at_zero = state.project_pixel_uncertainty(fx, fy, 0.0);
    let clamp_at_one = state.project_pixel_uncertainty(fx, fy, 1.0);
    let clamp_at_neg = state.project_pixel_uncertainty(fx, fy, -50.0);
    assert_eq!(clamp_at_zero, clamp_at_one);
    assert_eq!(clamp_at_neg, clamp_at_one);
}

#[test]
fn test_sigma_px_rotational_covariance_projection() {
    let mut state = KinematicState::new([0.0, 0.0, 100.0], Matrix3::identity());
    state.pos_cov_enu_m2 = Matrix3::from_diagonal(&Vector3::new(100.0, 1.0, 1.0));
    state.att_sigma_rad = 0.0;
    let (fx, fy, depth) = (1000.0, 1000.0, 100.0);

    let sigma_identity = state.project_pixel_uncertainty(fx, fy, depth);
    assert!((sigma_identity - 100.0).abs() < 1e-6);

    let rot_yaw_90 =
        Rotation3::from_axis_angle(&Vector3::z_axis(), std::f64::consts::FRAC_PI_2).into_inner();
    state.rot_world_to_cam = rot_yaw_90;
    let sigma_rotated = state.project_pixel_uncertainty(fx, fy, depth);
    assert!(
        (sigma_rotated - 100.0).abs() < 1e-6,
        "Rotated uncertainty must remain 100.0 px along the swapped axis"
    );

    let sigma_aniso = state.project_pixel_uncertainty(1000.0, 500.0, depth);
    assert!((50.0..=100.0).contains(&sigma_aniso));
}

#[test]
fn test_search_radius_clamp_lower_bound() {
    let test_cases = [0.0, 0.1, 1.0, 5.0, 7.999, 8.0];
    for sigma in test_cases {
        let strategy = determine_search_strategy(sigma);
        match strategy {
            TrackingSearchStrategy::RadiusSearch { radius_px } => {
                assert!(
                    (radius_px - 24.0).abs() < 1e-9,
                    "Search radius for sigma={sigma} must clamp to 24.0 px, got {radius_px}"
                );
            }
            TrackingSearchStrategy::GlobalFrustumMatch => {
                panic!("sigma={sigma} must not trigger GlobalFrustumMatch");
            }
        }
    }
}

#[test]
fn test_search_radius_dynamic_scaling_window() {
    let test_sigmas = [8.1, 10.0, 12.5, 15.0, 18.0, 20.0, 64.0 / 3.0];
    for sigma in test_sigmas {
        let strategy = determine_search_strategy(sigma);
        match strategy {
            TrackingSearchStrategy::RadiusSearch { radius_px } => {
                let expected = 3.0 * sigma;
                assert!(
                    (radius_px - expected).abs() < 1e-9,
                    "Dynamic radius for sigma={sigma} must equal {expected}, got {radius_px}"
                );
                assert!((24.0..=64.0).contains(&radius_px));
            }
            TrackingSearchStrategy::GlobalFrustumMatch => {
                panic!("sigma={sigma} must remain in RadiusSearch");
            }
        }
    }
}

#[test]
fn test_search_radius_global_frustum_switching_boundary() {
    let exact_switch = 64.0 / 3.0;
    let strategy_at_bound = determine_search_strategy(exact_switch);
    assert_eq!(
        strategy_at_bound,
        TrackingSearchStrategy::RadiusSearch { radius_px: 64.0 },
        "At exact boundary 3*sigma=64, strategy must be RadiusSearch(64.0)"
    );

    let just_above = exact_switch + 1e-7;
    let strategy_above = determine_search_strategy(just_above);
    assert_eq!(
        strategy_above,
        TrackingSearchStrategy::GlobalFrustumMatch,
        "Just above boundary, strategy must switch to GlobalFrustumMatch"
    );

    for large_sigma in [21.5, 30.0, 50.0, 100.0, 1000.0] {
        let strat = determine_search_strategy(large_sigma);
        assert_eq!(
            strat,
            TrackingSearchStrategy::GlobalFrustumMatch,
            "sigma={large_sigma} must select GlobalFrustumMatch"
        );
    }
}

#[test]
fn test_search_strategy_failure_mode_inputs() {
    let strat_neg = determine_search_strategy(-10.0);
    assert_eq!(
        strat_neg,
        TrackingSearchStrategy::RadiusSearch { radius_px: 24.0 },
        "Negative sigma must clamp to minimum radius 24.0 px"
    );

    let strat_inf = determine_search_strategy(f64::INFINITY);
    assert_eq!(
        strat_inf,
        TrackingSearchStrategy::GlobalFrustumMatch,
        "Infinity must transition to GlobalFrustumMatch"
    );

    let strat_nan = determine_search_strategy(f64::NAN);
    assert_eq!(
        strat_nan,
        TrackingSearchStrategy::GlobalFrustumMatch,
        "NaN must safely fall back to GlobalFrustumMatch"
    );
}

#[test]
fn test_interval_trigger_frame_count() {
    let empty_cells = BTreeMap::new();
    assert!(!should_force_anchor(
        30,
        0,
        None,
        0.0,
        None,
        None,
        false,
        None,
        &empty_cells,
        500.0
    ));
    assert!(!should_force_anchor(
        30,
        29,
        None,
        0.0,
        None,
        None,
        false,
        None,
        &empty_cells,
        500.0
    ));
    assert!(should_force_anchor(
        30,
        30,
        None,
        0.0,
        None,
        None,
        false,
        None,
        &empty_cells,
        500.0
    ));
    assert!(should_force_anchor(
        30,
        31,
        None,
        0.0,
        None,
        None,
        false,
        None,
        &empty_cells,
        500.0
    ));
    assert!(!should_force_anchor(
        0,
        100,
        None,
        0.0,
        None,
        None,
        false,
        None,
        &empty_cells,
        500.0
    ));
}

#[test]
fn test_interval_trigger_elapsed_time() {
    let empty_cells = BTreeMap::new();
    assert!(!should_force_anchor(
        0,
        0,
        Some(2.0),
        1.99,
        None,
        None,
        false,
        None,
        &empty_cells,
        500.0
    ));
    assert!(should_force_anchor(
        0,
        0,
        Some(2.0),
        2.0,
        None,
        None,
        false,
        None,
        &empty_cells,
        500.0
    ));
    assert!(should_force_anchor(
        0,
        0,
        Some(2.0),
        2.05,
        None,
        None,
        false,
        None,
        &empty_cells,
        500.0
    ));
    assert!(!should_force_anchor(
        0,
        0,
        None,
        10.0,
        None,
        None,
        false,
        None,
        &empty_cells,
        500.0
    ));
}

#[test]
fn test_interval_trigger_horizontal_uncertainty() {
    let empty_cells = BTreeMap::new();
    let mut state = KinematicState::new([0.0, 0.0, 100.0], Matrix3::identity());

    state.pos_cov_enu_m2 = Matrix3::from_diagonal(&Vector3::new(16.0, 9.0, 100.0));
    assert!(!should_force_anchor(
        0,
        0,
        None,
        0.0,
        Some(5.01),
        Some(&state),
        false,
        None,
        &empty_cells,
        500.0
    ));
    let check = |thresh: Option<f64>, kin: Option<&KinematicState>| {
        should_force_anchor(
            0,
            0,
            None,
            0.0,
            thresh,
            kin,
            false,
            None,
            &empty_cells,
            500.0,
        )
    };
    assert!(check(Some(5.0), Some(&state)));
    assert!(check(Some(4.99), Some(&state)));

    state.pos_cov_enu_m2 = Matrix3::from_diagonal(&Vector3::new(100.0, 0.0, 0.0));
    assert!(check(Some(10.0), Some(&state)));

    state.pos_cov_enu_m2 = Matrix3::from_diagonal(&Vector3::new(0.0, 100.0, 0.0));
    assert!(check(Some(10.0), Some(&state)));
    assert!(!check(Some(5.0), None));
}

#[test]
fn test_interval_trigger_map_cell_boundary_crossing() {
    let mut visited = BTreeMap::new();
    let cell_origin = MapCell {
        map_manifest_sha256: "manifest_a".to_owned(),
        east: 0,
        north: 0,
    };
    visited.insert(cell_origin, "manifest_a".to_owned());

    let pose_inside = ([250.0, 250.0, 100.0], Matrix3::identity());
    let pose_east_cross = ([501.0, 250.0, 100.0], Matrix3::identity());
    let pose_north_cross = ([250.0, 501.0, 100.0], Matrix3::identity());
    let pose_neg_east_cross = ([-1.0, 250.0, 100.0], Matrix3::identity());
    let pose_neg_north_cross = ([250.0, -1.0, 100.0], Matrix3::identity());

    let cell_m = 500.0;
    for (pose, expected) in [
        (&pose_inside, false),
        (&pose_east_cross, true),
        (&pose_north_cross, true),
        (&pose_neg_east_cross, true),
        (&pose_neg_north_cross, true),
    ] {
        assert_eq!(
            should_force_anchor(
                0,
                0,
                None,
                0.0,
                None,
                None,
                true,
                Some(pose),
                &visited,
                cell_m
            ),
            expected
        );
    }

    assert!(!should_force_anchor(
        0,
        0,
        None,
        0.0,
        None,
        None,
        false,
        Some(&pose_east_cross),
        &visited,
        cell_m
    ));
    assert!(!should_force_anchor(
        0, 0, None, 0.0, None, None, true, None, &visited, cell_m
    ));
}

#[test]
fn test_interval_trigger_16_state_combinatorial_truth_table() {
    let mut visited = BTreeMap::new();
    let visited_cell = MapCell {
        map_manifest_sha256: "m".to_owned(),
        east: 0,
        north: 0,
    };
    visited.insert(visited_cell, "m".to_owned());

    let pose_inside = ([100.0, 100.0, 50.0], Matrix3::identity());
    let pose_outside = ([1500.0, 1500.0, 50.0], Matrix3::identity());

    let mut state_low_unc = KinematicState::new([0.0, 0.0, 50.0], Matrix3::identity());
    state_low_unc.pos_cov_enu_m2 = Matrix3::identity() * 1.0;

    let mut state_high_unc = KinematicState::new([0.0, 0.0, 50.0], Matrix3::identity());
    state_high_unc.pos_cov_enu_m2 = Matrix3::identity() * 100.0;

    for frame_cond in [false, true] {
        let (interval_frames, frames_since) = if frame_cond { (10, 10) } else { (10, 5) };
        for time_cond in [false, true] {
            let (interval_s, time_since) = if time_cond {
                (Some(2.0), 2.1)
            } else {
                (Some(2.0), 0.5)
            };
            for unc_cond in [false, true] {
                let kin_ref = if unc_cond {
                    Some(&state_high_unc)
                } else {
                    Some(&state_low_unc)
                };
                let unc_thresh = Some(10.0);
                for cell_cond in [false, true] {
                    let pose_ref = if cell_cond {
                        Some(&pose_outside)
                    } else {
                        Some(&pose_inside)
                    };
                    let outcome = should_force_anchor(
                        interval_frames,
                        frames_since,
                        interval_s,
                        time_since,
                        unc_thresh,
                        kin_ref,
                        true,
                        pose_ref,
                        &visited,
                        500.0,
                    );
                    let any_active = frame_cond || time_cond || unc_cond || cell_cond;
                    assert_eq!(
                        outcome, any_active,
                        "Truth table failed for (frame={frame_cond}, time={time_cond}, unc={unc_cond}, cell={cell_cond})"
                    );
                }
            }
        }
    }
}
