use super::*;

#[test]
fn dynamics_profiles_and_gates_expand_with_interval() {
    let survey = VehicleDynamicsProfile::rotary_survey();
    let fixed = VehicleDynamicsProfile::fixed_wing();
    let fast = VehicleDynamicsProfile::high_performance();

    assert_eq!(
        VehicleDynamicsProfile::from_name("rotary-survey"),
        Some(survey)
    );
    assert_eq!(VehicleDynamicsProfile::from_name("fixed-wing"), Some(fixed));
    assert_eq!(
        VehicleDynamicsProfile::from_name("high-performance"),
        Some(fast)
    );
    assert_eq!(VehicleDynamicsProfile::from_name("unknown"), None);

    let (dt_short, dt_long) = (0.033, 0.5);
    assert!(fast.max_displacement(dt_long) > fast.max_displacement(dt_short));
    assert!(fixed.max_displacement(dt_long) > survey.max_displacement(dt_long));
    assert!(survey.min_trace_cosine(dt_short) > survey.min_trace_cosine(dt_long));
}

#[test]
fn compute_dt_s_clamps_and_falls_back() {
    assert!((compute_dt_s(Some(10.5), Some(10.0), 30.0) - 0.5).abs() < 1e-9);
    assert!((compute_dt_s(Some(10.0), Some(10.5), 30.0) - (1.0 / 30.0)).abs() < 1e-9);
    assert!((compute_dt_s(None, None, 20.0) - 0.05).abs() < 1e-9);
    assert_eq!(compute_dt_s(Some(100.0), Some(0.0), 30.0), 60.0);
}

#[test]
fn kinematic_state_prediction_and_covariance_propagation() {
    let mut state = KinematicState::new([100.0, 200.0, 50.0], Matrix3::identity());
    state.vel_enu_mps = [600.0, -20.0, 5.0];

    let pred_30hz = state.predict_position(1.0 / 30.0);
    assert!((pred_30hz[0] - 120.0).abs() < 1e-6);

    let pred_2hz = state.predict_position(0.5);
    assert!((pred_2hz[0] - 400.0).abs() < 1e-6);

    state.propagate_covariance(0.5, 10.0, 0.1);
    let sigma_px = state.project_pixel_uncertainty(600.0, 600.0, 100.0);
    assert!(sigma_px > 0.0);

    let strategy_small = determine_search_strategy(10.0);
    assert!(matches!(
        strategy_small,
        TrackingSearchStrategy::RadiusSearch { .. }
    ));

    let strategy_large = determine_search_strategy(30.0);
    assert_eq!(strategy_large, TrackingSearchStrategy::GlobalFrustumMatch);
}

#[test]
fn mahalanobis_innovation_gate_accepts_consistent_and_rejects_outlier() {
    let state = KinematicState::new([0.0, 0.0, 100.0], Matrix3::identity());
    let geom_cov = Matrix3::identity() * 4.0;

    let consistent = [1.0, 0.5, 100.2];
    assert!(state.mahalanobis_innovation_gate(consistent, geom_cov, 11.345));

    let outlier = [500.0, 300.0, 100.0];
    assert!(!state.mahalanobis_innovation_gate(outlier, geom_cov, 11.345));
}

#[test]
fn vertical_velocity_gate_rules() {
    assert!(can_update_vertical_velocity(12, 10));
    assert!(can_update_vertical_velocity(5, 48));
    assert!(!can_update_vertical_velocity(11, 47));
}
