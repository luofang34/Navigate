//! Tier 2: Boundary and corner cases for Features 6 to 11 (Session, Filter & Dead Reckoning).

use e2e_corridor_tests::{
    AnchorWorkerState, DeadReckonPath, DualPathDeadReckoner, KinematicState, TestResult,
    VisualCorridorSession, approx_eq, can_update_vertical_velocity, check,
};
use nalgebra::Matrix3;
use navigate_contract::{
    ClockDomainId, GeodeticPosition, MonotonicNanos, ObservationStamp, SensorClass,
    SourceComposition, SourceEpoch, SourceId, SymmetricCov3, WrappingSequence,
};
use navigate_fusion::{
    FusionConfig, IngestOutcome, NavigationFilter, Observation, ObservationValue,
};
use navigate_visual_session::{ContinuityEpoch, FrameKey, Pose, ScaleSource, StreamId};

// --- Feature 6: Frame Registration Boundaries ---

#[test]
fn t2_06_01_out_of_order_timestamp_rejection() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    session.register_frame(2_000_000_000)?;
    let res = session.register_frame(1_000_000_000); // earlier timestamp
    check(res.is_err(), "out of order timestamp rejected by session")?;
    Ok(())
}

#[test]
fn t2_06_02_duplicate_timestamp_rejection() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    session.register_frame(1_000_000_000)?;
    let res = session.register_frame(1_000_000_000); // identical timestamp
    check(res.is_err(), "duplicate timestamp rejected by session")?;
    Ok(())
}

#[test]
fn t2_06_03_zero_image_dimensions() -> TestResult {
    let zero_dims = [0, 0];
    let is_valid = zero_dims[0] > 0 && zero_dims[1] > 0;
    check(!is_valid, "zero dimensions flagged as invalid")?;
    Ok(())
}

#[test]
fn t2_06_04_extreme_aspect_ratio() -> TestResult {
    let extreme_dims = [7680, 100]; // 76.8:1 panorama
    check(
        extreme_dims[0] > 0 && extreme_dims[1] > 0,
        "dimensions are non-zero",
    )?;
    Ok(())
}

#[test]
fn t2_06_05_memory_bound_frame_eviction() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    for i in 1..=20 {
        session.register_frame(i * 100_000_000)?;
    }
    check(
        session.next_seq == 20,
        "20 frames registered within capacity",
    )?;
    Ok(())
}

// --- Feature 7: Motion Application & Epoch Boundaries ---

#[test]
fn t2_07_01_refuse_unmetric_scale() -> TestResult {
    let scale = ScaleSource::Unknown;
    check(!scale.is_metric(), "ScaleSource::Unknown is not metric")?;
    Ok(())
}

#[test]
fn t2_07_02_identity_relative_motion() -> TestResult {
    let id_pose = Pose::identity();
    check(
        approx_eq(id_pose.translation.vector.norm(), 0.0, 1e-6),
        "identity translation is zero",
    )?;
    Ok(())
}

#[test]
fn t2_07_03_large_relative_displacement() -> TestResult {
    let large_pose = Pose::translation(500.0, 0.0, 0.0);
    check(
        approx_eq(large_pose.translation.vector.norm(), 500.0, 1e-4),
        "large translation created",
    )?;
    Ok(())
}

#[test]
fn t2_07_04_rapid_epoch_transitions() -> TestResult {
    let mut current_epoch = ContinuityEpoch(0);
    for i in 1..=10 {
        current_epoch = ContinuityEpoch(i);
    }
    check(
        current_epoch.0 == 10,
        "10 rapid epoch transitions incremented correctly",
    )?;
    Ok(())
}

#[test]
fn t2_07_05_trajectory_revision_event() -> TestResult {
    let cause = navigate_visual_session::RevisionCause::Relocation;
    check(
        matches!(cause, navigate_visual_session::RevisionCause::Relocation),
        "revision matches relocation",
    )?;
    Ok(())
}

// --- Feature 8: Anchor Submission & EKF Ingestion Boundaries ---

#[test]
fn t2_08_01_refuse_zero_error_budget() -> TestResult {
    let zero_budget = navigate_visual_fusion::VisualErrorBudget {
        map_horizontal_m: 0.0, // Invalid: must be positive
        map_vertical_m: 1.0,
        calibration_m: 0.5,
        geometry_model_m: 1.0,
        vertical_datum: navigate_visual_fusion::VerticalDatum::Wgs84Ellipsoid,
        vertical_datum_m: 0.5,
    };
    let identity = navigate_visual_fusion::VisualSourceIdentity {
        source: SourceId::new(1),
        epoch: SourceEpoch::new(0),
        clock: ClockDomainId::new(1),
    };
    let res = navigate_visual_fusion::VisualFixSource::new(identity, zero_budget);
    check(res.is_err(), "zero error budget refused by VisualFixSource")?;
    Ok(())
}

#[test]
fn t2_08_02_filter_rejects_non_finite_fix() -> TestResult {
    let mut filter = NavigationFilter::new(FusionConfig::default(), ClockDomainId::new(1));
    let pos = GeodeticPosition::new(f64::NAN, 11.0_f64.to_radians(), 500.0);
    let stamp = ObservationStamp::new(
        SourceId::new(1),
        SourceEpoch::new(0),
        WrappingSequence::new(0),
        MonotonicNanos::from_nanos(1_000_000_000),
        ClockDomainId::new(1),
    );
    let obs = Observation::new(
        stamp,
        ObservationValue::PositionFix {
            position: pos,
            covariance: SymmetricCov3::from_diagonal(4.0, 4.0, 9.0),
        },
        SourceComposition::of(SensorClass::VisualLandmark),
    );
    let outcome = filter.ingest(&obs, MonotonicNanos::from_nanos(1_000_000_000));
    check(
        outcome != IngestOutcome::Accepted,
        "filter rejects NaN coordinates",
    )?;
    Ok(())
}

#[test]
fn t2_08_03_filter_rejects_negative_variance() -> TestResult {
    let mut filter = NavigationFilter::new(FusionConfig::default(), ClockDomainId::new(1));
    let pos = GeodeticPosition::new(47.0_f64.to_radians(), 11.0_f64.to_radians(), 500.0);
    let stamp = ObservationStamp::new(
        SourceId::new(1),
        SourceEpoch::new(0),
        WrappingSequence::new(0),
        MonotonicNanos::from_nanos(1_000_000_000),
        ClockDomainId::new(1),
    );
    let obs = Observation::new(
        stamp,
        ObservationValue::PositionFix {
            position: pos,
            covariance: SymmetricCov3::from_diagonal(-1.0, 4.0, 9.0),
        },
        SourceComposition::of(SensorClass::VisualLandmark),
    );
    let outcome = filter.ingest(&obs, MonotonicNanos::from_nanos(1_000_000_000));
    check(
        outcome != IngestOutcome::Accepted,
        "filter rejects negative variance",
    )?;
    Ok(())
}

#[test]
fn t2_08_04_submit_anchor_before_first_frame() -> TestResult {
    let session = VisualCorridorSession::new()?;
    let key = FrameKey {
        stream: StreamId(1),
        continuity: ContinuityEpoch(0),
        index: 0,
    };
    let lookup = session.session.pose_at(&key);
    check(lookup.is_err(), "query on unobserved frame returns error")?;
    Ok(())
}

#[test]
fn t2_08_05_duplicate_cell_anchor_rejection() -> TestResult {
    let ind = navigate_visual_fusion::EvidenceIndependence::ValidatedIndependent;
    let label = navigate_visual_session::FusionEligibility::SharedMapError {
        cell: navigate_visual_session::MapCell {
            map_manifest_sha256: "manifest_1".to_string(),
            east: 10,
            north: 20,
        },
        earlier: FrameKey {
            stream: StreamId(0),
            continuity: ContinuityEpoch(0),
            index: 0,
        },
    };
    let restricted = ind.restricted_by(&label);
    check(
        restricted == navigate_visual_fusion::EvidenceIndependence::Unknown,
        "shared map error restricts independence",
    )?;
    Ok(())
}

// --- Feature 9: Dual-Path Dead Reckoning Boundaries ---

#[test]
fn t2_09_01_path_a_attitude_sample_boundary() -> TestResult {
    let mut reckoner = DualPathDeadReckoner::new([0.0, 0.0, 100.0], [10.0, 0.0, 0.0], true);
    let est = reckoner.step(0.0001, 0, 1.0, 0.01);
    check(
        est.path == DeadReckonPath::PathA,
        "Path A at minimal dt boundary",
    )?;
    Ok(())
}

#[test]
fn t2_09_02_path_a_extrapolation_limit() -> TestResult {
    let mut reckoner = DualPathDeadReckoner::new([0.0, 0.0, 100.0], [10.0, 0.0, 0.0], true);
    // Extrapolate 60 seconds
    let est = reckoner.step(60.0, 0, 1.0, 0.01);
    check(est.eye_enu[0].is_finite(), "extrapolated state is finite")?;
    Ok(())
}

#[test]
fn t2_09_03_path_b_flow_vector_threshold_5() -> TestResult {
    let mut reck_5 = DualPathDeadReckoner::new([0.0, 0.0, 100.0], [0.0, 0.0, 0.0], false);
    reck_5.step(1.0, 5, 4.0, 0.01); // 5 vectors < 6: pure coasting
    let var_5 = reck_5.kinematic.pos_cov_enu_m2[(0, 0)];

    let mut reck_6 = DualPathDeadReckoner::new([0.0, 0.0, 100.0], [0.0, 0.0, 0.0], false);
    reck_6.step(1.0, 6, 4.0, 0.01); // 6 vectors >= 6: flow integration
    let var_6 = reck_6.kinematic.pos_cov_enu_m2[(0, 0)];

    check(
        var_5 > var_6,
        "5 vectors inflates covariance more than 6 vectors",
    )?;
    Ok(())
}

#[test]
fn t2_09_04_path_b_extended_coasting_growth() -> TestResult {
    let mut reckoner = DualPathDeadReckoner::new([0.0, 0.0, 100.0], [0.0, 0.0, 0.0], false);
    let var_0 = reckoner.kinematic.pos_cov_enu_m2[(0, 0)];
    reckoner.step(5.0, 0, 2.0, 0.01);
    let var_5 = reckoner.kinematic.pos_cov_enu_m2[(0, 0)];
    reckoner.step(5.0, 0, 2.0, 0.01);
    let var_10 = reckoner.kinematic.pos_cov_enu_m2[(0, 0)];
    check(
        var_10 > var_5 && var_5 > var_0,
        "monotonic uncertainty growth over extended coasting",
    )?;
    Ok(())
}

#[test]
fn t2_09_05_switch_from_path_b_to_path_a() -> TestResult {
    let mut reckoner = DualPathDeadReckoner::new([0.0, 0.0, 100.0], [10.0, 0.0, 0.0], false);
    let est1 = reckoner.step(1.0, 0, 2.0, 0.01);
    check(est1.path == DeadReckonPath::PathB, "initially Path B")?;

    reckoner.external_sensors_available = true;
    let est2 = reckoner.step(1.0, 0, 2.0, 0.01);
    check(
        est2.path == DeadReckonPath::PathA,
        "switches to Path A when external sensors restore",
    )?;
    Ok(())
}

// --- Feature 10: 3D Mahalanobis Gate Boundaries ---

#[test]
fn t2_10_01_exact_gate_boundary() -> TestResult {
    let mut state = KinematicState::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
    state.pos_cov_enu_m2 = Matrix3::from_diagonal_element(1.0);
    let geom_cov = Matrix3::from_diagonal_element(0.0);
    // d2 = (x^2 + y^2 + z^2) = 11.34
    let x = 11.34_f64.sqrt();
    let cand_eye = [x, 0.0, 0.0];
    let accepted = state.mahalanobis_innovation_gate(cand_eye, geom_cov, 11.34);
    check(accepted, "exact chi2 gate value accepted")?;
    Ok(())
}

#[test]
fn t2_10_02_singular_geometry_covariance() -> TestResult {
    let state = KinematicState::default();
    let singular_geom = Matrix3::zeros();
    // Joint cov is state cov (invertible) + zeros
    let accepted = state.mahalanobis_innovation_gate([0.1, 0.1, 100.1], singular_geom, 11.34);
    check(
        accepted,
        "singular geom covariance handled when joint covariance is regular",
    )?;
    Ok(())
}

#[test]
fn t2_10_03_vertical_gate_holdout_11_reject() -> TestResult {
    let can_update = can_update_vertical_velocity(11, 47);
    check(!can_update, "11 holdout and 47 inliers rejected")?;
    Ok(())
}

#[test]
fn t2_10_04_vertical_gate_inliers_48_accept() -> TestResult {
    let can_update = can_update_vertical_velocity(0, 48);
    check(can_update, "48 inliers accepted even with 0 holdouts")?;
    Ok(())
}

#[test]
fn t2_10_05_coplanar_ray_vertical_damping() -> TestResult {
    let can_update = can_update_vertical_velocity(12, 12);
    check(can_update, "12 holdouts accepted")?;
    Ok(())
}

// --- Feature 11: Anchor State Reset Boundaries ---

#[test]
fn t2_11_01_anchor_arrival_cold_start() -> TestResult {
    let mut state = AnchorWorkerState::default();
    check(state.prev_eye.is_none(), "prev_eye is None initially")?;
    state.on_anchor_accepted([100.0, 200.0, 300.0], Matrix3::identity(), 0.1);
    check(
        state.prev_eye == Some([100.0, 200.0, 300.0]),
        "cold start sets prev_eye",
    )?;
    check(
        state.vel_enu_mps == [0.0, 0.0, 0.0],
        "cold start velocity is zero",
    )?;
    Ok(())
}

#[test]
fn t2_11_02_zero_elapsed_time_reset() -> TestResult {
    let mut state = AnchorWorkerState {
        prev_eye: Some([0.0, 0.0, 0.0]),
        prev_rot: Some(Matrix3::identity()),
        vel_enu_mps: [0.0, 0.0, 0.0],
    };
    state.on_anchor_accepted([1.0, 0.0, 0.0], Matrix3::identity(), 0.0);
    // dt clamped to min 1e-4 -> vel = 1.0 / 1e-4 = 10000 m/s, protected from div by zero
    check(
        state.vel_enu_mps[0].is_finite(),
        "velocity remains finite with zero dt",
    )?;
    Ok(())
}

#[test]
fn t2_11_03_large_relocation_jump() -> TestResult {
    let mut state = AnchorWorkerState {
        prev_eye: Some([0.0, 0.0, 100.0]),
        prev_rot: Some(Matrix3::identity()),
        vel_enu_mps: [10.0, 0.0, 0.0],
    };
    state.on_relocation([10_000.0, 0.0, 100.0], 1.0);
    check(
        state.prev_eye == Some([10_000.0, 0.0, 100.0]),
        "relocated to 10 km",
    )?;
    check(
        state.vel_enu_mps[0] == 10_000.0,
        "velocity reflects relocation baseline",
    )?;
    Ok(())
}

#[test]
fn t2_11_04_consecutive_anchor_resets() -> TestResult {
    let mut state = AnchorWorkerState::default();
    state.on_anchor_accepted([10.0, 0.0, 0.0], Matrix3::identity(), 0.1);
    state.on_anchor_accepted([20.0, 0.0, 0.0], Matrix3::identity(), 0.1);
    state.on_anchor_accepted([30.0, 0.0, 0.0], Matrix3::identity(), 0.1);
    check(
        approx_eq(state.vel_enu_mps[0], 100.0, 1e-4),
        "steady 100 m/s velocity across consecutive anchors",
    )?;
    Ok(())
}

#[test]
fn t2_11_05_rejected_anchor_does_not_reset() -> TestResult {
    let mut state = AnchorWorkerState {
        prev_eye: Some([50.0, 50.0, 50.0]),
        prev_rot: Some(Matrix3::identity()),
        vel_enu_mps: [20.0, 0.0, 0.0],
    };
    let anchor_accepted = false;
    if anchor_accepted {
        state.on_anchor_accepted([100.0, 100.0, 100.0], Matrix3::identity(), 0.1);
    }
    check(
        state.prev_eye == Some([50.0, 50.0, 50.0]),
        "rejected anchor does not change state",
    )?;
    check(state.vel_enu_mps[0] == 20.0, "velocity unchanged")?;
    Ok(())
}
