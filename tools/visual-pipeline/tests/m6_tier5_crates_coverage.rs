//! Tier 5 white-box adversarial coverage test suite for Milestone 6 crates.

use nalgebra::{Matrix3, Matrix6, SMatrix, SVector, Translation3, UnitQuaternion, Vector3};
use navigate_contract::{
    ClockDomainId, GeodeticPosition, MonotonicNanos, ObservationStamp, SensorClass,
    SourceComposition, SourceEpoch, SourceId, SymmetricCov3, WrappingSequence,
};
use navigate_fusion::{
    FusionConfig, IngestOutcome, NavigationFilter, Observation, ObservationValue, RejectionReason,
};
use navigate_place::{Descriptor, GRID, Record, Tokens, describe, rank};
use navigate_visual::{
    CameraPose, Estimate, EstimateQuality, FrameStamp, LocalFrame, MapRevision, SearchTier,
};
use navigate_visual_pipeline::corridor::{
    KinematicState, VisualCorridorSession,
    anchor::{
        multitier::resolve_search_tier,
        plausibility::{
            PhysicalPlausibilityConfig, PlausibilityError, verify_physical_plausibility,
        },
    },
    can_update_vertical_velocity,
    kinematics::CHI2_GATE_3DOF_99,
};
use navigate_visual_session::{
    AnchorObservation, ContinuityEpoch, FrameKey, MapReliability, Pose, RegionChange,
    RelativeMotion, ScaleSource, SessionError, StreamId, SurfaceModel,
};
use std::error::Error;

type TestResult = Result<(), Box<dyn Error>>;

fn test_map() -> MapRevision {
    MapRevision {
        release_id: "m6-coverage-map".into(),
        manifest_sha256: "c".repeat(64),
    }
}

fn make_anchor(frame: FrameKey, pos: [f64; 3]) -> Result<AnchorObservation, Box<dyn Error>> {
    let local_frame = LocalFrame::anchor_mercator(47.0, 11.0).map_err(|e| e.to_string())?;
    let pose = Pose::from_parts(
        Translation3::from(Vector3::from(pos)),
        UnitQuaternion::identity(),
    );
    Ok(AnchorObservation {
        frame,
        observation_sha256: format!("{:064x}", frame.index + 1),
        map: test_map(),
        local_frame,
        pose,
        geometry_covariance: Matrix6::identity() * 0.1,
        backend: "m6-coverage-matcher".into(),
        reliability: MapReliability {
            horizontal_m: 1.0,
            vertical_m: 2.0,
            imagery_age_years: Some(1.0),
            age_growth_m_per_year: 0.1,
            unknown_age_years: 5.0,
            surface: SurfaceModel::Surface,
            change: RegionChange::Unchanged,
        },
    })
}

fn make_estimate(frame: FrameKey, pos: [f64; 3], time_ns: u64) -> Result<Estimate, Box<dyn Error>> {
    let local_frame = LocalFrame::anchor_mercator(47.0, 11.0).map_err(|e| e.to_string())?;
    Ok(Estimate {
        stamp: FrameStamp {
            sequence: frame.index,
            capture_time_ns: time_ns,
        },
        observation_sha256: format!("{:064x}", frame.index + 1),
        map: test_map(),
        frame: local_frame,
        pose: CameraPose {
            position: Vector3::from(pos),
            orientation: UnitQuaternion::identity(),
        },
        quality: EstimateQuality {
            depth_matches: 60,
            inliers: 60,
            spatial_support: 25,
            reprojection_rms_px: 0.8,
            occupied_cells: 12,
            condition_number: 1.5,
        },
        geometry_covariance: SMatrix::<f64, 6, 6>::identity() * 0.1,
        backend: "m6-coverage-matcher".into(),
    })
}

fn step_session(
    session: &mut VisualCorridorSession,
    t_ns: u64,
    dx: f64,
    scale: &ScaleSource,
) -> Result<FrameKey, Box<dyn Error>> {
    session.register_frame(t_ns)?;
    let k = session.last_registered_key.ok_or("missing key")?;
    session
        .apply_motion_step(1.0, Pose::translation(dx, 0.0, 0.0), scale.clone())
        .map_err(|e| e.to_string())?;
    Ok(k)
}

#[test]
fn test_consensus_conflicting_anchors_tie_and_outvoting() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    let scale = ScaleSource::SeededMapPoints { map: test_map() };

    // Initial accepted anchor A0 at [1000.0, 1000.0, 100.0] on frame 0
    session.register_frame(1_000_000_000)?;
    let k0 = session.last_registered_key.ok_or("missing k0")?;
    let a0 = make_anchor(k0, [1000.0, 1000.0, 100.0])?;
    let e0 = make_estimate(k0, [1000.0, 1000.0, 100.0], 1_000_000_000)?;
    assert!(
        session
            .submit_and_ingest_estimate(1_000_000_000, a0, &e0)?
            .is_some()
    );

    // Advance 20.0m to frame 1. Candidate B1 disagrees with A0 -> remains pending
    let k1 = step_session(&mut session, 2_000_000_000, 20.0, &scale)?;
    let a1 = make_anchor(k1, [20.0, 0.0, 100.0])?;
    let e1 = make_estimate(k1, [20.0, 0.0, 100.0], 2_000_000_000)?;
    assert!(
        session
            .submit_and_ingest_estimate(2_000_000_000, a1, &e1)?
            .is_none()
    );

    // Advance 20.0m to frame 2. Candidate B2 agrees with B1 (20m >= 5m) -> relocates
    let k2 = step_session(&mut session, 3_000_000_000, 20.0, &scale)?;
    let a2 = make_anchor(k2, [40.0, 0.0, 100.0])?;
    let e2 = make_estimate(k2, [40.0, 0.0, 100.0], 3_000_000_000)?;
    assert!(
        session
            .submit_and_ingest_estimate(3_000_000_000, a2, &e2)?
            .is_some()
    );

    // Adversary attempts contradictory group C at [5000.0, 5000.0, 100.0]
    let k3 = step_session(&mut session, 4_000_000_000, 20.0, &scale)?;
    let a3 = make_anchor(k3, [5000.0, 5000.0, 100.0])?;
    let e3 = make_estimate(k3, [5000.0, 5000.0, 100.0], 4_000_000_000)?;
    assert!(
        session
            .submit_and_ingest_estimate(4_000_000_000, a3, &e3)?
            .is_none()
    );

    // Frame 4: Group C has 2 anchors, conflicting has 2 anchors -> tie blocked
    let k4 = step_session(&mut session, 5_000_000_000, 20.0, &scale)?;
    let a4 = make_anchor(k4, [5020.0, 5000.0, 100.0])?;
    let e4 = make_estimate(k4, [5020.0, 5000.0, 100.0], 5_000_000_000)?;
    assert!(
        session
            .submit_and_ingest_estimate(5_000_000_000, a4, &e4)?
            .is_none()
    );

    // Frame 5: Group C submits 3rd anchor -> outvotes active (3 > 2) and relocates
    let k5 = step_session(&mut session, 6_000_000_000, 20.0, &scale)?;
    let a5 = make_anchor(k5, [5040.0, 5000.0, 100.0])?;
    let e5 = make_estimate(k5, [5040.0, 5000.0, 100.0], 6_000_000_000)?;
    assert!(
        session
            .submit_and_ingest_estimate(6_000_000_000, a5, &e5)?
            .is_some()
    );
    Ok(())
}

#[test]
fn test_continuity_epoch_rollover_and_cross_epoch_isolation() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    assert_eq!(session.current_epoch, ContinuityEpoch(0));

    session.register_frame(1_000_000_000)?;
    let k0 = session.last_registered_key.ok_or("missing k0")?;
    assert_eq!(k0.continuity, ContinuityEpoch(0));

    session.register_frame(2_000_000_000)?;
    let k1 = session.last_registered_key.ok_or("missing k1")?;
    let scale = ScaleSource::External {
        sensor: "altimeter".into(),
    };
    let step1 = session.apply_motion_step(1.0, Pose::translation(5.0, 0.0, 0.0), scale.clone());
    assert!(step1.is_ok());

    session.advance_epoch();
    assert_eq!(session.current_epoch, ContinuityEpoch(1));

    session.register_frame(3_000_000_000)?;
    let k2 = session.last_registered_key.ok_or("missing k2")?;
    assert_eq!(k2.continuity, ContinuityEpoch(1));

    let cross_motion = RelativeMotion {
        from: k1,
        to: k2,
        from_to: Pose::translation(1.0, 0.0, 0.0),
        scale,
        from_observation_sha256: format!("{:064x}", k1.index + 1),
        to_observation_sha256: format!("{:064x}", k2.index + 1),
    };
    assert!(matches!(
        session.session.apply_motion(cross_motion),
        Err(SessionError::MotionOrder { .. })
    ));

    let max_epoch = ContinuityEpoch(u32::MAX);
    assert_eq!(max_epoch.0, u32::MAX);
    let key_max = FrameKey {
        stream: StreamId(1),
        continuity: max_epoch,
        index: 42,
    };
    assert_eq!(key_max.track(), (StreamId(1), max_epoch));
    Ok(())
}

#[test]
fn test_scale_source_switching_and_unknown_degradation() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    let map = test_map();

    session.register_frame(1_000_000_000)?;
    session.register_frame(2_000_000_000)?;

    let res1 = session.apply_motion_step(
        1.0,
        Pose::translation(2.0, 0.0, 0.0),
        ScaleSource::MapDepth { map: map.clone() },
    );
    assert!(res1.is_ok());

    session.register_frame(3_000_000_000)?;
    let res2 = session.apply_motion_step(
        1.0,
        Pose::translation(2.0, 0.0, 0.0),
        ScaleSource::SeededMapPoints { map },
    );
    assert!(res2.is_ok());

    session.register_frame(4_000_000_000)?;
    let res3 = session.apply_motion_step(
        1.0,
        Pose::translation(2.0, 0.0, 0.0),
        ScaleSource::External {
            sensor: "sonar".into(),
        },
    );
    assert!(res3.is_ok());

    session.register_frame(5_000_000_000)?;
    let epoch_before = session.current_epoch;
    let res4 =
        session.apply_motion_step(1.0, Pose::translation(2.0, 0.0, 0.0), ScaleSource::Unknown);
    assert!(matches!(res4, Err(SessionError::ScaleUnobservable { .. })));
    assert_eq!(session.current_epoch, ContinuityEpoch(epoch_before.0 + 1));
    Ok(())
}

#[test]
fn test_fisher_condition_boundaries_and_degenerate_geometry() -> TestResult {
    let cfg = PhysicalPlausibilityConfig::default();

    assert!(verify_physical_plausibility(100.0, 50.0, 0.99, 1e-6, 1.0, &cfg).is_ok());
    assert!(matches!(
        verify_physical_plausibility(100.0, 50.0, 0.99, 0.999e-6, 1.0, &cfg),
        Err(PlausibilityError::IllConditionedFisher { .. })
    ));

    let mut h_collinear = Matrix6::<f64>::zeros();
    for i in 1..=10 {
        let ju = SVector::<f64, 6>::from_row_slice(&[1.0 / (i as f64), 0.0, 0.0, 0.0, 0.0, 0.0]);
        h_collinear += ju * ju.transpose();
    }
    assert!(h_collinear.symmetric_eigen().eigenvalues.min() <= 1e-12);

    let s = Matrix6::<f64>::from_diagonal(&SVector::<f64, 6>::from_row_slice(&[
        1.0, 1.0, 1.0, 0.001, 0.001, 0.001,
    ]));
    let mut h_sim = Matrix6::<f64>::identity();
    h_sim[(3, 3)] = 1e6;
    h_sim[(4, 4)] = 1e6;
    h_sim[(5, 5)] = 1e6;
    let cond_unscaled =
        h_sim.symmetric_eigen().eigenvalues.max() / h_sim.symmetric_eigen().eigenvalues.min();
    let h_scaled = s * h_sim * s;
    let cond_scaled =
        h_scaled.symmetric_eigen().eigenvalues.max() / h_scaled.symmetric_eigen().eigenvalues.min();
    assert!(cond_unscaled >= 1e6);
    assert!((cond_scaled - 1.0).abs() < 1e-6);
    Ok(())
}

fn make_record(centre: [f64; 2], score: f32) -> Record {
    let mut vals = vec![0.0_f32; 384];
    vals[0] = score;
    vals[1] = (1.0 - score * score).sqrt();
    Record {
        centre,
        descriptor: vals,
    }
}

#[test]
fn test_multitier_dinov2_fallback_and_rank_spacing() -> TestResult {
    assert_eq!(resolve_search_tier(None, 50.0, 500.0), SearchTier::Global);
    assert_eq!(
        resolve_search_tier(Some(30.0), 50.0, 500.0),
        SearchTier::Local
    );
    assert_eq!(
        resolve_search_tier(Some(200.0), 50.0, 500.0),
        SearchTier::Region
    );
    assert_eq!(
        resolve_search_tier(Some(600.0), 50.0, 500.0),
        SearchTier::Global
    );

    let tokens = Tokens {
        width: 384,
        values: vec![0.5_f32; (1 + GRID * GRID) * 384],
    };
    let all_masked_cells = vec![false; GRID * GRID];
    let desc_zero = describe(&tokens, &all_masked_cells, None)?;
    assert_eq!(desc_zero.values.len(), 384);
    assert!(desc_zero.values.iter().all(|&v| v == 0.0));

    let mut q_vals = vec![0.0_f32; 384];
    q_vals[0] = 1.0;
    let query_desc = Descriptor { values: q_vals };

    let r1 = make_record([47.0, 11.0], 0.95);
    let r2 = make_record([47.00018, 11.0], 0.90);
    let r3 = make_record([47.0045, 11.0], 0.80);

    let records = vec![r1, r2, r3];
    let ranked = rank(
        std::slice::from_ref(&query_desc),
        &records,
        |_| true,
        5,
        100.0,
    );
    assert_eq!(ranked.len(), 2);
    assert!((ranked[0].score - 0.95).abs() < 1e-4);
    assert!((ranked[1].score - 0.80).abs() < 1e-4);

    let ranked_filtered = rank(
        std::slice::from_ref(&query_desc),
        &records,
        |c| c[0] <= 47.002,
        5,
        100.0,
    );
    assert_eq!(ranked_filtered.len(), 1);
    Ok(())
}

fn make_pos_obs(
    seq: u32,
    t_ns: u64,
    domain: u32,
    pos: GeodeticPosition,
    cov: SymmetricCov3,
) -> Observation {
    let stamp = ObservationStamp::new(
        SourceId::new(1),
        SourceEpoch::new(0),
        WrappingSequence::new(seq),
        MonotonicNanos::from_nanos(t_ns),
        ClockDomainId::new(domain),
    );
    Observation::new(
        stamp,
        ObservationValue::PositionFix {
            position: pos,
            covariance: cov,
        },
        SourceComposition::of(SensorClass::VisualLandmark),
    )
}

#[test]
fn test_mahalanobis_innovation_gate_noise_spikes_and_filter_rejection() -> TestResult {
    let mut kin = KinematicState::new([0.0, 0.0, 100.0], Matrix3::identity());
    kin.pos_cov_enu_m2 = Matrix3::identity() * 1.0;
    let geom_cov = Matrix3::identity() * 1.0;

    let eye_boundary = [4.763, 0.0, 100.0];
    assert!(kin.mahalanobis_innovation_gate(eye_boundary, geom_cov, CHI2_GATE_3DOF_99));

    let eye_outside = [4.765, 0.0, 100.0];
    assert!(!kin.mahalanobis_innovation_gate(eye_outside, geom_cov, CHI2_GATE_3DOF_99));

    let eye_spike = [1000.0, 0.0, 100.0];
    assert!(!kin.mahalanobis_innovation_gate(eye_spike, geom_cov, CHI2_GATE_3DOF_99));

    let singular_cov = Matrix3::zeros();
    kin.pos_cov_enu_m2 = Matrix3::zeros();
    assert!(!kin.mahalanobis_innovation_gate([1.0, 0.0, 100.0], singular_cov, CHI2_GATE_3DOF_99));

    assert!(!can_update_vertical_velocity(11, 47));
    assert!(can_update_vertical_velocity(12, 0));
    assert!(can_update_vertical_velocity(0, 48));

    let mut filter = NavigationFilter::new(FusionConfig::default(), ClockDomainId::new(1));
    let initial_pos = GeodeticPosition::new(47.0_f64.to_radians(), 11.0_f64.to_radians(), 500.0);
    let cov = SymmetricCov3::from_diagonal(1.0, 1.0, 1.0);

    let obs0 = make_pos_obs(0, 1_000_000_000, 1, initial_pos, cov);
    assert_eq!(
        filter.ingest(&obs0, MonotonicNanos::from_nanos(1_000_000_000)),
        IngestOutcome::Accepted
    );

    let spike_pos = GeodeticPosition::new(48.0_f64.to_radians(), 11.0_f64.to_radians(), 500.0);
    let obs1 = make_pos_obs(1, 2_000_000_000, 1, spike_pos, cov);
    assert!(matches!(
        filter.ingest(&obs1, MonotonicNanos::from_nanos(2_000_000_000)),
        IngestOutcome::Rejected(RejectionReason::InnovationGate { .. })
    ));

    let obs_bad_clock = make_pos_obs(2, 3_000_000_000, 99, initial_pos, cov);
    assert!(matches!(
        filter.ingest(&obs_bad_clock, MonotonicNanos::from_nanos(3_000_000_000)),
        IngestOutcome::Rejected(RejectionReason::ClockDomainMismatch { .. })
    ));
    Ok(())
}
