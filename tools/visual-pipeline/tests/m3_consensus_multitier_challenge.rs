//! Empirical challenge test suite for Milestone 3 (Features 13 and 14).
//!
//! This test suite challenges:
//! - Consensus trajectory relocation across agreeing map anchors.
//! - Sub-five-meter odometry separation gating.
//! - Anchor worker velocity recomputation without stale velocity carryover.
//! - Multi-tier search resolution boundary conditions.
//! - DINOv2 canvas mapping, descriptor pooling, and region ranking.

use image::{DynamicImage, Rgb, RgbImage};
use nalgebra::{Matrix3, Matrix6, SMatrix, Translation3, UnitQuaternion, Vector3};
use navigate_contract::MonotonicNanos;
use navigate_match::{CorridorLandmarkBank, CorridorTileStore};
use navigate_place::{CANVAS_PX, Canvas, Descriptor, GRID, Record, Tokens};
use navigate_visual::{
    CameraPose, Estimate, EstimateQuality, FrameStamp, LocalFrame, MapRevision, SearchConfig,
    SearchPrior, SearchTier,
};
use navigate_visual_pipeline::corridor::{
    AnchorWorkerState, VisualCorridorSession,
    anchor::multitier::{
        MultiTierSearchConfig, SearchTierMatcher, descriptor_from_tokens_and_canvas,
        ground_crop_to_canvas, load_candidate_tiles, rank_candidate_regions, resolve_search_tier,
        resolve_search_tier_from_prior,
    },
};
use navigate_visual_session::{
    AnchorObservation, FrameKey, MapReliability, Pose, RegionChange, ScaleSource, SurfaceModel,
};
use std::error::Error;

type TestResult = Result<(), Box<dyn Error>>;

fn test_map() -> MapRevision {
    MapRevision {
        release_id: "test-map".into(),
        manifest_sha256: "b".repeat(64),
    }
}

fn make_test_anchor(frame: FrameKey, pos: [f64; 3]) -> Result<AnchorObservation, Box<dyn Error>> {
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
        backend: "test-matcher".into(),
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

fn make_test_estimate(
    frame: FrameKey,
    pos: [f64; 3],
    time_ns: u64,
) -> Result<Estimate, Box<dyn Error>> {
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
            depth_matches: 50,
            inliers: 50,
            spatial_support: 20,
            reprojection_rms_px: 1.0,
            occupied_cells: 10,
            condition_number: 1.0,
        },
        geometry_covariance: SMatrix::<f64, 6, 6>::identity() * 0.1,
        backend: "test-matcher".into(),
    })
}

#[test]
fn test_consensus_relocation_full_lifecycle() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    let scale = ScaleSource::SeededMapPoints { map: test_map() };

    // Register initial frame 0 at t = 1.0s
    session.register_frame(1_000_000_000)?;
    let k0 = session.last_registered_key.ok_or("missing k0")?;

    // Ingest initial anchor at [1000.0, 1000.0, 100.0] (false / spoofed prior)
    let a0 = make_test_anchor(k0, [1000.0, 1000.0, 100.0])?;
    let e0 = make_test_estimate(k0, [1000.0, 1000.0, 100.0], 1_000_000_000)?;
    let outcome0 = session.submit_and_ingest_estimate(1_000_000_000, a0, &e0)?;
    assert!(outcome0.is_some(), "Initial anchor must be accepted");

    // Advance vehicle 10.0m along X to frame 1 at t = 2.0s
    session.register_frame(2_000_000_000)?;
    let k1 = session.last_registered_key.ok_or("missing k1")?;
    session
        .apply_motion_step(1.0, Pose::translation(10.0, 0.0, 0.0), scale.clone())
        .map_err(|e| e.to_string())?;

    // Matcher finds frame 1 is at true coordinate [10.0, 0.0, 100.0]
    let a1 = make_test_anchor(k1, [10.0, 0.0, 100.0])?;
    let e1 = make_test_estimate(k1, [10.0, 0.0, 100.0], 2_000_000_000)?;
    let outcome1 = session.submit_and_ingest_estimate(2_000_000_000, a1, &e1)?;
    assert!(
        outcome1.is_none(),
        "Single contradictory anchor must remain pending"
    );

    // Advance vehicle another 10.0m along X to frame 2 at t = 3.0s (separation = 10.0m >= 5.0m)
    session.register_frame(3_000_000_000)?;
    let k2 = session.last_registered_key.ok_or("missing k2")?;
    session
        .apply_motion_step(1.0, Pose::translation(10.0, 0.0, 0.0), scale)
        .map_err(|e| e.to_string())?;

    // Matcher finds frame 2 is at true coordinate [20.0, 0.0, 100.0] (agrees with frame 1)
    let a2 = make_test_anchor(k2, [20.0, 0.0, 100.0])?;
    let e2 = make_test_estimate(k2, [20.0, 0.0, 100.0], 3_000_000_000)?;
    let outcome2 = session.submit_and_ingest_estimate(3_000_000_000, a2, &e2)?;
    assert!(
        outcome2.is_some(),
        "Agreeing anchor with >= 5.0m separation must trigger relocation"
    );

    // Verify dead reckoner kinematic state was re-centered cleanly to relocated position
    assert!((session.dead_reckoner.kinematic.eye_enu[0] - 20.0).abs() < 1e-4);
    assert!((session.dead_reckoner.kinematic.eye_enu[1] - 0.0).abs() < 1e-4);
    assert!((session.dead_reckoner.kinematic.eye_enu[2] - 100.0).abs() < 1e-4);
    assert_eq!(session.dead_reckoner.kinematic.vel_enu_mps, [0.0, 0.0, 0.0]);

    // Verify filter is initialized with relocated fix and produces valid solution
    assert!(
        session.filter.is_initialized(),
        "Filter must be initialized with relocated position fix (found uninitialized due to handle_relocation resetting filter after ingest)"
    );
    let sol = session
        .filter
        .tick(MonotonicNanos::from_nanos(3_000_000_000));
    assert!(
        sol.is_some(),
        "Filter must have valid solution post-relocation"
    );
    Ok(())
}

#[test]
fn test_consensus_relocation_sub_five_meter_separation_remains_pending() -> TestResult {
    let mut session = VisualCorridorSession::new()?;
    let scale = ScaleSource::SeededMapPoints { map: test_map() };

    session.register_frame(1_000_000_000)?;
    let k0 = session.last_registered_key.ok_or("missing k0")?;
    let a0 = make_test_anchor(k0, [1000.0, 1000.0, 100.0])?;
    let e0 = make_test_estimate(k0, [1000.0, 1000.0, 100.0], 1_000_000_000)?;
    session.submit_and_ingest_estimate(1_000_000_000, a0, &e0)?;

    // Advance 10.0m to frame 1
    session.register_frame(2_000_000_000)?;
    let k1 = session.last_registered_key.ok_or("missing k1")?;
    session
        .apply_motion_step(1.0, Pose::translation(10.0, 0.0, 0.0), scale.clone())
        .map_err(|e| e.to_string())?;

    // Contradictory anchor 1
    let a1 = make_test_anchor(k1, [10.0, 0.0, 100.0])?;
    let e1 = make_test_estimate(k1, [10.0, 0.0, 100.0], 2_000_000_000)?;
    let outcome1 = session.submit_and_ingest_estimate(2_000_000_000, a1, &e1)?;
    assert!(outcome1.is_none());

    // Advance only 2.0m to frame 2 (< 5.0m consensus threshold)
    session.register_frame(3_000_000_000)?;
    let k2 = session.last_registered_key.ok_or("missing k2")?;
    session
        .apply_motion_step(1.0, Pose::translation(2.0, 0.0, 0.0), scale.clone())
        .map_err(|e| e.to_string())?;

    // Anchor 2 agrees with anchor 1, but separation is only 2.0m < 5.0m
    let a2 = make_test_anchor(k2, [12.0, 0.0, 100.0])?;
    let e2 = make_test_estimate(k2, [12.0, 0.0, 100.0], 3_000_000_000)?;
    let outcome2 = session.submit_and_ingest_estimate(3_000_000_000, a2, &e2)?;
    assert!(
        outcome2.is_none(),
        "Separation of 2.0m < 5.0m must not trigger relocation"
    );

    // Advance another 4.0m to frame 3 (total separation from frame 1 is 6.0m >= 5.0m)
    session.register_frame(4_000_000_000)?;
    let k3 = session.last_registered_key.ok_or("missing k3")?;
    session
        .apply_motion_step(1.0, Pose::translation(4.0, 0.0, 0.0), scale)
        .map_err(|e| e.to_string())?;

    // Anchor 3 agrees with anchor 1 (+ 6.0m)
    let a3 = make_test_anchor(k3, [16.0, 0.0, 100.0])?;
    let e3 = make_test_estimate(k3, [16.0, 0.0, 100.0], 4_000_000_000)?;
    let outcome3 = session.submit_and_ingest_estimate(4_000_000_000, a3, &e3)?;
    assert!(
        outcome3.is_some(),
        "Total separation of 6.0m >= 5.0m must trigger relocation"
    );
    Ok(())
}

#[test]
fn test_relocation_velocity_recomputation_lifecycle() -> TestResult {
    let mut state = AnchorWorkerState::default();

    // Initial anchor accepted: velocity is zero because no prior position exists
    state.on_anchor_accepted([100.0, 200.0, 50.0], Matrix3::identity(), 0.1);
    assert_eq!(state.prev_eye, Some([100.0, 200.0, 50.0]));
    assert_eq!(state.vel_enu_mps, [0.0, 0.0, 0.0]);

    // Motion over dt = 0.5s: displacement [10, -5, 2] -> velocity = [20, -10, 4]
    state.on_anchor_accepted([110.0, 195.0, 52.0], Matrix3::identity(), 0.5);
    assert_eq!(state.prev_eye, Some([110.0, 195.0, 52.0]));
    assert!((state.vel_enu_mps[0] - 20.0).abs() < 1e-6);
    assert!((state.vel_enu_mps[1] - (-10.0)).abs() < 1e-6);
    assert!((state.vel_enu_mps[2] - 4.0).abs() < 1e-6);

    // Track loss occurs: state resets and velocity becomes zero
    state.on_track_loss();
    assert!(state.prev_eye.is_none());
    assert_eq!(state.vel_enu_mps, [0.0, 0.0, 0.0]);

    // Anchor accepted after track loss: prev_eye was None, so velocity is zero
    state.on_anchor_accepted([500.0, 500.0, 100.0], Matrix3::identity(), 1.0);
    assert_eq!(
        state.vel_enu_mps,
        [0.0, 0.0, 0.0],
        "Velocity must not carry over across track loss"
    );
    assert_eq!(state.prev_eye, Some([500.0, 500.0, 100.0]));

    // Subsequent motion recomputes velocity cleanly
    state.on_anchor_accepted([504.0, 502.0, 99.0], Matrix3::identity(), 0.2);
    assert!((state.vel_enu_mps[0] - 20.0).abs() < 1e-6);
    assert!((state.vel_enu_mps[1] - 10.0).abs() < 1e-6);
    assert!((state.vel_enu_mps[2] - (-5.0)).abs() < 1e-6);

    // Relocation event: displacement over dt_s recomputes vel_enu_mps
    state.on_relocation([604.0, 552.0, 109.0], 2.0);
    assert_eq!(state.prev_eye, Some([604.0, 552.0, 109.0]));
    assert!((state.vel_enu_mps[0] - 50.0).abs() < 1e-6);
    assert!((state.vel_enu_mps[1] - 25.0).abs() < 1e-6);
    assert!((state.vel_enu_mps[2] - 5.0).abs() < 1e-6);

    // Relocation after track loss without prior eye: velocity is zero
    state.on_track_loss();
    state.on_relocation([800.0, 800.0, 100.0], 1.0);
    assert_eq!(state.vel_enu_mps, [0.0, 0.0, 0.0]);

    // Extreme dt_s clamping bounds: [1e-4, 60.0]
    state.prev_eye = Some([0.0, 0.0, 0.0]);
    state.on_anchor_accepted([0.01, 0.0, 0.0], Matrix3::identity(), 0.0);
    assert!((state.vel_enu_mps[0] - 100.0).abs() < 1e-4);

    state.prev_eye = Some([0.0, 0.0, 0.0]);
    state.on_anchor_accepted([600.0, 0.0, 0.0], Matrix3::identity(), 120.0);
    assert!((state.vel_enu_mps[0] - 10.0).abs() < 1e-4);
    Ok(())
}

#[test]
fn test_multi_tier_search_resolution_boundaries() -> TestResult {
    let local_lim = 30.0;
    let region_lim = 2000.0;

    // Local tier: r <= 30m
    assert_eq!(
        resolve_search_tier(Some(0.0), local_lim, region_lim),
        SearchTier::Local
    );
    assert_eq!(
        resolve_search_tier(Some(15.0), local_lim, region_lim),
        SearchTier::Local
    );
    assert_eq!(
        resolve_search_tier(Some(30.0), local_lim, region_lim),
        SearchTier::Local
    );

    // Region tier: 30m < r <= 2000m
    assert_eq!(
        resolve_search_tier(Some(30.0001), local_lim, region_lim),
        SearchTier::Region
    );
    assert_eq!(
        resolve_search_tier(Some(100.0), local_lim, region_lim),
        SearchTier::Region
    );
    assert_eq!(
        resolve_search_tier(Some(1999.9), local_lim, region_lim),
        SearchTier::Region
    );
    assert_eq!(
        resolve_search_tier(Some(2000.0), local_lim, region_lim),
        SearchTier::Region
    );

    // Global tier: r > 2000m or missing prior
    assert_eq!(
        resolve_search_tier(Some(2000.0001), local_lim, region_lim),
        SearchTier::Global
    );
    assert_eq!(
        resolve_search_tier(Some(10_000.0), local_lim, region_lim),
        SearchTier::Global
    );
    assert_eq!(
        resolve_search_tier(None, local_lim, region_lim),
        SearchTier::Global
    );

    // SearchPrior mapping through SearchConfig
    let config = SearchConfig {
        sigma_scale: 3.0,
        local_radius_m: 30.0,
        region_radius_m: 2000.0,
        candidate_spacing_m: 40.0,
        max_candidates: 64,
    };

    let prior_local = SearchPrior {
        center: CameraPose {
            position: Vector3::new(0.0, 0.0, 100.0),
            orientation: UnitQuaternion::identity(),
        },
        position_covariance_m2: Matrix3::from_diagonal(&Vector3::new(25.0, 25.0, 9.0)),
        attitude_sigma_rad: 0.01,
    };
    assert_eq!(
        resolve_search_tier_from_prior(&prior_local, &config),
        SearchTier::Local
    );

    let prior_region = SearchPrior {
        center: CameraPose {
            position: Vector3::new(0.0, 0.0, 100.0),
            orientation: UnitQuaternion::identity(),
        },
        position_covariance_m2: Matrix3::from_diagonal(&Vector3::new(10_000.0, 10_000.0, 9.0)),
        attitude_sigma_rad: 0.05,
    };
    assert_eq!(
        resolve_search_tier_from_prior(&prior_region, &config),
        SearchTier::Region
    );

    let prior_global = SearchPrior {
        center: CameraPose {
            position: Vector3::new(0.0, 0.0, 100.0),
            orientation: UnitQuaternion::identity(),
        },
        position_covariance_m2: Matrix3::from_diagonal(&Vector3::new(
            16_000_000.0,
            16_000_000.0,
            9.0,
        )),
        attitude_sigma_rad: 0.2,
    };
    assert_eq!(
        resolve_search_tier_from_prior(&prior_global, &config),
        SearchTier::Global
    );

    let matcher = SearchTierMatcher::new(MultiTierSearchConfig {
        search_config: config,
        top_k_global: 5,
        global_spacing_m: 100.0,
    });
    assert_eq!(matcher.execute(&prior_local, None)?, SearchTier::Local);
    assert_eq!(matcher.execute(&prior_region, None)?, SearchTier::Region);
    assert_eq!(matcher.execute(&prior_global, None)?, SearchTier::Global);
    Ok(())
}

#[test]
fn test_dinov2_canvas_mapping_descriptor_and_ranking() -> TestResult {
    // 1. Canvas mapping from input RGB image
    let mut rgb = RgbImage::new(100, 50);
    for pixel in rgb.pixels_mut() {
        *pixel = Rgb([120, 150, 180]);
    }
    let dyn_img = DynamicImage::ImageRgb8(rgb);
    let canvas = ground_crop_to_canvas(&dyn_img, None);
    assert_eq!(canvas.rgb.len(), CANVAS_PX * CANVAS_PX);
    assert_eq!(canvas.valid.len(), CANVAS_PX * CANVAS_PX);
    assert!(canvas.valid.iter().all(|&v| v));
    assert_eq!(canvas.rgb[0], [120.0, 150.0, 180.0]);

    // Masked canvas mapping: half valid, half invalid
    let mask: Vec<bool> = (0..5000).map(|i| i < 2500).collect();
    let canvas_masked = ground_crop_to_canvas(&dyn_img, Some(&mask));
    assert!(canvas_masked.valid.iter().any(|&v| v));
    assert!(canvas_masked.valid.iter().any(|&v| !v));

    // 2. Descriptor pooling from tokens (384 dimensions for ViT-S/14)
    let width = 384;
    let num_tokens = 1 + GRID * GRID;
    let tokens = Tokens {
        width,
        values: vec![0.05_f32; num_tokens * width],
    };
    let desc = descriptor_from_tokens_and_canvas(&tokens, &canvas)?;
    assert_eq!(desc.values.len(), 384);
    let norm_sq: f32 = desc.values.iter().map(|x| x * x).sum();
    assert!(
        (norm_sq - 1.0).abs() < 1e-4,
        "Descriptor must be unit normalized"
    );

    // Empty canvas with zero valid pixels produces zero descriptor
    let empty_canvas = Canvas::empty();
    let empty_desc = descriptor_from_tokens_and_canvas(&tokens, &empty_canvas)?;
    assert_eq!(empty_desc.values.len(), 384);
    assert!(empty_desc.values.iter().all(|&x| x == 0.0));

    // Mismatched token dimensions return error
    let bad_tokens = Tokens {
        width: 384,
        values: vec![0.0; 10],
    };
    assert!(descriptor_from_tokens_and_canvas(&bad_tokens, &canvas).is_err());

    // 3. Cosine similarity ranking
    let query_desc = Descriptor {
        values: {
            let mut v = vec![0.0; 384];
            v[0] = 1.0;
            v
        },
    };

    let rec_match = Record {
        centre: [47.0, 11.0],
        descriptor: query_desc.values.clone(),
    };
    let rec_ortho = Record {
        centre: [400.0, 11.0],
        descriptor: {
            let mut v = vec![0.0; 384];
            v[1] = 1.0;
            v
        },
    };
    let rec_close = Record {
        centre: [47.0001, 11.0001],
        descriptor: query_desc.values.clone(),
    };

    let ranked = rank_candidate_regions(&query_desc, &[rec_match, rec_ortho, rec_close], 5, 100.0);
    assert_eq!(
        ranked.len(),
        2,
        "Candidate within spacing must be suppressed"
    );
    assert_eq!(ranked[0].centre, [47.0, 11.0]);
    assert!((ranked[0].score - 1.0).abs() < 1e-4);

    // 4. Candidate tile resolution
    let bank = CorridorLandmarkBank::new([0u8; 32], [0u8; 32], 42, 250.0, Vec::new())
        .map_err(|e| format!("{e:?}"))?;
    let store = CorridorTileStore::from_bank(bank, 16);

    let tiles = load_candidate_tiles(&store, &ranked);
    assert_eq!(tiles.len(), 2);
    let expected_key0 = store.tile_key_at(47.0, 11.0);
    assert_eq!(tiles[0], expected_key0);
    Ok(())
}
