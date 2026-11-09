//! Basin verification and multi-hypothesis override regression tests.

mod replay;

use crate::{
    AnchorConfidence, Surface,
    anchor::{
        basin_priority_key, basin_quality_score, classify_anchor_confidence,
        should_override_provisional_lock,
    },
    anchor_arrival::{ActiveAnchorStatus, AnchorApplyCtx, RecentAnchorBackup},
    anchor_worker::AnchorOutcome,
};
use image::RgbImage;
use nalgebra::Matrix3;
use navigate_contract::{
    ClockDomainId, MonotonicNanos, ObservationStamp, SourceEpoch, SourceId, WrappingSequence,
};
use navigate_locate::corridor::{CorridorTracker, VehicleDynamicsProfile};
use navigate_match::{ActiveTrackSet, DEFAULT_KASHIN_SEED, HadamardProjector, SparseFeatures};
use navigate_visual::{CameraModel, PoseCandidate, PoseScore};

fn test_stamp(seq: u32, nanos: u64) -> ObservationStamp {
    ObservationStamp::new(
        SourceId::new(1),
        SourceEpoch::new(1),
        WrappingSequence::new(seq),
        MonotonicNanos::from_nanos(nanos),
        ClockDomainId::new(1),
    )
}

fn make_candidate(
    eye: [f64; 3],
    inliers: usize,
    holdout_inliers: usize,
    holdout_median_px: f64,
    train_rms_px: f64,
) -> PoseCandidate {
    let k = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
    let rot = Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0);
    let mut world = Vec::with_capacity(inliers);
    let mut px = Vec::with_capacity(inliers);
    let mut mask = Vec::with_capacity(inliers);
    for i in 0..inliers {
        let gx = ((i % 8) as f64 - 3.5) * 14.0;
        let gy = ((i / 8) as f64 - 3.0) * 14.0;
        let w = [eye[0] + gx, eye[1] + gy, 0.0];
        let u = 600.0 * (gx / eye[2]) + 399.5;
        let v = 600.0 * (-gy / eye[2]) + 299.5;
        world.push(w);
        px.push([u, v]);
        mask.push(true);
    }
    let holdout_count = (holdout_inliers + 2).max(10);
    PoseCandidate {
        eye,
        rotation: rot,
        k,
        size: [800, 600],
        points: world,
        pixels: px,
        inliers: mask,
        score: PoseScore {
            focal_px: 600.0,
            geometry_accepted: true,
            holdout_count,
            holdout_inlier_fraction: (holdout_inliers as f64) / (holdout_count as f64),
            holdout_median_px,
            image_support_fraction: 0.45,
            inliers,
            pairs: inliers + 10,
            positive_depth_fraction: 1.0,
            train_inliers: inliers.saturating_sub(holdout_inliers),
            train_rms_px,
        },
    }
}

/// Regression Test (d): `71 m` false spatial-ring basin (`9.0 px` holdout median) vs true basin (`2.4 px`).
///
/// Verifies both simultaneous multi-hypothesis selection AND delayed `Provisional -> Confirmed` override.
#[test]
fn multi_hypothesis_basin_gates_false_ring_alias_and_overrides_provisional_lock()
-> Result<(), String> {
    let alias_cand = make_candidate([57.0, 57.0, 98.0], 54, 12, 9.0, 6.7);
    let true_cand = make_candidate([7.0, 6.0, 98.0], 52, 13, 2.4, 2.6);
    let dist_m = (alias_cand.eye[0] - true_cand.eye[0]).hypot(alias_cand.eye[1] - true_cand.eye[1]);
    assert!(dist_m > 70.0 && dist_m < 72.0);

    assert!(basin_quality_score(&true_cand) > basin_quality_score(&alias_cand) * 2.0);
    assert!(basin_priority_key(true, &true_cand) > basin_priority_key(true, &alias_cand));
    assert_eq!(
        classify_anchor_confidence(
            &true_cand,
            &[(true, alias_cand.clone()), (true, true_cand.clone())]
        ),
        AnchorConfidence::Confirmed
    );

    let alias_conf = classify_anchor_confidence(&alias_cand, &[(true, alias_cand.clone())]);
    assert_eq!(alias_conf, AnchorConfidence::Provisional);
    assert!(should_override_provisional_lock(
        alias_cand.score.holdout_median_px,
        &true_cand,
        AnchorConfidence::Confirmed
    ));

    let surface = Surface::synthetic(39.0, -75.5, 0.0, RgbImage::new(64, 64));
    let lf = surface.local_frame().map_err(|e| e.to_string())?;
    let cam = CameraModel {
        width: 800,
        height: 600,
        fx: 600.0,
        fy: 600.0,
        cx: 399.5,
        cy: 299.5,
    };
    let mut tracker = CorridorTracker::try_new(cam)?;
    tracker.configure_session(cam, 400.0)?;
    let _ =
        tracker.observe_frame_monotonic(MonotonicNanos::from_nanos(100_000_000), "", "f-00001")?;

    let mut active_status: Option<ActiveAnchorStatus> = None;
    let mut recent_backup: Option<RecentAnchorBackup> = None;
    let mut ind_fixes = 0_usize;
    let mut focal_est = crate::anchor::FocalSweepEstimator::default();
    let proj = HadamardProjector::new(DEFAULT_KASHIN_SEED);
    let dummy_feat = SparseFeatures {
        pixels: true_cand.pixels.clone(),
        model_pixels: true_cand
            .pixels
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32])
            .collect(),
        scores: vec![0.9; true_cand.pixels.len()],
        descriptors: vec![0.125; true_cand.pixels.len() * 64],
    };
    let prep = navigate_match::PreparedKashinQuery::from_features(&proj, &dummy_feat)
        .map_err(|e| e.to_string())?;
    let world_opts: Vec<Option<[f64; 3]>> = alias_cand.points.iter().copied().map(Some).collect();
    let alias_tracks = ActiveTrackSet::from_prepared_query(&prep, &world_opts, 256);

    {
        let mut ctx = AnchorApplyCtx {
            state: &mut tracker,
            active_status: &mut active_status,
            recent_backup: &mut recent_backup,
            independent_fixes: &mut ind_fixes,
            focal_estimator: &mut focal_est,
            catalog_sha256: &surface.catalog_sha256,
            local_frame: lf,
            cell_m: 400.0,
            dynamics_profile: VehicleDynamicsProfile::rotary_survey(),
        };
        let (_, rec1) = ctx.apply(
            AnchorOutcome {
                sent_frame_seq: 0,
                capture_stamp: test_stamp(1, 100_000_000),
                frame_label: "f-00001".into(),
                query_prior_enu: Some([0.0, 0.0, 98.0]),
                k_render: alias_cand.k,
                render_size: [800, 600],
                agl_m: 98.0,
                best: Some((0, 0, alias_cand.clone())),
                seeded: Some(alias_tracks),
                locked_focal_px: None,
                confidence: Some(AnchorConfidence::Provisional),
                was_polished: true,
                lg_layers: 6,
                polish_rays: 0,
                attempts: Vec::new(),
                elapsed_ms: 120.0,
            },
            Some((1, 100_000_000, "f-00001")),
        );
        assert_eq!(rec1.confidence, Some(AnchorConfidence::Provisional));
    }
    assert_eq!(
        active_status.map(|s| s.confidence),
        Some(AnchorConfidence::Provisional)
    );

    tracker.frame_seq = 10;
    tracker.time_since_accepted_pose_s = 0.03;
    let _ =
        tracker.observe_frame_monotonic(MonotonicNanos::from_nanos(400_000_000), "", "f-00010")?;
    let true_world_opts: Vec<Option<[f64; 3]>> =
        true_cand.points.iter().copied().map(Some).collect();
    let true_tracks = ActiveTrackSet::from_prepared_query(&prep, &true_world_opts, 256);
    {
        let mut ctx = AnchorApplyCtx {
            state: &mut tracker,
            active_status: &mut active_status,
            recent_backup: &mut recent_backup,
            independent_fixes: &mut ind_fixes,
            focal_estimator: &mut focal_est,
            catalog_sha256: &surface.catalog_sha256,
            local_frame: lf,
            cell_m: 400.0,
            dynamics_profile: VehicleDynamicsProfile::rotary_survey(),
        };
        let (staged, rec2) = ctx.apply(
            AnchorOutcome {
                sent_frame_seq: 2,
                capture_stamp: test_stamp(2, 166_000_000),
                frame_label: "f-00002".into(),
                query_prior_enu: Some([0.0, 0.0, 98.0]),
                k_render: true_cand.k,
                render_size: [800, 600],
                agl_m: 98.0,
                best: Some((1, 1, true_cand.clone())),
                seeded: Some(true_tracks),
                locked_focal_px: None,
                confidence: Some(AnchorConfidence::Confirmed),
                was_polished: true,
                lg_layers: 6,
                polish_rays: 0,
                attempts: Vec::new(),
                elapsed_ms: 180.0,
            },
            Some((10, 400_000_000, "f-00010")),
        );
        assert_eq!(rec2.confidence, Some(AnchorConfidence::Confirmed));
        assert!(!rec2.merged_into_healthy_foreground);
        assert!(staged.is_some());
    }
    assert_eq!(
        active_status.map(|s| s.confidence),
        Some(AnchorConfidence::Confirmed)
    );
    let snapped_eye = tracker.prev_eye.ok_or("missing snapped eye")?;
    assert!((snapped_eye[0] - true_cand.eye[0]).hypot(snapped_eye[1] - true_cand.eye[1]) < 1.0);
    Ok(())
}

fn build_polar_lunar_engine() -> Result<(std::sync::Arc<Surface>, crate::FlightVisualEngine), String>
{
    use crate::{
        ExecutionPolicy, FlightEngineConfig, FlightEngineResources, FlightVisualEngine,
        LoadedXFeat, TileSynthParams, VisualAttitudeBudget, synthesize_corridor_tiles,
        tests::replay::{PatchGridExtractor, build_textured_tile},
    };
    use burn::tensor::Device;
    use nalgebra::{Rotation3, UnitQuaternion};
    use navigate_geodesy::BodyEllipsoid;
    use navigate_match::{CorridorTileKey, CorridorTileStore};
    use navigate_visual::LocalFrame;
    use std::sync::{Arc, atomic::AtomicBool};

    let surface = Arc::new(Surface::synthetic_on_body(
        BodyEllipsoid::MOON_PA,
        -89.9,
        15.0,
        10.0,
        build_textured_tile(),
    ));
    assert!(matches!(
        surface.local_frame().map_err(|e| e.to_string())?,
        LocalFrame::BodyTangentEnu { ellipsoid, .. } if ellipsoid == BodyEllipsoid::MOON_PA
    ));
    let contract = [9_u8; 32];
    let xfeat = LoadedXFeat::from_extractor(
        Arc::new(PatchGridExtractor {
            contract,
            jitter_on_worker: Arc::new(AtomicBool::new(false)),
        }),
        Device::flex(),
        contract,
    );
    let store = Arc::new(
        CorridorTileStore::in_memory([0_u8; 32], contract, DEFAULT_KASHIN_SEED, 150.0, 9)
            .map_err(|e| e.to_string())?,
    );
    for entry in synthesize_corridor_tiles(
        &surface,
        None,
        &xfeat,
        &HadamardProjector::new(DEFAULT_KASHIN_SEED),
        CorridorTileKey {
            tile_x: 0,
            tile_y: 0,
        },
        TileSynthParams {
            tile_m: 150.0,
            agl_m: 100.0,
            max_per_tile: 1024,
            stride: 4,
            include_obliques: false,
        },
    )
    .map_err(|e| e.to_string())?
    {
        store.insert_tile(entry, true).map_err(|e| e.to_string())?;
    }
    let eye_to_body = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(
        Matrix3::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0),
    ));
    let mut cfg = FlightEngineConfig {
        policy: ExecutionPolicy::DeterministicReplay {
            anchor_budget_ms: 50,
            tile_budget_ms: 100,
        },
        background_workers: false,
        enable_live_tile_synth: false,
        attitude_budget: Some(VisualAttitudeBudget {
            eye_to_body,
            mounting_rad: 0.005,
            map_rad: 0.005,
        }),
        ..FlightEngineConfig::default()
    };
    cfg.search_opts.polish_anchors = false;
    cfg.search_opts.query_keypoints = 512;
    let res = Arc::new(
        FlightEngineResources::from_parts(
            Arc::clone(&surface),
            store,
            xfeat,
            None,
            None,
            None,
            true,
        )
        .map_err(|e| e.to_string())?,
    );
    let engine = FlightVisualEngine::new(res, cfg).map_err(|e| e.to_string())?;
    Ok((surface, engine))
}

#[test]
fn polar_lunar_flight_engine_emits_visual_pose_for_inertial_filter() -> Result<(), String> {
    use crate::{
        CalibrationMode, CameraIntrinsics, CameraMountProfile, FramePixels, LiveFrameInput,
        SearchPrior,
    };
    use navigate_astrodynamics::PlanetaryBody;
    use navigate_contract::{NedVelocity, SensorClass, SourceComposition, SymmetricCov3};
    use navigate_fusion::{InertialConfig, InertialNavigationFilter, ObservationValue};
    use navigate_geodesy::BodyEllipsoid;

    let (surface, mut engine) = build_polar_lunar_engine()?;
    let k = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
    let eye = [75.0, 75.0, 227.5];
    let geo = surface.geodetic(eye);
    let rot = surface.camera_rotation(geo[0], geo[1], 0.0, 89.0, 0.0);
    let (rgb, _, _) = surface.render(eye, &rot, &k, [800, 600]);
    let t_ns = 100_000_000_u64;
    let (out, diag) = engine
        .process_frame_with_diagnostics(
            &LiveFrameInput {
                stamp: test_stamp(1, t_ns),
                pixels: FramePixels::Rgb8 {
                    data: rgb.as_raw(),
                    width: 800,
                    height: 600,
                    stride: 800 * 3,
                },
                band: navigate_camera::SpectralBand::Visible,
                static_valid_mask: None,
                intrinsics: CameraIntrinsics {
                    width: 800,
                    height: 600,
                    fx: 600.0,
                    fy: 600.0,
                    cx: 399.5,
                    cy: 299.5,
                    calibration_confidence: CalibrationMode::Calibrated,
                },
                mount: CameraMountProfile::rotary_gimbal_nadir(),
                imu_attitude: None,
                baro_alt_msl_m: Some((227.5, 5.0)),
                filter_prior: Some(SearchPrior {
                    center_geodetic: navigate_contract::GeodeticPosition::new(
                        geo[0].to_radians(),
                        geo[1].to_radians(),
                        227.5,
                    ),
                    horizontal_sigma_m: 25.0,
                    velocity_ned_mps: None,
                }),
            },
            "lunar-f1",
        )
        .map_err(|e| e.to_string())?;
    let fix = out.visual_fix.ok_or_else(|| {
        format!(
            "expected Lunar 6-DOF visual fix; tel={:?} cand={:?} att0={:?}",
            out.telemetry,
            diag.candidate.as_ref().map(|(_, _, c)| (&c.eye, &c.score)),
            &diag.attempts[..diag.attempts.len().min(4)]
        )
    })?;
    assert_eq!(fix.observation.frame, Some(BodyEllipsoid::MOON_PA.frame()));
    let ObservationValue::VisualPose {
        position, attitude, ..
    } = fix.observation.value
    else {
        return Err("expected ObservationValue::VisualPose".into());
    };
    let mut ins = InertialNavigationFilter::new_surface(
        InertialConfig::default(),
        ClockDomainId::new(1),
        PlanetaryBody::MOON_PA,
    );
    ins.initialize_surface(
        position,
        NedVelocity::new(0.0, 0.0, 0.0),
        attitude,
        SymmetricCov3::from_diagonal(25.0, 25.0, 25.0),
        SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
        SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
        MonotonicNanos::from_nanos(t_ns),
        SourceComposition::of(SensorClass::Inertial),
    )
    .map_err(|e| format!("{e:?}"))?;
    let accepted = ins.ingest(&fix.observation, MonotonicNanos::from_nanos(t_ns));
    assert!(accepted.is_accepted(), "{accepted:?}");
    Ok(())
}
