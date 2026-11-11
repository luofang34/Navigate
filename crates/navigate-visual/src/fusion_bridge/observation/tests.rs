use super::*;
use crate::fusion_bridge::{
    AttitudeSample, CameraAlignment, TimedCapture, VerticalDatum, camera_capture,
};
use crate::{CameraPose, EstimateQuality, FrameStamp, MapRevision};
use nalgebra::{Quaternion, SMatrix};
use navigate_contract::{ClockDomainId, DurationNanos, FrameId, SourceEpoch, SourceId};
use navigate_geodesy::BodyEllipsoid;

const CLOCK: ClockDomainId = ClockDomainId::new(5);

fn identity() -> VisualSourceIdentity {
    VisualSourceIdentity {
        source: SourceId::new(41),
        epoch: SourceEpoch::new(1),
        clock: CLOCK,
    }
}

fn budget() -> VisualErrorBudget {
    VisualErrorBudget {
        map_horizontal_m: 3.0,
        map_vertical_m: 5.0,
        calibration_m: 1.0,
        geometry_model_m: 2.0,
        vertical_datum: VerticalDatum::Wgs84Ellipsoid,
        vertical_datum_m: 2.0,
    }
}

fn nadir_mount() -> UnitQuaternion<f64> {
    UnitQuaternion::from_matrix(&Matrix3::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0))
}

fn attitude_budget() -> VisualAttitudeBudget {
    VisualAttitudeBudget {
        eye_to_body: nadir_mount(),
        mounting_rad: 0.002,
        map_rad: 0.001,
    }
}

fn estimate(
    frame: LocalFrame,
    position: Vector3<f64>,
    orientation: UnitQuaternion<f64>,
) -> Estimate {
    let mut geometry_covariance = SMatrix::<f64, 6, 6>::identity() * 1e-6;
    geometry_covariance[(0, 0)] = 4.0;
    geometry_covariance[(1, 1)] = 9.0;
    geometry_covariance[(2, 2)] = 16.0;
    geometry_covariance[(5, 5)] = 4e-6;
    Estimate {
        stamp: FrameStamp {
            sequence: 3,
            capture_time_ns: 1_020_000_000,
        },
        observation_sha256: "d".repeat(64),
        map: MapRevision {
            release_id: "release".into(),
            manifest_sha256: "a".repeat(64),
        },
        frame,
        pose: CameraPose {
            position,
            orientation,
        },
        quality: EstimateQuality {
            spatial_support: 30,
            depth_matches: 200,
            inliers: 150,
            reprojection_rms_px: 0.8,
            occupied_cells: 12,
            condition_number: 100.0,
        },
        geometry_covariance,
        backend: "test".into(),
    }
}

fn unit(q: AttitudeQuaternion) -> UnitQuaternion<f64> {
    UnitQuaternion::new_normalize(Quaternion::new(q.w, q.x, q.y, q.z))
}

fn matrix(c: &SymmetricCov3) -> Matrix3<f64> {
    let [xx, xy, xz, yy, yz, zz] = c.upper_triangle();
    Matrix3::new(xx, xy, xz, xy, yy, yz, xz, yz, zz)
}

#[test]
fn a_visual_pose_recovers_the_attitude_that_camera_capture_applied() {
    let body_to_ned = UnitQuaternion::from_euler_angles(0.05, -0.03, 1.2);
    let q = body_to_ned.quaternion();
    let sample = |ms: u64| AttitudeSample {
        time: TimedCapture {
            at: MonotonicNanos::from_nanos(ms * 1_000_000),
            clock: CLOCK,
            error_bound: DurationNanos::from_millis(1),
        },
        body_to_ned: AttitudeQuaternion::new(q.w, q.i, q.j, q.k),
        angular_rms_rad: Some(0.01),
    };
    let alignment = CameraAlignment {
        eye_to_body: nadir_mount(),
        mounting_rms_rad: Some(0.002),
        interpolation_rms_rad: Some(0.001),
        maximum_gap: DurationNanos::from_millis(100),
        angular_speed_bound_rad_s: 1.0,
    };
    let capture = TimedCapture {
        at: MonotonicNanos::from_nanos(1_020_000_000),
        clock: CLOCK,
        error_bound: DurationNanos::from_millis(1),
    };
    let captured = camera_capture(capture, sample(1_000), sample(1_040), alignment).unwrap();
    let frame = LocalFrame::wgs84_enu(47.0, 8.0).unwrap();
    let e = estimate(frame, Vector3::new(0.0, 0.0, 900.0), captured.orientation);
    let observation =
        visual_pose_observation(&e, &identity(), &budget(), &attitude_budget()).unwrap();
    let ObservationValue::VisualPose { attitude, .. } = observation.value else {
        panic!("visual pose expected");
    };
    assert!(unit(attitude).angle_to(&body_to_ned) < 1e-9);
    assert_eq!(observation.frame, Some(FrameId::EARTH_ITRF93));
}

#[test]
fn attitude_covariance_rotates_into_ned_and_carries_the_budget() {
    // Camera yaw error about the optical axis of a nadir camera is a rotation about down.
    let frame = LocalFrame::body_tangent_enu(
        BodyEllipsoid::MARS_IAU,
        GeodeticPosition::new(0.3, -1.0, 0.0),
    )
    .unwrap();
    let orientation = UnitQuaternion::identity();
    let e = estimate(frame, Vector3::new(0.0, 0.0, 900.0), orientation);
    let observation =
        visual_pose_observation(&e, &identity(), &budget(), &attitude_budget()).unwrap();
    assert_eq!(observation.frame, Some(BodyEllipsoid::MARS_IAU.frame()));
    let ObservationValue::VisualPose {
        position_covariance,
        attitude_covariance,
        ..
    } = observation.value
    else {
        panic!("visual pose expected");
    };
    let external = 2.0 * 0.003_f64.powi(2);
    let attitude = matrix(&attitude_covariance);
    assert!((attitude[(2, 2)] - (4.0 * 4e-6 + external)).abs() < 1e-12);
    assert!((attitude[(0, 0)] - (4.0 * 1e-6 + external)).abs() < 1e-12);
    // Doubling the blocks of the joint covariance doubles the image-geometry position term.
    let fix = position_fix_observation(&e, &identity(), &budget()).unwrap();
    let ObservationValue::PositionFix { covariance, .. } = fix.value else {
        panic!("position fix expected");
    };
    let difference = matrix(&position_covariance) - matrix(&covariance);
    assert!((difference[(0, 0)] - 2.0 * 9.0).abs() < 1e-9);
    assert!((difference[(1, 1)] - 2.0 * 4.0).abs() < 1e-9);
    assert!((difference[(2, 2)] - 2.0 * 16.0).abs() < 1e-9);
}

#[test]
fn invalid_attitude_budgets_and_geoid_offsets_on_body_frames_are_refused() {
    let frame = LocalFrame::body_tangent_enu(
        BodyEllipsoid::MOON_PA,
        GeodeticPosition::new(-1.5, 0.5, 0.0),
    )
    .unwrap();
    let e = estimate(
        frame,
        Vector3::new(10.0, -20.0, 500.0),
        UnitQuaternion::identity(),
    );
    for field in ["mounting_rad", "map_rad"] {
        let mut b = attitude_budget();
        match field {
            "mounting_rad" => b.mounting_rad = 0.0,
            _ => b.map_rad = f64::NAN,
        }
        assert_eq!(
            visual_pose_observation(&e, &identity(), &budget(), &b).err(),
            Some(VisualFusionError::InvalidBudget { field })
        );
    }
    let mut geoid = budget();
    geoid.vertical_datum = VerticalDatum::Orthometric {
        geoid_separation_m: 10.0,
    };
    assert_eq!(
        visual_pose_observation(&e, &identity(), &geoid, &attitude_budget()).err(),
        Some(VisualFusionError::FrameDatumMismatch)
    );
    let mut bad = e;
    bad.geometry_covariance[(4, 4)] = -1.0;
    assert_eq!(
        visual_pose_observation(&bad, &identity(), &budget(), &attitude_budget()).err(),
        Some(VisualFusionError::InvalidCovariance)
    );
}

#[test]
fn delayed_propagated_anchor_inflates_vo_covariance_and_preserves_monotonic_filter_order() {
    use crate::fusion_bridge::{EvidenceIndependence, VisualFixSource};
    use navigate_fusion::{FusionConfig, NavigationFilter};

    let frame = LocalFrame::wgs84_enu(47.0, 8.0).unwrap();
    let mut fg_est = estimate(
        frame,
        Vector3::new(5.0, 0.0, 900.0),
        UnitQuaternion::identity(),
    );
    fg_est.stamp = FrameStamp {
        sequence: 2,
        capture_time_ns: 133_000_000,
    };
    fg_est.observation_sha256 = "1".repeat(64);

    let mut source = VisualFixSource::new(identity(), budget()).unwrap();
    let mut filter = NavigationFilter::new(FusionConfig::default(), CLOCK);
    let fg1 = source
        .convert(&fg_est, EvidenceIndependence::ValidatedIndependent)
        .unwrap();
    assert_eq!(fg1.capture_lag_ns, 0);
    assert_eq!(fg1.observation.frame, Some(FrameId::EARTH_ITRF93));
    assert!(
        filter
            .ingest(&fg1.observation, MonotonicNanos::from_nanos(133_000_000))
            .is_accepted()
    );

    let mut delayed = estimate(
        frame,
        Vector3::new(12.0, 0.0, 900.0),
        UnitQuaternion::identity(),
    );
    delayed.stamp = FrameStamp {
        sequence: 1,
        capture_time_ns: 100_000_000,
    };
    delayed.observation_sha256 = "2".repeat(64);
    assert_eq!(
        source.convert_propagated(
            &delayed,
            EvidenceIndependence::ValidatedIndependent,
            266_000_000,
            6,
            &Matrix3::zeros(),
        ),
        Err(VisualFusionError::InvalidCovariance)
    );

    let q_vo = Matrix3::from_diagonal(&Vector3::new(25.0, 36.0, 9.0));
    let bg_fix = source
        .convert_propagated(
            &delayed,
            EvidenceIndependence::ValidatedIndependent,
            266_000_000,
            6,
            &q_vo,
        )
        .unwrap();
    assert_eq!(bg_fix.capture_lag_ns, 166_000_000);
    assert_eq!(bg_fix.observation.frame, Some(FrameId::EARTH_ITRF93));
    assert_eq!(
        bg_fix.observation.stamp.acquired_at,
        MonotonicNanos::from_nanos(266_000_000)
    );
    let ObservationValue::PositionFix {
        covariance: fg_cov, ..
    } = fg1.observation.value
    else {
        panic!("expected position fix");
    };
    let ObservationValue::PositionFix {
        covariance: bg_cov, ..
    } = bg_fix.observation.value
    else {
        panic!("expected position fix");
    };
    assert!(bg_cov.diagonal().0 >= fg_cov.diagonal().0 + 2.0 * 36.0 - 1e-6);
    assert!(bg_cov.diagonal().1 >= fg_cov.diagonal().1 + 2.0 * 25.0 - 1e-6);
    assert!(
        filter
            .ingest(&bg_fix.observation, MonotonicNanos::from_nanos(266_000_000))
            .is_accepted()
    );
}

#[test]
fn polar_lunar_propagated_visual_pose_carries_frame_and_inflates_both_blocks() {
    use crate::fusion_bridge::{EvidenceIndependence, VisualFixSource};

    let frame = LocalFrame::body_tangent_enu(
        BodyEllipsoid::MOON_PA,
        GeodeticPosition::new((-89.9_f64).to_radians(), 0.25, 0.0),
    )
    .unwrap();
    let mut delayed = estimate(
        frame,
        Vector3::new(8.0, -4.0, 250.0),
        UnitQuaternion::identity(),
    );
    delayed.stamp = FrameStamp {
        sequence: 4,
        capture_time_ns: 200_000_000,
    };
    delayed.observation_sha256 = "3".repeat(64);

    let mut source = VisualFixSource::new(identity(), budget()).unwrap();
    let q_pos = Matrix3::from_diagonal(&Vector3::new(9.0, 16.0, 4.0));
    let q_att = Matrix3::from_diagonal(&Vector3::new(1e-4, 1e-4, 4e-4));
    assert_eq!(
        source.convert_pose_propagated(
            &delayed,
            EvidenceIndependence::ValidatedIndependent,
            &attitude_budget(),
            350_000_000,
            9,
            &q_pos,
            &Matrix3::zeros(),
        ),
        Err(VisualFusionError::InvalidCovariance)
    );

    let sync_obs =
        visual_pose_observation(&delayed, &identity(), &budget(), &attitude_budget()).unwrap();
    let prop_fix = source
        .convert_pose_propagated(
            &delayed,
            EvidenceIndependence::ValidatedIndependent,
            &attitude_budget(),
            350_000_000,
            9,
            &q_pos,
            &q_att,
        )
        .unwrap();
    assert_eq!(prop_fix.capture_lag_ns, 150_000_000);
    assert_eq!(
        prop_fix.observation.frame,
        Some(BodyEllipsoid::MOON_PA.frame())
    );
    let (
        ObservationValue::VisualPose {
            position_covariance: sync_p,
            attitude_covariance: sync_a,
            ..
        },
        ObservationValue::VisualPose {
            position_covariance: prop_p,
            attitude_covariance: prop_a,
            ..
        },
    ) = (sync_obs.value, prop_fix.observation.value)
    else {
        panic!("expected visual poses");
    };
    assert!(prop_p.diagonal().0 > sync_p.diagonal().0 + 30.0);
    assert!(prop_a.diagonal().2 > sync_a.diagonal().2 + 1e-3);
}

#[test]
fn a_composed_fix_adds_the_callers_covariance_without_inflation() {
    let frame = LocalFrame::wgs84_enu(47.0, 11.0).unwrap();
    let fix = estimate(
        frame,
        Vector3::new(0.0, 0.0, 900.0),
        UnitQuaternion::identity(),
    );
    // Independent terms of 3 m and 4 m horizontal give a 5 m root sum square.
    let horizontal_m2 = 3.0_f64.powi(2) + 4.0_f64.powi(2);
    let external = Matrix3::from_diagonal(&Vector3::new(horizontal_m2, horizontal_m2, 36.0));
    let observation = position_fix_observation_composed(&fix, &identity(), &external).unwrap();
    assert_eq!(observation.frame, Some(FrameId::EARTH_ITRF93));
    assert_eq!(observation.stamp.source, identity().source);
    let ObservationValue::PositionFix {
        position,
        covariance,
    } = observation.value
    else {
        panic!("a composed fix is a position fix");
    };
    assert!((position.altitude_m - 900.0).abs() < 1e-6);
    // East 4, north 9, up 16 m² of image geometry move to north, east, down.
    let ned = matrix(&covariance);
    for (axis, expected) in [(0, 9.0 + 25.0), (1, 4.0 + 25.0), (2, 16.0 + 36.0)] {
        assert!((ned[(axis, axis)] - expected).abs() < 1e-9, "{ned}");
    }
    let linear = position_fix_observation(&fix, &identity(), &budget()).unwrap();
    let ObservationValue::PositionFix {
        covariance: linear_cov,
        ..
    } = linear.value
    else {
        panic!("a position fix");
    };
    assert!(matrix(&linear_cov)[(0, 0)] > 5.0 * ned[(0, 0)]);

    let mut negative = external;
    negative[(2, 2)] = -1.0;
    let mut skew = external;
    skew[(0, 1)] = 1.0;
    for bad in [negative, skew, external * f64::NAN] {
        assert_eq!(
            position_fix_observation_composed(&fix, &identity(), &bad),
            Err(VisualFusionError::InvalidCovariance)
        );
    }
}
