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
    let mut bad = e.clone();
    bad.geometry_covariance[(4, 4)] = -1.0;
    assert_eq!(
        visual_pose_observation(&bad, &identity(), &budget(), &attitude_budget()).err(),
        Some(VisualFusionError::InvalidCovariance)
    );
}
