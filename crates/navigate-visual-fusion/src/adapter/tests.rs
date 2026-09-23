#![allow(clippy::expect_used, clippy::panic)]

use nalgebra::{SMatrix, UnitQuaternion, Vector3};
use navigate_contract::{ClockDomainId, SourceEpoch, SourceId};
use navigate_fusion::{FusionConfig, NavigationFilter};
use navigate_visual::{CameraPose, EstimateQuality, FrameStamp, LocalFrame};

use super::*;
use crate::VerticalDatum;

const CLOCK: ClockDomainId = ClockDomainId::new(3);

fn identity() -> VisualSourceIdentity {
    VisualSourceIdentity {
        source: SourceId::new(40),
        epoch: SourceEpoch::new(1),
        clock: CLOCK,
    }
}

fn budget() -> VisualErrorBudget {
    VisualErrorBudget {
        map_horizontal_m: 3.0,
        map_vertical_m: 5.0,
        calibration_m: 1.0,
        vertical_datum: VerticalDatum::Orthometric {
            geoid_separation_m: 48.0,
        },
        vertical_datum_m: 2.0,
    }
}

fn estimate(sequence: u64, capture_time_ns: u64, digest: &str, east_m: f64) -> Estimate {
    let mut geometry_covariance = SMatrix::<f64, 6, 6>::identity() * 1e-4;
    geometry_covariance[(0, 0)] = 4.0;
    geometry_covariance[(1, 1)] = 9.0;
    geometry_covariance[(2, 2)] = 16.0;
    Estimate {
        stamp: FrameStamp {
            sequence,
            capture_time_ns,
        },
        observation_sha256: digest.into(),
        map: MapRevision {
            release_id: "release".into(),
            manifest_sha256: "a".repeat(64),
        },
        frame: LocalFrame::anchor_mercator(47.0, 8.0).expect("valid anchor"),
        pose: CameraPose {
            position: Vector3::new(east_m, 0.0, 900.0),
            orientation: UnitQuaternion::identity(),
        },
        quality: EstimateQuality {
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

#[test]
fn zero_or_missing_budget_terms_are_refused() {
    for field in ["map_horizontal_m", "calibration_m", "vertical_datum_m"] {
        let mut b = budget();
        match field {
            "map_horizontal_m" => b.map_horizontal_m = 0.0,
            "calibration_m" => b.calibration_m = f64::NAN,
            _ => b.vertical_datum_m = -1.0,
        }
        assert_eq!(
            VisualFixSource::new(identity(), b).err(),
            Some(VisualFusionError::InvalidBudget { field })
        );
    }
}

#[test]
fn fix_axes_move_to_ned_and_carry_the_budget() {
    let mut source = VisualFixSource::new(identity(), budget()).expect("valid budget");
    let fix = source
        .convert(&estimate(5, 1_000, "one", 0.0))
        .expect("fix");
    let ObservationValue::PositionFix {
        position,
        covariance,
    } = fix.observation.value
    else {
        panic!("position fix expected");
    };
    let (n, e, d) = covariance.diagonal();
    // North takes the frame's north variance (9), east the east variance (4).
    assert!((n - (9.0 + 9.0 + 1.0)).abs() < 1e-9);
    assert!((e - (4.0 + 9.0 + 1.0)).abs() < 1e-9);
    assert!((d - (16.0 + 25.0 + 1.0 + 4.0)).abs() < 1e-9);
    assert!((position.altitude_m - 948.0).abs() < 1e-9);
    assert!((position.latitude_rad.to_degrees() - 47.0).abs() < 1e-9);
    assert!(
        fix.observation
            .composition
            .contains(SensorClass::VisualLandmark)
    );
    assert_eq!(fix.map.release_id, "release");
}

#[test]
fn far_positions_carry_the_frame_model_error() {
    let mut source = VisualFixSource::new(identity(), budget()).expect("valid budget");
    let near = source.convert(&estimate(1, 1, "near", 0.0)).expect("fix");
    let far = source
        .convert(&estimate(2, 2, "far", 40_000.0))
        .expect("fix");
    let east = |fix: &VisualFix| match fix.observation.value {
        ObservationValue::PositionFix { covariance, .. } => covariance.diagonal().1,
        _ => panic!("position fix expected"),
    };
    assert!(east(&far) > east(&near) + 100.0 * 100.0);
}

#[test]
fn repeated_evidence_and_old_frames_are_refused() {
    let mut source = VisualFixSource::new(identity(), budget()).expect("valid budget");
    source
        .convert(&estimate(1, 100, "frame-a", 0.0))
        .expect("fix");
    assert_eq!(
        source.convert(&estimate(2, 200, "frame-a", 0.0)).err(),
        Some(VisualFusionError::RepeatedEvidence {
            observation_sha256: "frame-a".into()
        })
    );
    assert_eq!(
        source.convert(&estimate(3, 100, "frame-b", 0.0)).err(),
        Some(VisualFusionError::FrameOrder {
            previous_ns: 100,
            received_ns: 100
        })
    );
    source
        .convert(&estimate(4, 300, "frame-c", 0.0))
        .expect("later frame");
}

#[test]
fn the_navigation_filter_admits_a_visual_fix() {
    let mut source = VisualFixSource::new(identity(), budget()).expect("valid budget");
    let fix = source
        .convert(&estimate(1, 1_000_000, "admit", 0.0))
        .expect("fix");
    let mut filter = NavigationFilter::new(FusionConfig::default(), CLOCK);
    let outcome = filter.ingest(&fix.observation, MonotonicNanos::from_nanos(1_000_000));
    assert!(outcome.is_accepted(), "{outcome:?}");
}

#[test]
fn a_foreign_clock_domain_is_rejected_by_the_filter() {
    let mut source = VisualFixSource::new(identity(), budget()).expect("valid budget");
    let fix = source
        .convert(&estimate(1, 1_000_000, "clock", 0.0))
        .expect("fix");
    let mut filter = NavigationFilter::new(FusionConfig::default(), ClockDomainId::new(9));
    let outcome = filter.ingest(&fix.observation, MonotonicNanos::from_nanos(1_000_000));
    assert!(!outcome.is_accepted());
}
