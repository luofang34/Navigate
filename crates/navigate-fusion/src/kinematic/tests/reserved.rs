use navigate_contract::AttitudeQuaternion;

use super::*;
use crate::observation::MeasurementKind;

fn reserved_values() -> [ObservationValue; 2] {
    let covariance = SymmetricCov3::from_diagonal(25.0, 25.0, 25.0);
    [
        ObservationValue::Pseudorange {
            emitter_position_m: [15_600_000.0, 7_540_000.0, 20_140_000.0],
            pseudorange_m: 21_000_000.0,
            sagnac_and_media_correction_m: 0.0,
            variance_m2: 25.0,
        },
        ObservationValue::VisualPose {
            position: origin(),
            position_covariance: covariance,
            attitude: AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0),
            attitude_covariance: SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
        },
    ]
}

#[test]
fn reserved_measurements_are_refused_by_name_and_counted() {
    let mut f = filter();
    assert!(
        f.ingest(&gnss_fix(1, 1, 1000, origin()), t(1000))
            .is_accepted()
    );
    let before = f.tick(t(1000));
    for (index, value) in reserved_values().into_iter().enumerate() {
        let sequence = u32::try_from(index).expect("small index") + 1;
        let obs = Observation::new(
            stamp(9, 1, sequence, 1000),
            value,
            SourceComposition::of(SensorClass::Gnss),
        );
        assert_eq!(
            f.ingest(&obs, t(1000)),
            IngestOutcome::Rejected(RejectionReason::UnsupportedMeasurement { kind: value.kind() })
        );
    }
    assert_eq!(f.rejections().unsupported_measurement, 2);
    let before = before.expect("solution before");
    let after = f.tick(t(1000)).expect("solution after");
    assert_eq!(
        (after.position, after.position_cov, after.velocity_cov),
        (before.position, before.position_cov, before.velocity_cov),
        "a refused measurement leaves the state unchanged"
    );
    let kinds = reserved_values().map(|v| v.kind());
    assert_eq!(
        kinds,
        [MeasurementKind::Pseudorange, MeasurementKind::VisualPose]
    );
}

#[test]
fn inertial_values_are_refused_by_the_six_state_filter() {
    use navigate_contract::{CartesianPosition, CartesianVelocity};

    let mut f = filter();
    assert_accepted(f.ingest(&gnss_fix(1, 1, 1000, origin()), t(1000)));
    let covariance = SymmetricCov3::from_diagonal(1.0, 1.0, 1.0);
    let values = [
        ObservationValue::TerrainClearance {
            clearance_m: 300.0,
            terrain_height_m: 200.0,
            terrain_slope_north_east: [0.0, 0.0],
            beam_body_unit: [0.0, 0.0, 1.0],
            variance_m2: 4.0,
        },
        ObservationValue::AttitudeFix {
            attitude: AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0),
            covariance,
        },
        ObservationValue::CartesianPositionFix {
            position: CartesianPosition::new(1.0, 2.0, 3.0),
            covariance,
        },
        ObservationValue::CartesianVelocityFix {
            velocity: CartesianVelocity::new(1.0, 2.0, 3.0),
            covariance,
        },
        ObservationValue::CartesianRange {
            beacon: CartesianPosition::new(1.0, 2.0, 3.0),
            range_m: 10.0,
            variance_m2: 1.0,
        },
    ];
    for (index, value) in values.into_iter().enumerate() {
        assert!(!value.is_supported());
        let sequence = u32::try_from(index).expect("small index") + 1;
        let obs = Observation::new(
            stamp(9, 1, sequence, 1000),
            value,
            SourceComposition::of(SensorClass::Gnss),
        );
        assert_eq!(
            f.ingest(&obs, t(1000)),
            IngestOutcome::Rejected(RejectionReason::UnsupportedMeasurement { kind: value.kind() })
        );
    }
    assert_eq!(f.rejections().unsupported_measurement, 5);
}
