use navigate_contract::AttitudeQuaternion;

use super::*;
use crate::observation::MeasurementKind;

fn reserved_values() -> [ObservationValue; 3] {
    let covariance = SymmetricCov3::from_diagonal(25.0, 25.0, 25.0);
    [
        ObservationValue::Range {
            station: position_north_m(10_000.0),
            range_m: 10_000.0,
            variance_m2: 25.0,
        },
        ObservationValue::Pseudorange {
            satellite_ecef_m: [15_600_000.0, 7_540_000.0, 20_140_000.0],
            pseudorange_m: 21_000_000.0,
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
    assert_eq!(f.rejections().unsupported_measurement, 3);
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
        [
            MeasurementKind::Range,
            MeasurementKind::Pseudorange,
            MeasurementKind::VisualPose
        ]
    );
}
