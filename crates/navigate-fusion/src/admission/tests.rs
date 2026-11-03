use navigate_contract::{
    AttitudeQuaternion, CartesianPosition, CartesianVelocity, GeodeticPosition, SymmetricCov3,
};

use super::check_value;
use crate::observation::ObservationValue;
use crate::rejection::RejectionReason;

fn cov() -> SymmetricCov3 {
    SymmetricCov3::from_diagonal(1.0, 1.0, 1.0)
}

fn identity() -> AttitudeQuaternion {
    AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0)
}

fn terrain(beam: [f64; 3], clearance_m: f64) -> ObservationValue {
    ObservationValue::TerrainClearance {
        clearance_m,
        terrain_height_m: 120.0,
        terrain_slope_north_east: [0.1, -0.05],
        beam_body_unit: beam,
        variance_m2: 4.0,
    }
}

#[test]
fn terrain_clearance_needs_a_positive_range_and_a_unit_beam() {
    assert_eq!(check_value(&terrain([0.0, 0.0, 1.0], 300.0)), Ok(()));
    for value in [
        terrain([0.0, 0.0, 1.01], 300.0),
        terrain([0.0, 0.0, f64::NAN], 300.0),
        terrain([0.0, 0.0, 1.0], 0.0),
        terrain([0.0, 0.0, 1.0], f64::INFINITY),
    ] {
        assert_eq!(check_value(&value), Err(RejectionReason::NonFiniteValue));
    }
}

#[test]
fn attitude_values_need_a_unit_quaternion_and_definite_covariances() {
    let fix = |attitude, covariance| ObservationValue::AttitudeFix {
        attitude,
        covariance,
    };
    assert_eq!(check_value(&fix(identity(), cov())), Ok(()));
    assert_eq!(
        check_value(&fix(AttitudeQuaternion::new(1.01, 0.0, 0.0, 0.0), cov())),
        Err(RejectionReason::NonFiniteValue)
    );
    let singular = SymmetricCov3::from_diagonal(1.0, 0.0, 1.0);
    assert_eq!(
        check_value(&fix(identity(), singular)),
        Err(RejectionReason::ImplausibleCovariance)
    );
    let pose = |position_covariance, attitude_covariance| ObservationValue::VisualPose {
        position: GeodeticPosition::new(0.8, 0.1, 400.0),
        position_covariance,
        attitude: identity(),
        attitude_covariance,
    };
    assert_eq!(check_value(&pose(cov(), cov())), Ok(()));
    assert_eq!(
        check_value(&pose(singular, cov())),
        Err(RejectionReason::ImplausibleCovariance)
    );
    assert_eq!(
        check_value(&pose(cov(), singular)),
        Err(RejectionReason::ImplausibleCovariance)
    );
}

#[test]
fn cartesian_values_need_finite_components() {
    let position = ObservationValue::CartesianPositionFix {
        position: CartesianPosition::new(1.0, f64::NAN, 0.0),
        covariance: cov(),
    };
    let velocity = ObservationValue::CartesianVelocityFix {
        velocity: CartesianVelocity::new(1.0, 0.0, 0.0),
        covariance: cov(),
    };
    let range = ObservationValue::CartesianRange {
        beacon: CartesianPosition::new(0.0, 0.0, 0.0),
        range_m: -1.0,
        variance_m2: 1.0,
    };
    assert_eq!(check_value(&position), Err(RejectionReason::NonFiniteValue));
    assert_eq!(check_value(&velocity), Ok(()));
    assert_eq!(check_value(&range), Err(RejectionReason::NonFiniteValue));
}
