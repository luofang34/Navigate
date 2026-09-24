#![allow(clippy::expect_used)]
use super::*;

#[test]
fn range_jacobian_points_from_station_to_vehicle() {
    let model = RangeModel::new(Vector3::new(1000.0, 0.0, 0.0), 1000.0, 25.0);
    let x = Vec6::zeros();
    let h = model.jacobian(&x).expect("defined");
    assert!((h[(0, 0)] + 1.0).abs() < 1e-12);
    assert_eq!(model.innovation(&x)[0], 0.0);
}

#[test]
fn range_jacobian_is_undefined_at_the_station() {
    let model = RangeModel::new(Vector3::zeros(), 0.0, 25.0);
    assert!(model.jacobian(&Vec6::zeros()).is_none());
}
