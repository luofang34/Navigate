#![allow(clippy::expect_used)]
use super::*;

fn innsbruck() -> LocalFrame {
    LocalFrame::anchor_mercator(47.3239, 11.4917).expect("valid anchor")
}

#[test]
fn geodetic_round_trip_is_exact_near_the_anchor() {
    let frame = innsbruck();
    let expected = [47.3242, 11.4928, 1650.0];
    let point = frame.local(expected).expect("valid position");
    let [lat, lon, alt] = frame.geodetic(point);
    assert!((lat - expected[0]).abs() < 1e-10);
    assert!((lon - expected[1]).abs() < 1e-10);
    assert_eq!(alt, expected[2]);
}

#[test]
fn geodetic_round_trip_is_exact_at_thirty_kilometres() {
    let frame = innsbruck();
    let point = Vector3::new(21_000.0, -21_000.0, 2400.0);
    let geodetic = frame.geodetic(point);
    let back = frame.local(geodetic).expect("valid position");
    assert!((back - point).norm() < 1e-6);
}

#[test]
fn model_error_grows_with_range_and_is_zero_at_the_origin() {
    let frame = innsbruck();
    assert_eq!(frame.model_error_m(Vector3::new(0.0, 0.0, 900.0)), 0.0);
    let near = frame.model_error_m(Vector3::new(0.0, 1000.0, 0.0));
    let far = frame.model_error_m(Vector3::new(0.0, 50_000.0, 0.0));
    assert!(near < 1.0);
    // Curvature drop alone at 50 km is about 196 m.
    assert!(far > 196.0);
}

#[test]
fn rejects_positions_outside_the_mercator_limit() {
    let frame = innsbruck();
    assert!(frame.local([90.0, 0.0, 1000.0]).is_err());
    assert!(frame.local([f64::NAN, 0.0, 1000.0]).is_err());
    assert!(LocalFrame::anchor_mercator(86.0, 0.0).is_err());
}

#[test]
fn mercator_xy_matches_the_anchor_at_the_origin() {
    let frame = innsbruck();
    let [x, y] = frame.mercator_xy(Vector3::zeros());
    assert!((x - (11.4917 + 180.0) / 360.0).abs() < 1e-12);
    assert!(y > 0.0 && y < 0.5);
}
