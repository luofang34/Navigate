use super::*;

#[test]
fn external_checks_validate_metadata_without_claiming_sensor_accuracy() {
    let mut check = PositionCheck {
        source: "photo EXIF GPS".into(),
        latitude_longitude: [40.0, -75.0],
        reported_horizontal_error_m: Some(8000.0),
        associated_capture_time_ns: 100,
    };
    assert!(check.valid());
    let missing = check.compare(125, None);
    assert!(missing["horizontal_disagreement_m"].is_null());
    assert_eq!(missing["capture_time_difference_ns"], 25);
    assert_eq!(missing["reported_horizontal_error_m"], 8000.0);
    check.reported_horizontal_error_m = None;
    assert!(check.valid());
    check.reported_horizontal_error_m = Some(-1.0);
    assert!(!check.valid());
    check.reported_horizontal_error_m = Some(f64::NAN);
    assert!(!check.valid());
    check.reported_horizontal_error_m = None;
    check.latitude_longitude = [91.0, 0.0];
    assert!(!check.valid());
    check.latitude_longitude = [0.0, 181.0];
    assert!(!check.valid());
    check.latitude_longitude = [0.0, 0.0];
    check.source = " ".into();
    assert!(!check.valid());
}

#[test]
fn horizontal_comparison_handles_the_dateline_and_poles() {
    assert_eq!(horizontal_distance([40.0, -75.0], [40.0, -75.0]), 0.0);
    let distance = horizontal_distance([0.0, 179.999], [0.0, -179.999]);
    assert!((distance - 222.39016).abs() < 0.001);
    let quarter = horizontal_distance([0.0, 0.0], [90.0, 0.0]);
    assert!((quarter - 6_371_008.8 * std::f64::consts::FRAC_PI_2).abs() < 1e-6);
}
