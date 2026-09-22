#![allow(clippy::expect_used)]
use super::*;

#[test]
fn geographic_coordinates_round_trip_in_renderer_frame() {
    let frame = MapFrame::new([47.3239, 11.4917]);
    let expected = [47.3242, 11.4928, 1650.0];
    let point = frame.local(expected).expect("valid coordinates");
    let [lon, lat, alt] = frame.longitude_latitude_altitude(point);
    assert!((lat - expected[0]).abs() < 1e-10);
    assert!((lon - expected[1]).abs() < 1e-10);
    assert_eq!(alt, expected[2]);
    assert!(frame.local([90.0, 0.0, 1000.0]).is_err());
}
