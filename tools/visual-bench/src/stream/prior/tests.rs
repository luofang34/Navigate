#![allow(clippy::expect_used)]
use super::*;

#[test]
fn heading_and_tilt_place_the_camera_optical_axis() {
    let frame = navigate_visual::LocalFrame::anchor_mercator(47.0, 11.0).expect("valid anchor");
    let mut record = PriorRecord {
        position_enu_m: None,
        geodetic_lat_lon_alt_m: Some([47.0, 11.0, 1000.0]),
        eye_to_enu_xyzw: None,
        heading_tilt_roll_deg: Some([90.0, 90.0, 0.0]),
        position_radius_m: 100.0,
        attitude_radius_rad: 0.1,
    };
    let pose = record.prior(frame).expect("valid prior").pose;
    assert!((pose.orientation * -Vector3::z() - Vector3::x()).norm() < 1e-10);
    assert!((pose.position - Vector3::new(0.0, 0.0, 1000.0)).norm() < 1e-10);
    record.position_enu_m = Some([0.0; 3]);
    assert!(record.prior(frame).is_err());
}
