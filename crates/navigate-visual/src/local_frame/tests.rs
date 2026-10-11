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

#[test]
fn spherical_frame_keeps_curvature_in_cartesian_height() {
    let frame = LocalFrame::spherical_enu(0.0, 0.0).unwrap();
    let angle = 0.01_f64;
    let point = frame.local([0.0, angle.to_degrees(), 0.0]).unwrap();
    assert!((point.x - MERCATOR_SPHERE_RADIUS_M * angle.sin()).abs() < 1e-6);
    assert!((point.z - MERCATOR_SPHERE_RADIUS_M * (angle.cos() - 1.0)).abs() < 1e-6);
    assert!(point.z < -300.0);
    assert!(frame.geodetic(point)[2].abs() < 1e-6);
}

#[test]
fn spherical_positions_and_local_attitudes_round_trip_across_the_date_line() {
    let frame = LocalFrame::spherical_enu(48.0, 179.9).unwrap();
    for location in [[48.2, -179.9, 1500.0], [47.7, 179.4, -400.0]] {
        let point = frame.local(location).unwrap();
        let recovered = frame.geodetic(point);
        assert!((recovered[0] - location[0]).abs() < 1e-10);
        assert!((recovered[1] - location[1]).abs() < 1e-10);
        assert!((recovered[2] - location[2]).abs() < 1e-6);
        let rotation = frame.local_rotation([location[0], location[1]]).unwrap();
        let higher = frame
            .local([location[0], location[1], location[2] + 10.0])
            .unwrap();
        assert!((higher - point - rotation * Vector3::new(0.0, 0.0, 10.0)).norm() < 1e-6);
        let pixels = frame.mercator_xy(point);
        assert!((pixels[0] - (location[1] + 180.0) / 360.0).abs() < 1e-10);
    }
}

#[test]
fn spherical_model_error_retains_the_ellipsoid_difference() {
    let frame = LocalFrame::spherical_enu(0.0, 0.0).unwrap();
    assert!(frame.model_error_m(Vector3::zeros()) < 1e-6);
    let point = frame.local([0.0, 0.5, 0.0]).unwrap();
    let expected = 2.0 * (6_378_137.0 - MERCATOR_SPHERE_RADIUS_M) * (0.25_f64.to_radians()).sin();
    assert!((frame.model_error_m(point) - expected).abs() < 1e-6);
}

#[test]
fn spherical_coordinates_recover_the_frozen_photo_altitude() {
    let frame = LocalFrame::spherical_enu(41.075, -75.275).unwrap();
    let position = Vector3::new(-6100.954727028143, 4073.292560812611, 925.9738818901951);
    let actual = frame.geodetic(position);
    let expected = [41.111603701005166, -75.34781249167567, 930.1965665966272];
    assert!((actual[0] - expected[0]).abs() < 1e-10);
    assert!((actual[1] - expected[1]).abs() < 1e-10);
    assert!((actual[2] - expected[2]).abs() < 1e-6);
}
