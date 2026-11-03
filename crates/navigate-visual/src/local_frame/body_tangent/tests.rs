use crate::LocalFrame;
use nalgebra::Vector3;
use navigate_contract::GeodeticPosition;
use navigate_geodesy::BodyEllipsoid;

const BODIES: [BodyEllipsoid; 3] = [
    BodyEllipsoid::WGS84,
    BodyEllipsoid::MOON_PA,
    BodyEllipsoid::MARS_IAU,
];

fn frame(ellipsoid: BodyEllipsoid, [lat, lon, height]: [f64; 3]) -> LocalFrame {
    let origin = GeodeticPosition::new(lat.to_radians(), lon.to_radians(), height);
    LocalFrame::body_tangent_enu(ellipsoid, origin).unwrap()
}

/// Horizontal distance between two geodetic points in metres, with the longitude
/// error scaled by the latitude, so that a pole has one position.
fn surface_error_m(ellipsoid: BodyEllipsoid, a: [f64; 3], b: [f64; 3]) -> f64 {
    let radius = ellipsoid.semi_major_axis_m();
    let north = (a[0] - b[0]).to_radians() * radius;
    let lon = (a[1] - b[1] + 180.0).rem_euclid(360.0) - 180.0;
    let east = lon.to_radians() * a[0].to_radians().cos() * radius;
    north.hypot(east).hypot(a[2] - b[2])
}

#[test]
fn exact_poles_round_trip_on_the_earth_the_moon_and_mars() {
    for ellipsoid in BODIES {
        for pole in [90.0, -90.0] {
            let frame = frame(ellipsoid, [pole, 30.0, 0.0]);
            assert_eq!(frame.reference_ellipsoid(), ellipsoid);
            assert!(frame.local([pole, 30.0, 0.0]).unwrap().norm() < 1e-6);
            let inside = pole - pole.signum() * 0.5;
            for location in [
                [pole, 0.0, 2_000.0],
                [pole, -150.0, -1_000.0],
                [inside, 120.0, 15_000.0],
                [inside, -60.0, 0.0],
            ] {
                let point = frame.local(location).unwrap();
                let recovered = frame.geodetic(point);
                assert!(surface_error_m(ellipsoid, location, recovered) < 1e-3);
                let back = frame.local(recovered).unwrap();
                assert!((back - point).norm() < 1e-3, "{ellipsoid:?} {location:?}");
                assert_eq!(frame.model_error_m(point), 0.0);
            }
        }
    }
}

#[test]
fn a_track_across_the_lunar_south_pole_stays_continuous() {
    let moon = BodyEllipsoid::MOON_PA;
    let frame = frame(moon, [-89.9, 30.0, 0.0]);
    let radius = moon.semi_major_axis_m();
    // A meridian through the pole: 0.1 degrees on each side of the pole.
    let near = frame.local([-89.9, 30.0, 0.0]).unwrap();
    let pole = frame.local([-90.0, 30.0, 0.0]).unwrap();
    let far = frame.local([-89.9, -150.0, 0.0]).unwrap();
    let step = 0.1_f64.to_radians() * radius;
    assert!(((pole - near).norm() - step).abs() < 1.0);
    assert!(((far - pole).norm() - step).abs() < 1.0);
    // The pole is south of the origin, and the far point continues on the same axis.
    assert!((far.x - near.x).abs() < 1e-3 && far.y < pole.y && pole.y < near.y);
}

#[test]
fn local_rotation_at_a_pole_is_the_up_axis_of_the_frame_point() {
    for ellipsoid in BODIES {
        let frame = frame(ellipsoid, [-89.9, 30.0, 0.0]);
        for location in [[-90.0, 75.0], [-89.95, -100.0]] {
            let rotation = frame.local_rotation(location).unwrap();
            let base = frame.local([location[0], location[1], 0.0]).unwrap();
            let higher = frame.local([location[0], location[1], 10.0]).unwrap();
            assert!((higher - base - rotation * Vector3::new(0.0, 0.0, 10.0)).norm() < 1e-6);
        }
    }
}

#[test]
fn wgs84_body_tangent_frames_agree_with_the_wgs84_tile_frame() {
    let tile = LocalFrame::wgs84_enu(47.0, 11.0).unwrap();
    let body = frame(BodyEllipsoid::WGS84, [47.0, 11.0, 0.0]);
    for location in [[47.05, 11.1, 2_500.0], [46.9, 10.8, -300.0]] {
        assert_eq!(tile.local(location).unwrap(), body.local(location).unwrap());
    }
    let raised = frame(BodyEllipsoid::WGS84, [47.0, 11.0, 1_000.0]);
    let point = raised.local([47.0, 11.0, 1_250.0]).unwrap();
    assert!((point - Vector3::new(0.0, 0.0, 250.0)).norm() < 1e-6);
}

#[test]
fn only_body_tangent_frames_pass_the_tile_latitude_limit() {
    assert!(LocalFrame::wgs84_enu(86.0, 0.0).is_err());
    assert!(
        LocalFrame::wgs84_enu(47.0, 0.0)
            .unwrap()
            .local([89.0, 0.0, 0.0])
            .is_err()
    );
    let polar = frame(BodyEllipsoid::MARS_IAU, [89.0, 0.0, 0.0]);
    assert!(polar.local([90.0, 0.0, 0.0]).is_ok());
    assert!(polar.local([90.5, 0.0, 0.0]).is_err());
    assert!(polar.local([89.0, 181.0, 0.0]).is_err());
    let invalid = GeodeticPosition::new(f64::NAN, 0.0, 0.0);
    assert!(LocalFrame::body_tangent_enu(BodyEllipsoid::MOON_PA, invalid).is_err());
    let beyond = GeodeticPosition::new(1.6, 0.0, 0.0);
    assert!(LocalFrame::body_tangent_enu(BodyEllipsoid::MOON_PA, beyond).is_err());
    assert!(
        frame(BodyEllipsoid::MOON_PA, [0.0, 0.0, 0.0])
            .local([0.0, 0.0, -1_737_400.0])
            .is_err()
    );
}
