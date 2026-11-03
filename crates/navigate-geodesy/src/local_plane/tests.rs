use core::f64::consts::FRAC_PI_2;

use super::*;
use crate::wgs84::geodetic_to_ecef;

fn plane_at_47n_8e() -> LocalTangentPlane {
    let origin = GeodeticPosition::new(47.0_f64.to_radians(), 8.0_f64.to_radians(), 500.0);
    LocalTangentPlane::new(origin, BodyEllipsoid::WGS84).expect("plausible origin")
}

#[test]
fn implausible_origin_is_rejected() {
    let origin = GeodeticPosition::new(2.0, 0.0, 0.0);
    assert!(matches!(
        LocalTangentPlane::new(origin, BodyEllipsoid::WGS84),
        Err(GeodesyError::ImplausibleOrigin { .. })
    ));
}

#[test]
fn origin_maps_to_zero_offset() {
    let plane = plane_at_47n_8e();
    let ned = plane.to_ned(&plane.origin());
    assert!(ned.north_m.abs() < 1e-9);
    assert!(ned.east_m.abs() < 1e-9);
    assert!(ned.down_m.abs() < 1e-9);
}

#[test]
fn point_above_origin_is_pure_negative_down() {
    let plane = plane_at_47n_8e();
    let above = GeodeticPosition::new(47.0_f64.to_radians(), 8.0_f64.to_radians(), 600.0);
    let ned = plane.to_ned(&above);
    assert!(ned.north_m.abs() < 1e-6);
    assert!(ned.east_m.abs() < 1e-6);
    assert!((ned.down_m + 100.0).abs() < 1e-6);
}

#[test]
fn small_northward_step_matches_the_meridian_arc() {
    let plane = plane_at_47n_8e();
    let north = GeodeticPosition::new(47.01_f64.to_radians(), 8.0_f64.to_radians(), 500.0);
    let ned = plane.to_ned(&north);
    // Meridian arc for 0.01° at 47°N is ≈ 1111.7 m; the plane drops
    // below the ellipsoid ahead, so down is small and positive.
    assert!((ned.north_m - 1_111.7).abs() < 1.0);
    assert!(ned.east_m.abs() < 1e-6);
    assert!(ned.down_m > 0.0 && ned.down_m < 0.2);
}

#[test]
fn geodetic_round_trip_within_100_km_stays_below_a_micrometer() {
    let plane = plane_at_47n_8e();
    let candidates = [
        GeodeticPosition::new(47.5_f64.to_radians(), 8.0_f64.to_radians(), 10_500.0),
        GeodeticPosition::new(46.4_f64.to_radians(), 9.0_f64.to_radians(), 0.0),
        GeodeticPosition::new(47.0_f64.to_radians(), 6.9_f64.to_radians(), 3_000.0),
    ];
    for original in candidates {
        let ned = plane.to_ned(&original);
        let round_tripped = plane.from_ned(&ned);
        let error_m = geodetic_to_ecef(&original).distance_m(&geodetic_to_ecef(&round_tripped));
        assert!(error_m < 1e-6, "round trip error {error_m} m");
    }
}

#[test]
fn ned_round_trip_within_100_km_stays_below_a_micrometer() {
    let plane = plane_at_47n_8e();
    let offsets = [
        NedOffset::new(100_000.0, 0.0, 0.0),
        NedOffset::new(-30_000.0, 70_000.0, -9_500.0),
        NedOffset::new(0.0, -100_000.0, 2_000.0),
    ];
    for original in offsets {
        let geodetic = plane.from_ned(&original);
        let round_tripped = plane.to_ned(&geodetic);
        let dn = round_tripped.north_m - original.north_m;
        let de = round_tripped.east_m - original.east_m;
        let dd = round_tripped.down_m - original.down_m;
        let error_m = (dn * dn + de * de + dd * dd).sqrt();
        assert!(error_m < 1e-6, "round trip error {error_m} m");
    }
}

#[test]
fn ned_rotation_between_planes_matches_ecef_transport() {
    let a = plane_at_47n_8e();
    let b = LocalTangentPlane::new(
        GeodeticPosition::new(48.5_f64.to_radians(), 10.2_f64.to_radians(), 800.0),
        BodyEllipsoid::WGS84,
    )
    .expect("plausible origin");
    let r = a.ned_rotation_to(&b);
    let target = GeodeticPosition::new(48.0_f64.to_radians(), 9.0_f64.to_radians(), 1_200.0);
    let delta_a = a.to_ned(&target);
    let origin_b_in_a = a.to_ned(&b.origin());
    let rel_a = [
        delta_a.north_m - origin_b_in_a.north_m,
        delta_a.east_m - origin_b_in_a.east_m,
        delta_a.down_m - origin_b_in_a.down_m,
    ];
    let rotated = [
        r[0][0] * rel_a[0] + r[0][1] * rel_a[1] + r[0][2] * rel_a[2],
        r[1][0] * rel_a[0] + r[1][1] * rel_a[1] + r[1][2] * rel_a[2],
        r[2][0] * rel_a[0] + r[2][1] * rel_a[1] + r[2][2] * rel_a[2],
    ];
    let direct_b = b.to_ned(&target);
    assert!((rotated[0] - direct_b.north_m).abs() < 1e-8);
    assert!((rotated[1] - direct_b.east_m).abs() < 1e-8);
    assert!((rotated[2] - direct_b.down_m).abs() < 1e-8);
}

/// Asserts both round trips of `plane` below a micrometer for offsets up
/// to `reach_m`.
fn assert_plane_round_trips(plane: &LocalTangentPlane, reach_m: f64) {
    let offsets = [
        NedOffset::new(reach_m, 0.0, 0.0),
        NedOffset::new(-0.3 * reach_m, 0.7 * reach_m, -9_500.0),
        NedOffset::new(0.0, -reach_m, 2_000.0),
        NedOffset::new(0.0, 0.0, -1_000.0),
    ];
    let ellipsoid = plane.ellipsoid();
    for original in offsets {
        let geodetic = plane.from_ned(&original);
        let round_tripped = plane.to_ned(&geodetic);
        let dn = round_tripped.north_m - original.north_m;
        let de = round_tripped.east_m - original.east_m;
        let dd = round_tripped.down_m - original.down_m;
        let error_m = (dn * dn + de * de + dd * dd).sqrt();
        assert!(
            error_m < 1e-6,
            "{:?}: NED round trip error {error_m} m for {original:?}",
            ellipsoid.frame()
        );
        let back = plane.from_ned(&round_tripped);
        let geodetic_error_m = geodetic_to_body_fixed(&geodetic, ellipsoid)
            .distance_m(&geodetic_to_body_fixed(&back, ellipsoid));
        assert!(
            geodetic_error_m < 1e-6,
            "geodetic round trip {geodetic_error_m} m"
        );
    }
}

#[test]
fn new_anchors_on_wgs84() {
    assert_eq!(plane_at_47n_8e().ellipsoid(), BodyEllipsoid::WGS84);
}

#[test]
fn lunar_and_martian_planes_round_trip_below_a_micrometer() {
    let origins = [
        GeodeticPosition::new(0.0, 0.0, 0.0),
        GeodeticPosition::new(-0.3, 2.1, 1_200.0),
        GeodeticPosition::new(FRAC_PI_2, 0.0, 100.0),
        GeodeticPosition::new(-FRAC_PI_2, 1.0, -2_000.0),
    ];
    for ellipsoid in [
        BodyEllipsoid::MOON_ME,
        BodyEllipsoid::MOON_PA,
        BodyEllipsoid::MARS_IAU,
    ] {
        for origin in origins {
            let plane = LocalTangentPlane::new(origin, ellipsoid).expect("plausible origin");
            assert_eq!(plane.ellipsoid(), ellipsoid);
            assert_plane_round_trips(&plane, 50_000.0);
        }
    }
}

#[test]
fn a_point_above_a_lunar_origin_is_pure_negative_down() {
    let origin = GeodeticPosition::new(0.2, -0.4, 0.0);
    let plane = LocalTangentPlane::new(origin, BodyEllipsoid::MOON_ME).expect("plausible origin");
    let ned = plane.to_ned(&GeodeticPosition::new(0.2, -0.4, 250.0));
    assert!(ned.north_m.abs() < 1e-6 && ned.east_m.abs() < 1e-6);
    assert!((ned.down_m + 250.0).abs() < 1e-6);
}

#[test]
fn a_lunar_northward_step_follows_the_lunar_radius() {
    let origin = GeodeticPosition::new(0.0, 0.0, 0.0);
    let plane = LocalTangentPlane::new(origin, BodyEllipsoid::MOON_ME).expect("plausible origin");
    let step_rad = 1e-3;
    let ned = plane.to_ned(&GeodeticPosition::new(step_rad, 0.0, 0.0));
    let chord_m = 2.0 * 1_737_400.0 * (0.5 * step_rad).sin();
    let expected_north_m = chord_m * (0.5 * step_rad).cos();
    assert!(
        (ned.north_m - expected_north_m).abs() < 1e-6,
        "{}",
        ned.north_m
    );
}

#[test]
fn the_same_coordinates_differ_between_bodies() {
    let origin = GeodeticPosition::new(0.3, 0.1, 0.0);
    let target = GeodeticPosition::new(0.301, 0.1, 0.0);
    let earth = LocalTangentPlane::new(origin, BodyEllipsoid::WGS84).expect("plausible origin");
    let mars = LocalTangentPlane::new(origin, BodyEllipsoid::MARS_IAU).expect("plausible origin");
    let ratio = earth.to_ned(&target).north_m / mars.to_ned(&target).north_m;
    assert!((ratio - 1.88).abs() < 0.02, "{ratio}");
}

#[test]
fn the_accessors_reconstruct_a_plane_offset_in_the_body_fixed_frame() {
    let plane = plane_at_47n_8e();
    let offset = NedOffset::new(1_200.0, -800.0, 35.0);
    let truth = geodetic_to_ecef(&plane.from_ned(&offset));
    let origin = plane.origin_body_fixed();
    let rows = plane.body_fixed_to_ned_rows();
    let ned = [offset.north_m, offset.east_m, offset.down_m];
    let mut body_fixed = [origin.x_m, origin.y_m, origin.z_m];
    for (axis, value) in body_fixed.iter_mut().enumerate() {
        *value += rows[0][axis] * ned[0] + rows[1][axis] * ned[1] + rows[2][axis] * ned[2];
    }
    assert!((body_fixed[0] - truth.x_m).abs() < 1e-6);
    assert!((body_fixed[1] - truth.y_m).abs() < 1e-6);
    assert!((body_fixed[2] - truth.z_m).abs() < 1e-6);
}
