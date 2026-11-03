use super::*;
use crate::wgs84;

const EARTH: BodyEllipsoid = BodyEllipsoid::WGS84;

fn deg(latitude_deg: f64, longitude_deg: f64) -> GeodeticPosition {
    GeodeticPosition::new(latitude_deg.to_radians(), longitude_deg.to_radians(), 0.0)
}

#[test]
fn lax_to_jfk_matches_the_formulary_within_half_a_percent() {
    // Aviation Formulary (Ed Williams) worked example: LAX
    // (33°57'N, 118°24'W) to JFK (40°38'N, 73°47'W) is 0.623585 rad
    // ≈ 2144 nm ≈ 3_970_688 m.
    let lax = deg(33.95, -118.4);
    let jfk = deg(40.0 + 38.0 / 60.0, -(73.0 + 47.0 / 60.0));
    let reference_m = 3_970_688.0;
    let measured_m = distance_m(&lax, &jfk, EARTH);
    assert!(
        ((measured_m - reference_m) / reference_m).abs() < 0.005,
        "distance {measured_m} m vs reference {reference_m} m"
    );
    // Same worked example: the initial true course is ≈ 66°.
    let bearing_deg = initial_bearing_rad(&lax, &jfk).to_degrees();
    assert!(
        (bearing_deg - 65.89).abs() < 0.5,
        "bearing {bearing_deg}° vs reference 66°"
    );
}

#[test]
fn bearing_stays_in_zero_to_two_pi() {
    let north = initial_bearing_rad(&deg(0.0, 0.0), &deg(1.0, 0.0));
    let west = initial_bearing_rad(&deg(0.0, 0.0), &deg(0.0, -1.0));
    let south = initial_bearing_rad(&deg(1.0, 0.0), &deg(0.0, 0.0));
    assert!(north.abs() < 1e-12);
    assert!((west - 270.0_f64.to_radians()).abs() < 1e-12);
    assert!((south - 180.0_f64.to_radians()).abs() < 1e-9);
    let coincident = initial_bearing_rad(&deg(10.0, 10.0), &deg(10.0, 10.0));
    assert!(coincident.abs() < 1e-15);
}

#[test]
fn cross_track_sign_is_positive_right_of_track() {
    // Eastbound track along the equator; right of track is south.
    let start = deg(0.0, 0.0);
    let end = deg(0.0, 1.0);
    let south = cross_track_m(&deg(-0.1, 0.5), &start, &end, EARTH).expect("valid track");
    let north = cross_track_m(&deg(0.1, 0.5), &start, &end, EARTH).expect("valid track");
    // 0.1° of arc on the mean sphere is ≈ 11_119.5 m.
    assert!(
        (south - 11_119.5).abs() < 1.0,
        "south of eastbound track must be positive: {south}"
    );
    assert!(
        (north + 11_119.5).abs() < 1.0,
        "north of eastbound track must be negative: {north}"
    );
}

#[test]
fn along_track_is_signed_from_the_start() {
    let start = deg(0.0, 0.0);
    let end = deg(0.0, 1.0);
    let ahead = along_track_m(&deg(0.1, 0.5), &start, &end, EARTH).expect("valid track");
    let behind = along_track_m(&deg(0.0, -0.1), &start, &end, EARTH).expect("valid track");
    // 0.5° along the equator ≈ 55_597.5 m; 0.1° behind ≈ −11_119.5 m.
    assert!((ahead - 55_597.5).abs() < 1.0, "ahead: {ahead}");
    assert!((behind + 11_119.5).abs() < 1.0, "behind: {behind}");
}

#[test]
fn point_on_track_has_zero_cross_track() {
    let start = deg(0.0, 0.0);
    let end = deg(0.0, 1.0);
    let on_track = cross_track_m(&deg(0.0, 0.4), &start, &end, EARTH).expect("valid track");
    assert!(on_track.abs() < 1e-6, "on-track point: {on_track}");
}

#[test]
fn a_course_line_agrees_with_the_two_point_track_that_carries_it() {
    // The course line through the track's end, carrying the track's
    // own bearing at that end, IS the track's great circle, so both
    // helpers must read the same deviation with the same sign.
    let start = deg(0.0, 0.0);
    let end = deg(0.0, 1.0);
    let course = initial_bearing_rad(&end, &deg(0.0, 2.0));
    for point in [deg(-0.1, 0.5), deg(0.1, 0.5), deg(-0.05, 1.4)] {
        let track = cross_track_m(&point, &start, &end, EARTH).expect("valid track");
        let line = cross_track_from_course_m(&point, &end, course, EARTH);
        assert!(
            (track - line).abs() < 1e-6,
            "track {track} vs course line {line}"
        );
    }
}

#[test]
fn a_course_line_is_positive_right_of_course() {
    // Eastbound course through (0°, 1°E); right of it is south.
    let fix = deg(0.0, 1.0);
    let eastbound = core::f64::consts::FRAC_PI_2;
    let south = cross_track_from_course_m(&deg(-0.1, 0.5), &fix, eastbound, EARTH);
    let north = cross_track_from_course_m(&deg(0.1, 0.5), &fix, eastbound, EARTH);
    assert!((south - 11_119.5).abs() < 1.0, "south of course: {south}");
    assert!((north + 11_119.5).abs() < 1.0, "north of course: {north}");
}

#[test]
fn a_course_line_has_no_degenerate_case_to_refuse() {
    // The anchor itself lies on its own course line, and so does a
    // point directly ahead of it — neither refuses, where a
    // two-point track of zero length must.
    let fix = deg(0.0, 20.0);
    let eastbound = core::f64::consts::FRAC_PI_2;
    assert!(cross_track_from_course_m(&fix, &fix, eastbound, EARTH).abs() < 1e-9);
    // An eastbound course line from the equator is the equator.
    let ahead = cross_track_from_course_m(&deg(0.0, 20.5), &fix, eastbound, EARTH);
    assert!(ahead.abs() < 1e-6, "a point on the course line: {ahead}");
}

#[test]
fn zero_length_track_is_refused_as_degenerate() {
    let start = deg(10.0, 20.0);
    let point = deg(10.5, 20.5);
    let cross = cross_track_m(&point, &start, &start, EARTH);
    let along = along_track_m(&point, &start, &start, EARTH);
    assert!(
        matches!(cross, Err(GeodesyError::DegenerateTrack { separation_m }) if separation_m < 1e-3),
        "cross-track on a zero-length track: {cross:?}"
    );
    assert!(
        matches!(along, Err(GeodesyError::DegenerateTrack { separation_m }) if separation_m < 1e-3),
        "along-track on a zero-length track: {along:?}"
    );
}

#[test]
fn wgs84_helpers_match_the_explicit_wgs84_forms() {
    let a = deg(10.0, 20.0);
    let b = deg(11.0, 21.5);
    let p = deg(10.4, 20.9);
    let earth = BodyEllipsoid::WGS84;
    assert_eq!(wgs84::distance_m(&a, &b), distance_m(&a, &b, earth));
    assert_eq!(
        wgs84::cross_track_m(&p, &a, &b).expect("valid track"),
        cross_track_m(&p, &a, &b, earth).expect("valid track")
    );
    assert_eq!(
        wgs84::along_track_m(&p, &a, &b).expect("valid track"),
        along_track_m(&p, &a, &b, earth).expect("valid track")
    );
    assert_eq!(
        wgs84::cross_track_from_course_m(&p, &a, 0.7),
        cross_track_from_course_m(&p, &a, 0.7, earth)
    );
}

#[test]
fn lunar_distances_use_the_lunar_radius() {
    let moon = BodyEllipsoid::MOON_ME;
    // A quarter of a great circle on the lunar sphere.
    let quarter = distance_m(&deg(0.0, 0.0), &deg(0.0, 90.0), moon);
    assert!((quarter - 1_737_400.0 * core::f64::consts::FRAC_PI_2).abs() < 1e-6);
    // 0.1° of arc on the Moon is ≈ 3_032.3 m.
    let start = deg(0.0, 0.0);
    let end = deg(0.0, 1.0);
    let south = cross_track_m(&deg(-0.1, 0.5), &start, &end, moon).expect("valid track");
    assert!(
        (south - 3_032.3).abs() < 0.1,
        "south of lunar track: {south}"
    );
    let ahead = along_track_m(&deg(0.1, 0.5), &start, &end, moon).expect("valid track");
    assert!(
        (ahead - 15_161.6).abs() < 0.2,
        "ahead on lunar track: {ahead}"
    );
    let course =
        cross_track_from_course_m(&deg(0.1, 0.5), &end, core::f64::consts::FRAC_PI_2, moon);
    assert!(
        (course + 3_032.3).abs() < 0.1,
        "north of lunar course: {course}"
    );
}

#[test]
fn deviations_scale_with_the_body_mean_radius() {
    let start = deg(-4.5, 137.4);
    let end = deg(-4.0, 138.0);
    let point = deg(-4.4, 137.9);
    let earth = cross_track_m(&point, &start, &end, EARTH).expect("valid track");
    for body in [BodyEllipsoid::MARS_IAU, BodyEllipsoid::MOON_PA] {
        let on_body = cross_track_m(&point, &start, &end, body).expect("valid track");
        let ratio = on_body / earth;
        let expected = body.mean_radius_m() / BodyEllipsoid::WGS84.mean_radius_m();
        assert!((ratio - expected).abs() < 1e-12, "{ratio} vs {expected}");
        let along = along_track_m(&point, &start, &end, body).expect("valid track");
        let earth_along = along_track_m(&point, &start, &end, EARTH).expect("valid track");
        assert!((along / earth_along - expected).abs() < 1e-12);
    }
}

#[test]
fn a_lunar_track_shorter_than_a_millimeter_is_degenerate() {
    let moon = BodyEllipsoid::MOON_ME;
    let start = deg(10.0, 20.0);
    // 1e-8° of arc on the Moon is ≈ 0.3 mm.
    let end = deg(10.0, 20.000_000_01);
    assert!(matches!(
        cross_track_m(&deg(10.5, 20.5), &start, &end, moon),
        Err(GeodesyError::DegenerateTrack { .. })
    ));
}
