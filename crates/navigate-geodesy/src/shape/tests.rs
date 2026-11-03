//! Ray–ellipsoid intersection on the Earth, the Moon, and Mars: nadir,
//! oblique, miss, and the refusals.

use navigate_contract::{FrameId, GeodeticPosition};

use super::*;
use crate::ecef::geodetic_to_body_fixed;
use crate::float_math::{cos, sin};

const BODIES: [BodyEllipsoid; 3] = [
    BodyEllipsoid::WGS84,
    BodyEllipsoid::MOON_PA,
    BodyEllipsoid::MARS_IAU,
];

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(v: [f64; 3], k: f64) -> [f64; 3] {
    [v[0] * k, v[1] * k, v[2] * k]
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    scale(v, 1.0 / norm(v))
}

/// The geodetic up, east, and north unit vectors at a latitude and a
/// longitude, in body-fixed axes.
fn local_axes(latitude_rad: f64, longitude_rad: f64) -> [[f64; 3]; 3] {
    let (sl, cl) = (sin(latitude_rad), cos(latitude_rad));
    let (so, co) = (sin(longitude_rad), cos(longitude_rad));
    [
        [cl * co, cl * so, sl],
        [-so, co, 0.0],
        [-sl * co, -sl * so, cl],
    ]
}

#[test]
fn a_nadir_ray_hits_the_equator_and_the_pole_at_the_altitude() {
    for body in BODIES {
        let a = body.semi_major_axis_m();
        let b = body.semi_minor_axis_m();
        let equator = body
            .intersect_ray([a + 400_000.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
            .expect("nadir hit");
        assert!(
            (equator.range_m - 400_000.0).abs() < 1e-6,
            "{:?}",
            body.frame()
        );
        assert!((equator.point_body_fixed_m[0] - a).abs() < 1e-6);
        assert_eq!(equator.outward_normal_unit, [1.0, 0.0, 0.0]);
        let pole = body
            .intersect_ray([0.0, 0.0, b + 25_000.0], [0.0, 0.0, -1.0])
            .expect("nadir hit");
        assert!((pole.range_m - 25_000.0).abs() < 1e-6);
        assert!((pole.point_body_fixed_m[2] - b).abs() < 1e-6);
        assert_eq!(pole.outward_normal_unit, [0.0, 0.0, 1.0]);
    }
}

#[test]
fn an_oblique_ray_hits_its_target_with_the_geodetic_normal() {
    for body in BODIES {
        for (lat_deg, lon_deg) in [(45.0_f64, 30.0_f64), (-62.5, -141.0), (5.0, 179.0)] {
            let (lat, lon) = (lat_deg.to_radians(), lon_deg.to_radians());
            let target =
                geodetic_to_body_fixed(&GeodeticPosition::new(lat, lon, 0.0), body).to_array();
            let [up, east, north] = local_axes(lat, lon);
            // A sensor 300 km away at an elevation of about 63°.
            let offset = scale(
                unit([up[0] + 0.5 * east[0], up[1] + 0.5 * east[1], up[2]]),
                3e5,
            );
            let origin = [
                target[0] + offset[0],
                target[1] + offset[1],
                target[2] + offset[2],
            ];
            let hit = body
                .intersect_ray(origin, unit(sub(target, origin)))
                .expect("oblique hit");
            assert!((hit.range_m - 3e5).abs() < 1e-6, "{:?} range", body.frame());
            assert!(norm(sub(hit.point_body_fixed_m, target)) < 1e-6);
            let normal = hit.outward_normal_unit;
            assert!((norm(normal) - 1.0).abs() < 1e-15);
            // The ellipsoidal normal is the geodetic up, and it is
            // orthogonal to the surface tangents.
            assert!(norm(sub(normal, up)) < 1e-12, "{:?} normal", body.frame());
            assert!(dot(normal, east).abs() < 1e-12);
            assert!(dot(normal, north).abs() < 1e-12);
        }
    }
}

#[test]
fn a_ray_that_passes_beside_or_away_from_the_body_misses() {
    for body in BODIES {
        let outside = [body.semi_major_axis_m() + 1_000.0, 0.0, 0.0];
        assert_eq!(body.intersect_ray(outside, [0.0, 1.0, 0.0]), None, "beside");
        assert_eq!(body.intersect_ray(outside, [1.0, 0.0, 0.0]), None, "away");
        // A ray toward the body whose closest approach is outside it.
        let far = [3.0 * body.semi_major_axis_m(), 0.0, 0.0];
        let grazing = unit([-1.0, 0.6, 0.0]);
        assert_eq!(body.intersect_ray(far, grazing), None, "wide");
    }
}

#[test]
fn an_origin_inside_the_body_and_a_bad_direction_are_refused() {
    let body = BodyEllipsoid::MARS_IAU;
    assert_eq!(body.intersect_ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]), None);
    let on_surface = [body.semi_major_axis_m(), 0.0, 0.0];
    assert_eq!(body.intersect_ray(on_surface, [-1.0, 0.0, 0.0]), None);
    let outside = [body.semi_major_axis_m() + 10.0, 0.0, 0.0];
    assert_eq!(body.intersect_ray(outside, [-2.0, 0.0, 0.0]), None);
    assert_eq!(body.intersect_ray(outside, [f64::NAN, 0.0, 0.0]), None);
    assert_eq!(
        body.intersect_ray([f64::INFINITY, 0.0, 0.0], [-1.0, 0.0, 0.0]),
        None
    );
    assert!(
        body.intersect_ray(outside, [-1.0 + 1e-9, 0.0, 0.0])
            .is_some()
    );
}

#[test]
fn the_shape_frame_is_the_ellipsoid_frame() {
    let shapes: [&dyn SurfaceShape; 3] = [
        &BodyEllipsoid::WGS84,
        &BodyEllipsoid::MOON_PA,
        &BodyEllipsoid::MARS_IAU,
    ];
    let frames: [FrameId; 3] = shapes.map(|shape| shape.frame());
    assert_eq!(
        frames,
        [FrameId::EARTH_ITRF93, FrameId::MOON_PA, FrameId::MARS_IAU]
    );
}
