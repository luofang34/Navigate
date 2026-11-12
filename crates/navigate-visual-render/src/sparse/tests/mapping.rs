//! The modelled map coordinates against the exact conversion, and a view
//! rendered with them against the exact view.

use super::super::mapping::{self, MapCoordinates};
use super::*;

/// Points spread through a ball of radius `reach_m` about `eye`, from a
/// fixed sequence.
fn ball(eye: Vector3<f64>, reach_m: f64) -> Vec<Vector3<f64>> {
    let mut state = 0x9E37_79B9_u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        f64::from(state) / f64::from(u32::MAX) * 2.0 - 1.0
    };
    (0..2_000)
        .filter_map(|_| {
            let v = Vector3::new(next(), next(), next());
            (v.norm() <= 1.0).then_some(eye + v * reach_m)
        })
        .collect()
}

#[test]
fn the_model_agrees_with_the_conversion_through_its_reach() {
    for (lat, lon) in [
        (0.0_f64, 0.0),
        (46.95, 7.45),
        (-33.9, 151.2),
        (69.6, 18.9),
        (77.0, 15.6),
    ] {
        for frame in [
            LocalFrame::wgs84_enu(lat, lon).unwrap(),
            LocalFrame::spherical_enu(lat, lon).unwrap(),
        ] {
            for eye in [
                Vector3::new(0.0, 0.0, 1_760.0),
                Vector3::new(4_000.0, -3_000.0, 400.0),
            ] {
                let map = MapCoordinates::about(frame, eye, 6_000.0);
                assert!(matches!(map, MapCoordinates::Model(_)), "{frame:?}");
                let metres = 2.0 * std::f64::consts::PI * 6_378_137.0 * lat.to_radians().cos();
                for point in ball(eye, 6_000.0) {
                    let ([mx, my], alt) = mapping::exact(frame, point);
                    let ([px, py], palt) = map.at(point);
                    let ground = (mx - px).hypot(my - py) * metres;
                    assert!(ground < 0.05, "{frame:?} {point:?}: {ground}");
                    assert!((alt - palt).abs() < 0.05, "{frame:?} {point:?}");
                }
            }
        }
    }
}

#[test]
fn a_frame_the_model_cannot_hold_converts_exactly() {
    let eye = Vector3::new(0.0, 0.0, 1_000.0);
    let far_north = LocalFrame::wgs84_enu(84.0, 0.0).unwrap();
    assert!(matches!(
        MapCoordinates::about(far_north, eye, 6_000.0),
        MapCoordinates::Exact(_)
    ));
    let antimeridian = LocalFrame::wgs84_enu(10.0, 179.99).unwrap();
    assert!(matches!(
        MapCoordinates::about(antimeridian, eye, 6_000.0),
        MapCoordinates::Exact(_)
    ));
    let flat = LocalFrame::anchor_mercator(47.0, 8.0).unwrap();
    assert!(matches!(
        MapCoordinates::about(flat, eye, 6_000.0),
        MapCoordinates::Exact(_)
    ));
}

/// A scene of rolling hills under a texture of fine detail and sharp
/// edges, about 3.3 km on a side.
fn hills() -> RenderScene {
    let frame = LocalFrame::wgs84_enu(46.95, 7.45).unwrap();
    let mut scene = make_scene(frame, 0.0);
    let layout = centered_layout(frame, 14);
    scene.imagery.layout = layout;
    scene.terrain.layout = layout;
    for (idx, (p, h)) in scene
        .imagery
        .pixels
        .iter_mut()
        .zip(scene.terrain.heights_m.iter_mut())
        .enumerate()
    {
        let (col, row) = ((idx % 512) as f64, (idx / 512) as f64);
        let texture = 128.0 + 60.0 * (col / 3.1).sin() * (row / 4.3).cos();
        let field = if (col / 40.0).floor() as i64 % 2 == 0 {
            30.0
        } else {
            -30.0
        };
        *p = (texture + field).clamp(0.0, 255.0) as u8;
        *h = (200.0 + 150.0 * (col / 37.0).sin() * (row / 53.0).cos()) as f32;
    }
    scene
}

#[test]
fn a_modelled_view_matches_the_exact_view_pixel_for_pixel() {
    let scene = hills();
    let modelled = SparseSceneSampler::new(&scene, 6_000.0, 32, 10).unwrap();
    let exact = SparseSceneSampler {
        modelled: false,
        ..modelled
    };
    let cam = CameraModel {
        width: 160,
        height: 120,
        fx: 120.0,
        fy: 120.0,
        cx: 79.5,
        cy: 59.5,
    };
    let ratio = (6_000.0_f32 / 10.0).powf(1.0 / 32.0);
    let bisection = (ratio - 1.0) / 1024.0 * 1.01;
    for (position, tilt) in [
        (Vector3::new(0.0, 0.0, 1_760.0), Vector3::new(0.0, 0.0, 0.0)),
        (
            Vector3::new(150.0, -200.0, 1_200.0),
            Vector3::new(0.5, 0.2, 0.3),
        ),
    ] {
        let pose = CameraPose {
            position,
            orientation: UnitQuaternion::from_scaled_axis(tilt),
        };
        for stride in [1, 4] {
            let (a, _) = modelled
                .render_adaptive(&cam, &pose, stride, 4, 0.015)
                .unwrap();
            let (b, _) = exact
                .render_adaptive(&cam, &pose, stride, 4, 0.015)
                .unwrap();
            let pairs = a.image.as_raw().iter().zip(b.image.as_raw());
            let differing = pairs.filter(|(x, y)| x != y).count();
            let largest = a
                .image
                .as_raw()
                .iter()
                .zip(b.image.as_raw())
                .map(|(x, y)| x.abs_diff(*y))
                .max()
                .unwrap();
            let depth = a
                .depth_m
                .iter()
                .zip(&b.depth_m)
                .map(|(x, y)| ((x - y) / y.max(1.0)).abs())
                .fold(0.0_f32, f32::max);
            let seen = b.depth_m.iter().filter(|d| **d > 0.0).count();
            assert!(seen > 160 * 120 / 2, "{seen}");
            assert!(
                largest <= 1,
                "{position:?} {stride}: {largest} {differing} {depth}"
            );
            assert!(differing * 200 < 160 * 120, "{differing}");
            // A hit can land one bisection interval apart when the model
            // tips the last comparison of the march.
            assert!(depth <= bisection, "{depth} {bisection}");
        }
    }
}
