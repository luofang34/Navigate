use std::sync::Arc;

use nalgebra::{UnitQuaternion, Vector2, Vector3};

use super::*;
use crate::CameraModel;
use image::GrayImage;

/// A hash of a lattice point, in `[0, 1)`.
fn lattice(x: i64, y: i64) -> f64 {
    let mut h = (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (y as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h >> 11) as f64 / (1_u64 << 53) as f64
}

/// Ground of fields some 60 m across, never repeating.
fn fields(p: Vector2<f64>) -> f64 {
    let (gx, gy) = (p.x / 60.0, p.y / 60.0);
    let (x, y) = (gx.floor(), gy.floor());
    let (fx, fy) = (gx - x, gy - y);
    let (x, y) = (x as i64, y as i64);
    let top = lattice(x, y) * (1.0 - fx) + lattice(x + 1, y) * fx;
    let bottom = lattice(x, y + 1) * (1.0 - fx) + lattice(x + 1, y + 1) * fx;
    40.0 + 180.0 * (top * (1.0 - fy) + bottom * fy)
}

/// Blocks repeating every 400 m east and north.
fn blocks(p: Vector2<f64>) -> f64 {
    fields(Vector2::new(p.x.rem_euclid(400.0), p.y.rem_euclid(400.0)))
}

fn map_of(ground: fn(Vector2<f64>) -> f64) -> GroundRaster {
    GroundRaster::from_samples(32.0, 110, |p| Some(ground(p))).unwrap()
}

/// The frame raster of ground `offset` metres from the map centre, with
/// the gain, offset and noise of another sensor.
fn frame_of(ground: fn(Vector2<f64>) -> f64, offset: Vector2<f64>) -> GroundRaster {
    let mut n = 0_i64;
    GroundRaster::from_samples(32.0, 30, |p| {
        n = n.wrapping_add(1);
        Some(0.8 * ground(p + offset) + 12.0 + 6.0 * (lattice(n, 7) - 0.5))
    })
    .unwrap()
}

#[test]
fn a_frame_from_far_off_the_belief_is_found_at_its_offset() {
    let map = map_of(fields);
    for offset in [
        Vector2::new(0.0, 0.0),
        Vector2::new(1_310.0, -705.0),
        Vector2::new(-1_950.0, 400.0),
    ] {
        let found = acquire(
            &frame_of(fields, offset),
            &map,
            2_200.0,
            &AcquisitionConfig::default(),
        )
        .unwrap();
        let Acquisition::Found(peak) = found else {
            panic!("{offset:?}: {found:?}");
        };
        assert!(
            (peak.offset_m - offset).norm() < 32.0,
            "{offset:?}: {peak:?}"
        );
        assert!(peak.score > 0.9, "{peak:?}");
    }
}

#[test]
fn ground_that_repeats_is_ambiguous() {
    let found = acquire(
        &frame_of(blocks, Vector2::new(620.0, 210.0)),
        &map_of(blocks),
        2_200.0,
        &AcquisitionConfig::default(),
    )
    .unwrap();
    assert!(matches!(found, Acquisition::Ambiguous(_)), "{found:?}");
}

#[test]
fn featureless_ground_or_ground_outside_the_search_is_weak() {
    let flat = acquire(
        // Only the noise of the sensor varies over the frame.
        &frame_of(|_| 100.0, Vector2::zeros()),
        &map_of(fields),
        2_200.0,
        &AcquisitionConfig::default(),
    )
    .unwrap();
    assert!(matches!(flat, Acquisition::Weak(_)), "{flat:?}");
    // The frame is 3 km away, past the 1 km radius searched.
    let elsewhere = acquire(
        &frame_of(fields, Vector2::new(3_000.0, 0.0)),
        &map_of(fields),
        1_000.0,
        &AcquisitionConfig::default(),
    )
    .unwrap();
    assert!(
        matches!(elsewhere, Acquisition::Weak(Some(_))),
        "{elsewhere:?}"
    );
}

/// The image of `ground` that a pinhole camera at `height_m` with the
/// eye-to-local `orientation` sees.
fn image_of(camera: &CameraModel, orientation: &UnitQuaternion<f64>, height_m: f64) -> GrayImage {
    image_at(camera, orientation, height_m, Vector2::zeros())
}

/// The image of `fields` from a camera above `eye_m` east and north.
fn image_at(
    camera: &CameraModel,
    orientation: &UnitQuaternion<f64>,
    height_m: f64,
    eye_m: Vector2<f64>,
) -> GrayImage {
    GrayImage::from_fn(camera.width, camera.height, |u, v| {
        let eye = Vector3::new(
            (f64::from(u) - camera.cx) / camera.fx,
            (camera.cy - f64::from(v)) / camera.fy,
            -1.0,
        );
        let ray = orientation * eye;
        let ground = eye_m + ray.xy() * (height_m / -ray.z);
        image::Luma([fields(ground).round() as u8])
    })
}

#[test]
fn a_frame_lies_on_the_ground_below_the_eye_whatever_the_heading() {
    let camera = CameraModel {
        width: 400,
        height: 300,
        fx: 300.0,
        fy: 300.0,
        cx: 199.5,
        cy: 149.5,
    };
    let ground = GroundRaster::from_samples(32.0, 20, |p| Some(fields(p))).unwrap();
    for (heading, tilt) in [(0.0, 0.0), (1.1, 0.0), (-2.4, 0.08)] {
        let orientation = UnitQuaternion::from_scaled_axis(Vector3::z() * heading)
            * UnitQuaternion::from_scaled_axis(Vector3::x() * tilt);
        let image = image_of(&camera, &orientation, 1_760.0);
        let frame =
            GroundRaster::of_frame(&image, &camera, &orientation, 1_760.0, (32.0, 20)).unwrap();
        let pairs: Vec<(f32, f32)> = frame
            .values
            .iter()
            .zip(&ground.values)
            .filter(|(f, _)| f.is_finite())
            .map(|(f, g)| (*f, *g))
            .collect();
        assert!(pairs.len() > 500, "{heading}: {}", pairs.len());
        let worst = pairs
            .iter()
            .map(|(f, g)| (f - g).abs())
            .fold(0.0_f32, f32::max);
        assert!(worst < 4.0, "{heading} {tilt}: {worst}");
    }
}

#[test]
fn a_search_refuses_what_it_cannot_compare() {
    let map = map_of(fields);
    let other = GroundRaster::from_samples(16.0, 30, |p| Some(fields(p))).unwrap();
    let config = AcquisitionConfig::default();
    assert!(acquire(&other, &map, 1_000.0, &config).is_err());
    assert!(acquire(&map, &map, f64::NAN, &config).is_err());
    let none = AcquisitionConfig { peaks: 0, ..config };
    assert!(acquire(&map, &map, 1_000.0, &none).is_err());
    assert!(GroundRaster::from_samples(0.0, 3, |_| Some(1.0)).is_err());
}

fn demo_camera() -> CameraModel {
    CameraModel {
        width: 400,
        height: 300,
        fx: 300.0,
        fy: 300.0,
        cx: 199.5,
        cy: 149.5,
    }
}

#[test]
fn a_search_in_slices_finds_what_one_call_finds() {
    let map = map_of(fields);
    let frame = frame_of(fields, Vector2::new(-830.0, 1_140.0));
    let config = AcquisitionConfig::default();
    let whole = acquire(&frame, &map, 2_000.0, &config).unwrap();
    let pyramid = Arc::new(MapPyramid::new(map));
    let mut search = AcquisitionSearch::new(
        vec![(0.0, frame)],
        pyramid,
        Vector2::zeros(),
        2_000.0,
        &config,
    )
    .unwrap();
    let mut slices = 0;
    let sliced = loop {
        slices += 1;
        if let Some(found) = search.advance(50_000) {
            break found;
        }
    };
    assert_eq!(sliced, whole);
    assert!(slices > 10, "{slices}");
    assert_eq!(search.pairs_done(), search.pairs_total());
}

#[test]
fn a_frame_under_a_wrong_heading_is_found_at_its_rotation() {
    let camera = demo_camera();
    let map = Arc::new(MapPyramid::new(map_of(fields)));
    let deg = std::f64::consts::PI / 180.0;
    for error in [-5.0, 2.0, 5.0] {
        let truth = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), error * deg);
        let eye = Vector2::new(640.0, -1_210.0);
        let image = image_at(&camera, &truth, 1_760.0, eye);
        let believed = UnitQuaternion::identity();
        let frames: Vec<(f64, GroundRaster)> = (-3..=3)
            .map(|k| {
                let rotation = f64::from(k) * 2.0 * deg;
                let laid = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), rotation) * believed;
                let raster =
                    GroundRaster::of_frame(&image, &camera, &laid, 1_760.0, (32.0, 48)).unwrap();
                (rotation, raster)
            })
            .collect();
        let config = AcquisitionConfig::default();
        let mut search =
            AcquisitionSearch::new(frames, Arc::clone(&map), Vector2::zeros(), 2_000.0, &config)
                .unwrap();
        let found = search.advance(u64::MAX).unwrap();
        let Acquisition::Found(peak) = found else {
            panic!("{error}: {found:?}");
        };
        assert!(
            (peak.rotation_rad / deg - error).abs() <= 1.01,
            "{error}: {peak:?}"
        );
        assert!((peak.offset_m - eye).norm() < 40.0, "{error}: {peak:?}");
    }
}

#[test]
fn a_search_about_another_centre_gives_offsets_from_the_map_centre() {
    let map = Arc::new(MapPyramid::new(map_of(fields)));
    let offset = Vector2::new(900.0, -100.0);
    let frames = vec![(0.0, frame_of(fields, offset))];
    let config = AcquisitionConfig::default();
    let centre = Vector2::new(500.0, -300.0);
    let mut search = AcquisitionSearch::new(frames, map, centre, 600.0, &config).unwrap();
    let Some(Acquisition::Found(peak)) = search.advance(u64::MAX) else {
        panic!("not found");
    };
    assert!((peak.offset_m - offset).norm() < 32.0, "{peak:?}");
}

#[test]
fn a_mosaic_of_parts_is_the_raster_of_the_whole() {
    let whole = GroundRaster::from_samples(32.0, 10, |p| Some(fields(p))).unwrap();
    let parts: Vec<((i64, i64), GroundRaster)> = [-7_i64, 0, 7]
        .iter()
        .flat_map(|&n| [-7_i64, 0, 7].map(|e| (e, n)))
        .map(|(e, n)| {
            let centre = Vector2::new(e as f64, n as f64) * 32.0;
            let part = GroundRaster::from_samples(32.0, 3, |p| Some(fields(p + centre))).unwrap();
            ((e, n), part)
        })
        .collect();
    let mosaic = GroundRaster::mosaic(32.0, 10, &parts).unwrap();
    for north in -10..=10 {
        for east in -10..=10 {
            let (a, b) = (
                mosaic.at(east, north).unwrap(),
                whole.at(east, north).unwrap(),
            );
            assert!((a - b).abs() < 1e-3, "{east} {north}: {a} {b}");
        }
    }
}

#[test]
fn a_place_in_the_corner_of_the_square_but_past_the_radius_is_not_searched() {
    // 1.27 km from the centre: inside the square of half side 1 km about
    // it, outside the disc of 1 km.
    let found = acquire(
        &frame_of(fields, Vector2::new(900.0, 900.0)),
        &map_of(fields),
        1_000.0,
        &AcquisitionConfig::default(),
    )
    .unwrap();
    assert!(matches!(found, Acquisition::Weak(_)), "{found:?}");
}
