use super::*;

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
    assert!(matches!(found, Acquisition::Ambiguous { .. }), "{found:?}");
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
    GrayImage::from_fn(camera.width, camera.height, |u, v| {
        let eye = Vector3::new(
            (f64::from(u) - camera.cx) / camera.fx,
            (camera.cy - f64::from(v)) / camera.fy,
            -1.0,
        );
        let ray = orientation * eye;
        let ground = ray.xy() * (height_m / -ray.z);
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
