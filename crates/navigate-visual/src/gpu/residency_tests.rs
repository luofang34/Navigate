use super::*;

fn texture() -> GrayImage {
    GrayImage::from_fn(320, 240, |x, y| {
        let h = (x / 7).wrapping_mul(374761393) ^ (y / 9).wrapping_mul(668265263);
        image::Luma([40 + ((h ^ (h >> 13)).wrapping_mul(1274126177) % 160) as u8])
    })
}

fn matcher() -> GpuPyramidalMatcher {
    pollster::block_on(GpuPyramidalMatcher::new()).expect("hardware adapter")
}

fn prime(matcher: &mut GpuPyramidalMatcher, image: &GrayImage) {
    let points = pyramid::corners(image);
    matcher
        .track_points_blocking(image, image, &points)
        .expect("prime GPU query from real tracking");
    assert!(matcher.previous_gpu.is_some());
}

fn assert_cold_equal(matcher: &mut GpuPyramidalMatcher, image: &GrayImage) {
    let points = pyramid::corners(image);
    assert!(!points.is_empty());
    let result = matcher
        .track_points_blocking(image, image, &points)
        .expect("tracking with retained image");
    let cold = self::matcher()
        .track_points_blocking(image, image, &points)
        .expect("cold tracking");
    assert_eq!(result, cold);
    assert!(result.iter().flatten().count() > 40);
}

#[test]
#[ignore = "requires a hardware compute adapter; run explicitly on the target device"]
fn resident_query_matches_cold_tracking_and_clone() {
    let first = texture();
    let second = GrayImage::from_fn(320, 240, |x, y| {
        if x >= 2 && y >= 3 {
            *first.get_pixel(x - 2, y - 3)
        } else {
            image::Luma([0])
        }
    });
    let mut warm = matcher();
    assert!(!warm.reuse_gpu);
    warm.set_gpu_image_reuse(true);
    let points = pyramid::corners(&first);
    warm.track_points_blocking(&first, &second, &points)
        .unwrap();
    assert_eq!(warm.previous_gpu.as_ref().unwrap().size(), 403200);
    assert_cold_equal(&mut warm, &second.clone());
    warm.set_gpu_image_reuse(false);
    assert!(warm.previous_gpu.is_none());
    assert_cold_equal(&mut warm, &second);
    assert!(warm.previous_gpu.is_none());
}

#[test]
#[ignore = "requires a hardware compute adapter; run explicitly on the target device"]
fn resident_query_rejects_mutated_pixels_and_equal_byte_count_dimensions() {
    let image = texture();
    let mut changed = image.clone();
    for pixel in changed.pixels_mut() {
        pixel.0[0] = 255 - pixel.0[0];
    }
    let reshaped = GrayImage::from_raw(240, 320, image.as_raw().clone()).unwrap();
    assert_eq!(image.as_raw(), reshaped.as_raw());
    let mut warm = matcher();
    warm.set_gpu_image_reuse(true);
    for reference in [changed, reshaped] {
        prime(&mut warm, &image);
        assert_cold_equal(&mut warm, &reference);
    }
}

#[test]
#[ignore = "requires a hardware compute adapter; run explicitly on the target device"]
fn refinement_empty_and_invalid_calls_discard_resident_query() {
    let image = texture();
    let points = pyramid::corners(&image);
    let mut warm = matcher();
    warm.set_gpu_image_reuse(true);
    prime(&mut warm, &image);
    let refined = warm
        .refine_points_blocking(&image, &image, &points, &points)
        .unwrap();
    assert_eq!(
        refined,
        matcher()
            .refine_points_blocking(&image, &image, &points, &points)
            .unwrap()
    );
    assert!(warm.previous_gpu.is_none());
    assert_cold_equal(&mut warm, &image);
    assert!(
        warm.track_points_blocking(&image, &image, &[])
            .unwrap()
            .is_empty()
    );
    assert!(warm.previous_gpu.is_none());
    assert_cold_equal(&mut warm, &image);
    assert!(
        warm.track_points_blocking(&image, &GrayImage::new(64, 64), &points)
            .is_err()
    );
    assert!(warm.previous_gpu.is_none());
    assert_cold_equal(&mut warm, &image);
    assert!(
        warm.refine_points_blocking(&image, &image, &points, &[])
            .is_err()
    );
    assert!(warm.previous_gpu.is_none());
    assert_cold_equal(&mut warm, &image);
}
