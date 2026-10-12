use super::*;

#[test]
fn equal_reference_reuses_every_pyramid_buffer() {
    let image = GrayImage::from_fn(320, 240, |x, y| image::Luma([(x ^ y) as u8]));
    let levels = pyramid::build(&image);
    let buffers: Vec<_> = levels.iter().map(|level| level.as_ptr()).collect();
    let mut cached = Some(levels);
    let reused =
        reference_pyramid_cached(&mut cached, &image.clone(), &mut profile::Profile::new());
    assert!(cached.is_none());
    assert_eq!(reused, pyramid::build(&image));
    assert_eq!(
        reused
            .iter()
            .map(|level| level.as_ptr())
            .collect::<Vec<_>>(),
        buffers
    );
}

#[test]
fn changed_pixels_or_dimensions_rebuild_the_reference_pyramid() {
    let image = GrayImage::new(320, 240);
    let mut changed = image.clone();
    changed.put_pixel(150, 100, image::Luma([1]));
    let reshaped = GrayImage::new(240, 320);
    assert_eq!(image.as_raw(), reshaped.as_raw());
    for reference in [changed, reshaped] {
        let mut cached = Some(pyramid::build(&image));
        let rebuilt =
            reference_pyramid_cached(&mut cached, &reference, &mut profile::Profile::new());
        assert_eq!(rebuilt, pyramid::build(&reference));
        assert_ne!(rebuilt, pyramid::build(&image));
        assert!(cached.is_none());
    }
}

#[test]
#[ignore = "requires a hardware compute adapter; run explicitly on the target device"]
fn cached_and_cold_tracking_are_identical() {
    let first = GrayImage::from_fn(320, 240, |x, y| {
        let h = (x / 7).wrapping_mul(374761393) ^ (y / 9).wrapping_mul(668265263);
        image::Luma([40 + ((h ^ (h >> 13)).wrapping_mul(1274126177) % 160) as u8])
    });
    let shift = |image: &GrayImage| {
        GrayImage::from_fn(320, 240, |x, y| {
            if x >= 2 && y >= 3 {
                *image.get_pixel(x - 2, y - 3)
            } else {
                image::Luma([0])
            }
        })
    };
    let second = shift(&first);
    let third = shift(&second);
    let mut warm = pollster::block_on(GpuPyramidalMatcher::new()).expect("hardware adapter");
    let first_points = pyramid::corners(&first);
    let points: Vec<_> = warm
        .track_points_blocking(&first, &second, &first_points)
        .expect("first tracking step")
        .into_iter()
        .flatten()
        .collect();
    assert!(points.len() > 40);
    let warm_output = warm
        .track_points_blocking(&second, &third, &points)
        .expect("cached tracking step");
    let mut cold = pollster::block_on(GpuPyramidalMatcher::new()).expect("hardware adapter");
    let cold_output = cold
        .track_points_blocking(&second, &third, &points)
        .expect("cold tracking step");
    assert_eq!(warm_output, cold_output);
    assert_eq!(warm.previous_pyramid, Some(pyramid::build(&third)));
}
