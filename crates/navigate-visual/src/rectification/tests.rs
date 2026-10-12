use super::*;
fn camera(size: u32) -> CameraModel {
    CameraModel {
        width: size,
        height: size,
        fx: 50.0,
        fy: 50.0,
        cx: 31.5,
        cy: 31.5,
    }
}
fn ramp() -> GrayImage {
    GrayImage::from_fn(64, 64, |x, y| image::Luma([(x + 2 * y) as u8]))
}
#[test]
fn identity_preserves_pixels_and_reuses_geometry_for_new_images() {
    let camera = camera(64);
    let map = RectificationMap::new(camera, camera, [0.0; 5], 64 * 64 * 8).unwrap();
    assert_eq!(map.camera(), camera);
    assert_eq!(map.map_bytes(), 64 * 64 * 8);
    assert_eq!(map.rectify(&ramp()).unwrap(), ramp());
    let other = GrayImage::from_pixel(64, 64, image::Luma([201]));
    assert_eq!(map.rectify(&other).unwrap(), other);
}
#[test]
fn explicit_crop_preserves_pixel_centres_and_pinhole_rays() {
    let source = camera(64);
    let mut target = camera(32);
    target.cx -= 16.0;
    target.cy -= 16.0;
    let map = RectificationMap::new(source, target, [0.0; 5], 8192).unwrap();
    assert_eq!(
        map.raw_pixel(Vector2::new(10.25, 9.75)).unwrap(),
        Vector2::new(26.25, 25.75)
    );
    assert_eq!(
        map.rectify(&ramp()).unwrap(),
        image::imageops::crop_imm(&ramp(), 16, 16, 32, 32).to_image()
    );
    let pose = crate::CameraPose {
        position: nalgebra::Vector3::zeros(),
        orientation: nalgebra::UnitQuaternion::identity(),
    };
    let pixel = Vector2::new(10.25, 9.75);
    assert!(
        (source.unproject(&pose, map.raw_pixel(pixel).unwrap(), 20.0)
            - target.unproject(&pose, pixel, 20.0))
        .norm()
            < 1e-12
    );
}
#[test]
fn radial_and_tangential_coefficients_follow_opencv_order() {
    let mut source = camera(64);
    source.cx = 30.0;
    source.cy = 30.0;
    let mut target = camera(32);
    target.cx = 15.5;
    target.cy = 15.5;
    let map =
        RectificationMap::new(source, target, [0.1, 0.01, 0.002, -0.001, 0.001], 8192).unwrap();
    let raw = map.raw_pixel(Vector2::new(25.5, 20.5)).unwrap();
    assert!((raw - Vector2::new(40.04775125, 35.030125625)).norm() < 1e-10);
    let image = map.rectify(&ramp()).unwrap();
    for y in 0..32 {
        for x in 0..32 {
            let p = map
                .raw_pixel(Vector2::new(f64::from(x), f64::from(y)))
                .unwrap();
            assert!((f64::from(image.get_pixel(x, y)[0]) - (p.x + 2.0 * p.y)).abs() <= 0.501);
        }
    }
}
#[test]
fn invalid_calibration_coverage_budget_and_source_size_are_rejected() {
    let c = camera(64);
    assert!(matches!(
        RectificationMap::new(c, c, [0.0; 5], 1),
        Err(RectificationError::Budget { .. })
    ));
    assert!(matches!(
        RectificationMap::new(c, c, [f64::NAN; 5], usize::MAX),
        Err(RectificationError::Coefficients)
    ));
    assert!(matches!(
        RectificationMap::new(c, c, [-1.0, 0.0, 0.0, 0.0, 0.0], usize::MAX),
        Err(RectificationError::Mapping { .. })
    ));
    let mut shifted = c;
    shifted.cx += 100.0;
    assert!(matches!(
        RectificationMap::new(c, shifted, [0.0; 5], usize::MAX),
        Err(RectificationError::Mapping { .. })
    ));
    let map = RectificationMap::new(c, c, [0.0; 5], 64 * 64 * 8).unwrap();
    assert!(matches!(
        map.rectify(&GrayImage::new(32, 32)),
        Err(RectificationError::SourceSize { .. })
    ));
    assert!(map.raw_pixel(Vector2::new(f64::NAN, 0.0)).is_none());
    assert!(map.raw_pixel(Vector2::new(64.0, 0.0)).is_none());
}

#[test]
fn rectified_residual_does_not_replace_a_raw_pixel_gate() {
    let source = camera(64);
    let mut target = camera(32);
    target.cx = 15.5;
    target.cy = 15.5;
    let map =
        RectificationMap::new(source, target, [0.1, 0.01, 0.002, -0.001, 0.001], 8192).unwrap();
    let observed = Vector2::new(26.0, 20.0);
    let predicted = Vector2::new(28.49, 20.0);
    assert!((observed - predicted).norm() < 2.5);
    assert!((map.raw_pixel(observed).unwrap() - map.raw_pixel(predicted).unwrap()).norm() > 2.5);
}

#[test]
fn published_hk_calibration_exposes_a_raw_gate_failure() {
    let source = CameraModel {
        width: 2448,
        height: 2048,
        fx: 1451.28,
        fy: 1451.29,
        cx: 1177.5,
        cy: 1043.5,
    };
    let target = CameraModel {
        width: 64,
        height: 32,
        cx: source.cx - 384.0,
        cy: source.cy - 1984.0,
        ..source
    };
    let map = RectificationMap::new(
        source,
        target,
        [-0.0572, 0.1209, 0.00124, -0.00018, -0.06327],
        64 * 32 * 8,
    )
    .unwrap();
    let observed = Vector2::new(407.80107602045075 - 384.0, 2005.0391949897473 - 1984.0);
    let predicted = Vector2::new(406.31492276033947 - 384.0, 2007.0370559677588 - 1984.0);
    assert!((observed - predicted).norm() < 2.5);
    let raw_error = (map.raw_pixel(observed).unwrap() - map.raw_pixel(predicted).unwrap()).norm();
    assert!((raw_error - 2.5648865661).abs() < 1e-9);
    assert!(raw_error > 2.5);
}
