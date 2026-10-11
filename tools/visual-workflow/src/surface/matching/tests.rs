use super::*;
#[test]
fn rotated_matches_return_to_original_non_square_raster_pixels() {
    let mut image = GrayImage::new(7, 5);
    image.put_pixel(2, 3, image::Luma([255]));
    for rotation in 0..4 {
        let rotated = rotate(&image, rotation);
        let (x, y, _) = rotated
            .enumerate_pixels()
            .find(|(_, _, p)| p.0[0] == 255)
            .unwrap();
        let mut pair = PixelMatch {
            reference: [f64::from(x), f64::from(y)].into(),
            query: [9.25, 10.5].into(),
        };
        restore(&mut pair, rotation, image.dimensions());
        assert_eq!(pair.reference, nalgebra::Vector2::new(2.0, 3.0));
        assert_eq!(pair.query, nalgebra::Vector2::new(9.25, 10.5));
    }
}
#[test]
fn orientation_work_is_bounded_and_unique() {
    assert!(validate_rotations(&[0, 1, 2, 3], 4).is_ok());
    for rotations in [&[][..], &[0, 0][..], &[4][..], &[0, 1, 2, 3, 0][..]] {
        assert!(validate_rotations(rotations, 4).is_err());
    }
    assert!(validate_rotations(&[0, 1], 1).is_err());
}
