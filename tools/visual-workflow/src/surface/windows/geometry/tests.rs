use super::*;
#[test]
fn crop_and_block_coordinates_recover_the_same_subpixel_points() {
    let size = (160, 120);
    for area in windows(size).unwrap() {
        let original = nalgebra::Vector2::new(f64::from(area[0]) + 42.25, 51.75);
        for rotation in 0..4 {
            let crop_size = if rotation % 2 == 0 {
                (area[2], area[3])
            } else {
                (area[3], area[2])
            };
            let local = original - nalgebra::Vector2::new(f64::from(area[0]), f64::from(area[1]));
            let encoded =
                crate::surface::matching::restore_point(local, (4 - rotation) % 4, crop_size);
            let mut pair = PixelMatch {
                reference: encoded,
                query: [13.0, 17.0].into(),
            };
            restore_crop(&mut pair, area, rotation);
            assert!((pair.reference - original).norm() < 1e-12);
            assert!(contains(&pair, area));
            assert_eq!(pair.query, nalgebra::Vector2::new(13.0, 17.0));
        }
    }
}
#[test]
fn candidate_budget_counts_windows_and_refuses_repeated_or_invalid_rotations() {
    assert!(validate_rotations(&[0, 1, 2, 3], 12).is_ok());
    for (rotations, limit) in [
        (vec![0, 1, 2, 3], 8),
        (vec![0, 0], 12),
        (vec![], 12),
        (vec![4], 12),
    ] {
        assert!(validate_rotations(&rotations, limit).is_err());
    }
    assert!(windows((32, 32)).is_err());
}

#[test]
fn subpixel_support_uses_original_coordinates_before_every_match() {
    let size = (160, 120);
    let area = [20, 0, 120, 120];
    for x in [23.5, 24.0, 24.5, 134.5, 135.0, 135.5] {
        let original = nalgebra::Vector2::new(x, 60.25);
        let expected = contains_pixel(original, area);
        for rotation in 0..4 {
            let rotated_size = if rotation % 2 == 0 {
                size
            } else {
                (size.1, size.0)
            };
            let encoded =
                crate::surface::matching::restore_point(original, (4 - rotation) % 4, rotated_size);
            let restored = crate::surface::matching::restore_point(encoded, rotation, size);
            assert_eq!(contains_pixel(restored, area), expected);
            assert!((restored - original).norm() < 1e-12);
        }
    }
}
