use super::*;
use image::imageops::{FilterType, resize};

#[test]
fn gray_reduction_matches_triangle_pixels_for_odd_and_even_levels() {
    for (width, height) in [
        (64, 64),
        (65, 67),
        (127, 95),
        (319, 239),
        (640, 480),
        (1920, 1080),
    ] {
        for pattern in 0..4 {
            let mut image = GrayImage::from_fn(width, height, |x, y| {
                let value = match pattern {
                    0 => {
                        ((x.wrapping_mul(374761393) ^ y.wrapping_mul(668265263))
                            .wrapping_mul(1274126177)
                            >> 13) as u8
                    }
                    1 => ((x + y) % 2 * 255) as u8,
                    2 => ((x * 3 + y * 7) % 256) as u8,
                    _ => {
                        if x < width / 2 {
                            0
                        } else {
                            255
                        }
                    }
                };
                image::Luma([value])
            });
            for _ in 0..3 {
                let expected = resize(
                    &image,
                    image.width() / 2,
                    image.height() / 2,
                    FilterType::Triangle,
                );
                let actual = half(&image);
                assert_eq!(
                    actual,
                    expected,
                    "dimensions {:?}, pattern {pattern}",
                    image.dimensions()
                );
                assert_eq!(actual.color_space(), expected.color_space());
                image = actual;
            }
        }
    }
}

#[test]
fn gray_reduction_matches_triangle_for_all_constant_intensities() {
    for value in 0..=255 {
        let image = GrayImage::from_pixel(65, 67, image::Luma([value]));
        assert_eq!(half(&image), resize(&image, 32, 33, FilterType::Triangle));
    }
}
