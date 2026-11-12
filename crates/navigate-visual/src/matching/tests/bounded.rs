//! The bounded matcher searches what its bound needs, and never more than
//! the plain matcher.

use super::texture;
use crate::{BoundedPyramidalMatcher, ImageMatcher, PyramidalMatcher};
use image::GrayImage;
use nalgebra::Vector2;

/// The reference texture moved by `shift` pixels.
fn moved(shift: (u32, u32)) -> (GrayImage, GrayImage) {
    let reference = texture();
    let query = GrayImage::from_fn(320, 240, |x, y| {
        if x >= shift.0 && y >= shift.1 {
            image::Luma([reference.get_pixel(x - shift.0, y - shift.1)[0]])
        } else {
            image::Luma([0])
        }
    });
    (reference, query)
}

/// The count of matches that found the shift.
fn correct(matcher: &mut impl ImageMatcher, shift: (u32, u32)) -> usize {
    let (reference, query) = moved(shift);
    let offset = Vector2::new(f64::from(shift.0), f64::from(shift.1));
    matcher
        .match_images_blocking(&reference, &query)
        .expect("valid images")
        .iter()
        .filter(|m| (m.query - m.reference - offset).norm() < 0.5)
        .count()
}

#[test]
fn a_wide_bound_searches_as_the_plain_matcher() {
    // The plain matcher seeds 8 pixels at its coarsest level, a quarter of
    // the resolution: 32 pixels.
    for shift in [(26, 6), (44, 6)] {
        let plain = correct(&mut PyramidalMatcher, shift);
        let wide = correct(&mut BoundedPyramidalMatcher::new(500.0), shift);
        assert_eq!(wide, plain, "{shift:?}");
    }
}

#[test]
fn a_tight_bound_finds_a_shift_inside_it_and_not_past_it() {
    let mut tight = BoundedPyramidalMatcher::new(12.0);
    assert!(correct(&mut tight, (9, 4)) > 40);
    assert!(correct(&mut tight, (30, 4)) < 10);
    assert!(correct(&mut PyramidalMatcher, (30, 4)) > 40);
}
