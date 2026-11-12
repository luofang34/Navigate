//! The bounded matcher searches what its bound needs, and never more than
//! the plain matcher; it matches the images with their local contrast
//! normalised.

use super::super::contrast::normalised;
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
    found(matcher, &reference, &query, shift)
}

/// The count of matches of `query`, `reference` moved by `shift`, that
/// found the shift.
fn found(
    matcher: &mut impl ImageMatcher,
    reference: &GrayImage,
    query: &GrayImage,
    shift: (u32, u32),
) -> usize {
    let offset = Vector2::new(f64::from(shift.0), f64::from(shift.1));
    matcher
        .match_images_blocking(reference, query)
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
        let (reference, query) = moved(shift);
        let (r, q) = (normalised(&reference), normalised(&query));
        let plain = found(&mut PyramidalMatcher, &r, &q, shift);
        let wide = correct(&mut BoundedPyramidalMatcher::new(500.0), shift);
        assert_eq!(wide, plain, "{shift:?}");
    }
}

/// `image` with another gain, response and contrast, and light that
/// changes across it, as a camera frame differs from a map render.
fn exposed(image: &GrayImage) -> GrayImage {
    GrayImage::from_fn(image.width(), image.height(), |x, y| {
        let v = f64::from(image.get_pixel(x, y)[0]) / 255.0;
        let light = 1.0 + 0.2 * (f64::from(x) / 60.0 + f64::from(y) / 90.0).cos();
        let out = 255.0 * (0.8 * light * (0.5 + 0.7 * (v.powf(1.3) - 0.5)));
        image::Luma([out.round().clamp(0.0, 255.0) as u8])
    })
}

#[test]
fn a_frame_of_another_exposure_and_light_keeps_its_matches() {
    let shift = (9, 4);
    let (reference, query) = moved(shift);
    let query = exposed(&query);
    let normalising = found(
        &mut BoundedPyramidalMatcher::new(30.0),
        &reference,
        &query,
        shift,
    );
    let plain = found(&mut PyramidalMatcher, &reference, &query, shift);
    assert!(normalising > 150, "{normalising}");
    assert!(plain < normalising / 2, "{plain} {normalising}");
}

#[test]
fn faint_ground_has_corners_relative_to_its_contrast() {
    let shift = (9, 4);
    let (reference, query) = moved(shift);
    let faint = |i: &GrayImage| {
        GrayImage::from_fn(i.width(), i.height(), |x, y| {
            image::Luma([120 + i.get_pixel(x, y)[0] / 64])
        })
    };
    let (reference, query) = (faint(&reference), faint(&query));
    let normalising = found(
        &mut BoundedPyramidalMatcher::new(30.0),
        &reference,
        &query,
        shift,
    );
    let plain = found(&mut PyramidalMatcher, &reference, &query, shift);
    assert!(normalising > 150, "{normalising}");
    assert_eq!(plain, 0);
}

#[test]
fn a_tight_bound_finds_a_shift_inside_it_and_not_past_it() {
    let mut tight = BoundedPyramidalMatcher::new(12.0);
    assert!(correct(&mut tight, (9, 4)) > 40);
    assert!(correct(&mut tight, (30, 4)) < 10);
    assert!(correct(&mut PyramidalMatcher, (30, 4)) > 40);
}
