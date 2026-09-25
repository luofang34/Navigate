#![allow(clippy::expect_used)]

use super::*;

fn texture() -> GrayImage {
    GrayImage::from_fn(320, 240, |x, y| {
        let h = (x / 7).wrapping_mul(374761393) ^ (y / 9).wrapping_mul(668265263);
        image::Luma([40 + ((h ^ (h >> 13)).wrapping_mul(1274126177) % 160) as u8])
    })
}

#[test]
fn translation_and_exposure_offset_preserve_matches() {
    let reference = texture();
    let query = GrayImage::from_fn(320, 240, |x, y| {
        if x >= 26 && y >= 11 {
            image::Luma([reference.get_pixel(x - 26, y - 11)[0] + 15])
        } else {
            image::Luma([0])
        }
    });
    let matches = PyramidalMatcher
        .match_images_blocking(&reference, &query)
        .expect("valid images");
    let correct = matches
        .iter()
        .filter(|m| (m.query - m.reference - Vector2::new(26.0, 11.0)).norm() < 0.5)
        .count();
    assert!(
        correct > 40,
        "{correct} correct of {} matches: {:?}",
        matches.len(),
        &matches[..matches.len().min(5)]
    );
    assert!(correct * 10 >= matches.len() * 9);
}

#[test]
fn blank_images_do_not_produce_matches() {
    let blank = GrayImage::new(320, 240);
    assert!(
        PyramidalMatcher
            .match_images_blocking(&blank, &blank)
            .expect("valid images")
            .is_empty()
    );
    assert!(
        PyramidalMatcher
            .match_images_blocking(&texture(), &blank)
            .expect("valid images")
            .is_empty()
    );
}

#[test]
fn border_points_use_the_finest_available_pyramid_support() {
    let reference = texture();
    let query = GrayImage::from_fn(320, 240, |x, y| {
        if x >= 2 && y >= 3 {
            *reference.get_pixel(x - 2, y - 3)
        } else {
            image::Luma([0])
        }
    });
    let points: Vec<_> = [20.0, 58.0, 134.0, 210.0, 286.0]
        .into_iter()
        .flat_map(|x| [20.0, 216.0].map(|y| Vector2::new(x, y)))
        .chain([Vector2::new(2.0, 2.0)])
        .collect();
    let tracked = PyramidalMatcher
        .track_points_blocking(&reference, &query, &points)
        .expect("border point tracks");
    let correct = points
        .iter()
        .zip(&tracked)
        .filter(|(p, q)| q.is_some_and(|q| (q - *p - Vector2::new(2.0, 3.0)).norm() < 0.5))
        .count();
    assert!(correct >= 6, "only {correct} border points retained");
    assert_eq!(correct, tracked.iter().filter(|p| p.is_some()).count());
    assert_eq!(tracked.last(), Some(&None));
}
