use super::*;
use image::Luma;

#[test]
fn rotated_non_square_pixels_return_to_the_original_query_grid() -> Result<(), InferenceError> {
    let reference = GrayImage::from_pixel(6, 4, Luma([7]));
    let mut query = GrayImage::new(6, 4);
    query.put_pixel(2, 1, Luma([99]));
    let mut dimensions = Vec::new();
    for attempt in 0..3 {
        let pairs = match_blocking(&reference, &query, attempt, |a, b| {
            assert_eq!(a, &reference);
            dimensions.push(b.dimensions());
            let (x, y, _) = b
                .enumerate_pixels()
                .find(|(_, _, p)| p[0] == 99)
                .ok_or_else(|| InferenceError::Invalid("rotated marker is absent".into()))?;
            Ok(vec![PixelMatch {
                reference: [3.5, 1.25].into(),
                query: [f64::from(x), f64::from(y)].into(),
            }])
        })?
        .ok_or_else(|| InferenceError::Invalid("missing rotation".into()))?;
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].reference, nalgebra::Vector2::new(3.5, 1.25));
        assert_eq!(pairs[0].query, nalgebra::Vector2::new(2.0, 1.0));
    }
    assert_eq!(dimensions, [(4, 6), (6, 4), (4, 6)]);
    Ok(())
}

#[test]
fn subpixel_mapping_and_exhaustion_do_not_merge_alternatives() -> Result<(), InferenceError> {
    let image = GrayImage::new(32, 24);
    let mut calls = 0_u32;
    let expected = [[10.5, 12.75], [20.75, 12.5], [20.5, 10.25]];
    for attempt in 0..5 {
        let pairs = match_blocking(&image, &image, attempt, |_, _| {
            calls = calls.wrapping_add(1);
            Ok(vec![PixelMatch {
                reference: [10.0, 10.0].into(),
                query: [10.25, 10.5].into(),
            }])
        })?;
        if let Some(pairs) = pairs {
            assert_eq!(pairs.len(), 1);
            assert_eq!(
                pairs[0].query,
                nalgebra::Vector2::from(expected[attempt as usize])
            );
        } else {
            assert!(attempt >= 3);
        }
    }
    assert_eq!(calls, 3);
    Ok(())
}

#[test]
fn a_rotation_failure_preserves_its_error() {
    let image = GrayImage::new(32, 24);
    let result = match_blocking(&image, &image, 0, |_, _| {
        Err(InferenceError::Invalid("injected rotation failure".into()))
    });
    assert!(
        matches!(result, Err(InferenceError::Invalid(reason)) if reason == "injected rotation failure")
    );
}
