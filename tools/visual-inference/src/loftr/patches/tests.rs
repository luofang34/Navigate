use super::*;
use image::Luma;

fn pair() -> PixelMatch {
    PixelMatch {
        reference: [10.0, 20.0].into(),
        query: [13.0, 18.0].into(),
    }
}

#[test]
fn crops_keep_displacement_and_sample_the_correct_pixels() -> Result<(), InferenceError> {
    let reference = GrayImage::from_fn(512, 288, |x, y| Luma([((x + 3 * y) % 251) as u8]));
    let query = GrayImage::from_fn(512, 288, |x, y| Luma([reference.get_pixel(x, y)[0] + 1]));
    let mut dimensions = Vec::new();
    let pairs = match_blocking(&reference, &query, |a, b| {
        dimensions.push(a.dimensions());
        assert_eq!(a.dimensions(), b.dimensions());
        for (left, right) in a.pixels().zip(b.pixels()) {
            assert_eq!(left[0] + 1, right[0]);
        }
        Ok(vec![pair()])
    })?;
    assert_eq!(
        dimensions,
        [(512, 288), (320, 180), (320, 180), (320, 180), (320, 180)]
    );
    assert_eq!(pairs.len(), 5);
    for matched in &pairs {
        assert_eq!(
            matched.query - matched.reference,
            nalgebra::Vector2::new(3.0, -2.0)
        );
    }
    assert_eq!(pairs[4].reference, nalgebra::Vector2::new(202.0, 128.0));
    Ok(())
}

#[test]
fn the_pair_limit_keeps_all_patch_regions() -> Result<(), InferenceError> {
    let image = GrayImage::new(512, 288);
    let pairs = match_blocking(&image, &image, |_, _| Ok(vec![pair(); 2000]))?;
    assert_eq!(pairs.len(), 4096);
    assert!(
        pairs
            .iter()
            .any(|p| p.reference == nalgebra::Vector2::new(202.0, 128.0))
    );
    Ok(())
}

#[test]
fn small_images_use_only_one_inference_call() -> Result<(), InferenceError> {
    let image = GrayImage::new(128, 128);
    let mut calls = 0u32;
    let pairs = match_blocking(&image, &image, |_, _| {
        calls = calls.wrapping_add(1);
        Ok(vec![pair()])
    })?;
    assert_eq!(calls, 1);
    assert_eq!(pairs.len(), 1);
    Ok(())
}

#[test]
fn inference_failure_stops_the_patch_batch() {
    let image = GrayImage::new(512, 288);
    let mut calls = 0u32;
    let result = match_blocking(&image, &image, |_, _| {
        calls = calls.wrapping_add(1);
        if calls == 3 {
            Err(InferenceError::Invalid("test patch failure".into()))
        } else {
            Ok(vec![pair()])
        }
    });
    assert_eq!(calls, 3);
    assert!(
        matches!(result, Err(InferenceError::Invalid(reason)) if reason == "test patch failure")
    );
}
