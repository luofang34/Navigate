use super::*;
use image::Luma;

#[test]
fn query_features_survive_candidate_refinement_without_recomputing() -> Result<(), InferenceError> {
    let query = GrayImage::from_pixel(64, 64, Luma([42]));
    let mut cache = FeatureCache::default();
    let mut calls = 0_u64;
    let mut extract = |_: &GrayImage| {
        calls = calls.wrapping_add(1);
        Ok(Features::empty(256))
    };
    let mut previous = None;
    for color in 0..5 {
        let reference = GrayImage::from_pixel(64, 64, Luma([color]));
        cache.get_or_extract(&reference, &mut extract)?;
        let features = cache.get_or_extract(&query, &mut extract)?;
        if let Some(previous) = &previous {
            assert!(Arc::ptr_eq(previous, &features));
        }
        previous = Some(features);
        assert_eq!(cache.entries.len(), 2);
    }
    assert_eq!(calls, 6);
    Ok(())
}

#[test]
fn shape_and_pixels_identify_features_and_old_entries_are_evicted() -> Result<(), InferenceError> {
    let mut cache = FeatureCache::default();
    let mut calls = 0_u64;
    let mut extract = |_: &GrayImage| {
        calls = calls.wrapping_add(1);
        Ok(Features::empty(256))
    };
    let a = GrayImage::from_pixel(64, 128, Luma([4]));
    let b = GrayImage::from_pixel(128, 64, Luma([4]));
    let mut c = b.clone();
    c.put_pixel(5, 5, Luma([5]));
    for image in [&a, &b, &b.clone(), &c, &a] {
        cache.get_or_extract(image, &mut extract)?;
    }
    assert_eq!(calls, 4);
    assert_eq!(cache.entries.len(), 2);
    Ok(())
}

#[test]
fn failed_extraction_can_be_retried() -> Result<(), InferenceError> {
    let mut cache = FeatureCache::default();
    let image = GrayImage::new(64, 64);
    assert!(
        cache
            .get_or_extract(&image, &mut |_| Err(InferenceError::Invalid(
                "device failure".into()
            )))
            .is_err()
    );
    assert!(cache.entries.is_empty());
    cache.get_or_extract(&image, &mut |_| Ok(Features::empty(256)))?;
    assert_eq!(cache.entries.len(), 1);
    Ok(())
}
