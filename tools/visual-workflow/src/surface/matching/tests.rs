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

struct PixelBackend;
impl FeatureBackend for PixelBackend {
    type Features = Vec<u8>;
    fn identity(&self) -> &str {
        "pixel-test"
    }
    fn extract_features_blocking(
        &mut self,
        image: &GrayImage,
    ) -> Result<Self::Features, navigate_visual::VisualError> {
        Ok(image.as_raw().iter().step_by(4).copied().collect())
    }
    fn validate_features(
        &self,
        features: &Self::Features,
        image: &GrayImage,
    ) -> Result<navigate_visual_workflow::FeatureMetadata, navigate_visual::VisualError> {
        assert_eq!(
            features,
            &image
                .as_raw()
                .iter()
                .step_by(4)
                .copied()
                .collect::<Vec<_>>()
        );
        Ok(navigate_visual_workflow::FeatureMetadata {
            resident_bytes: features.len(),
            count: features.len(),
        })
    }
    fn match_features_blocking(
        &mut self,
        _: &Self::Features,
        _: &Self::Features,
    ) -> Result<Vec<PixelMatch>, navigate_visual::VisualError> {
        Ok(Vec::new())
    }
}
#[test]
fn feature_preparation_releases_rasters_but_budgets_the_live_overlap() {
    let image = GrayImage::from_raw(4, 4, (0..16).collect()).unwrap();
    let mut backend = PixelBackend;
    let mut limits = Limits {
        image_pixels: 16,
        features_per_image: 4,
        candidates: 4,
        reference_bytes: 112,
        query_feature_bytes: 4,
    };
    let mut resident = 80;
    let mut handles = Vec::new();
    for rotation in 0..4 {
        let (features, retained) =
            extract_reference_blocking(&mut backend, rotate(&image, rotation), resident, &limits)
                .unwrap();
        resident = retained;
        handles.push(features);
    }
    assert_eq!(resident, 96);
    assert_eq!(handles[0], vec![0, 4, 8, 12]);
    assert_eq!(handles[1], vec![12, 13, 14, 15]);
    // Persistent features fit, but extracting one more overlaps a 16-byte raster.
    assert!(extract_reference_blocking(&mut backend, image.clone(), resident, &limits).is_err());
    limits.reference_bytes = 116;
    assert!(extract_reference_blocking(&mut backend, image.clone(), resident, &limits).is_ok());
    limits.features_per_image = 3;
    assert!(extract_reference_blocking(&mut backend, image, resident, &limits).is_err());
}
