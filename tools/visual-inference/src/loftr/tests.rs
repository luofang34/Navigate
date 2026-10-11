use super::*;
#[test]
fn padding_and_backend_scores_do_not_become_geometry_evidence() {
    let image = GrayImage::new(960, 540);
    let input = preprocessing::prepare(&image, 640, 480, 1).expect("input");
    let points = [100.0, 10.0, 320.0, 240.0, 350.0, 250.0];
    let pairs = decode(
        &points,
        &points,
        &[0.9, 0.8, 0.1],
        [&input, &input],
        [&image, &image],
    )
    .expect("matches");
    assert_eq!(pairs.len(), 1);
    assert!((pairs[0].reference.x - 480.25).abs() < 1e-9);
    assert!((pairs[0].query.y - 270.25).abs() < 1e-9);
    assert!(decode(&[1.0], &[], &[], [&input, &input], [&image, &image]).is_err());
    assert!(
        decode(
            &[f32::NAN, 1.0],
            &[1.0, 1.0],
            &[1.0],
            [&input, &input],
            [&image, &image]
        )
        .is_err()
    );
}

#[test]
fn zero_confidence_row_cannot_become_a_correspondence() {
    let image = GrayImage::new(960, 540);
    let input = preprocessing::prepare(&image, 640, 480, 1).expect("input");
    let pairs = decode(
        &[320.0, 240.0],
        &[320.0, 240.0],
        &[0.0],
        [&input, &input],
        [&image, &image],
    )
    .expect("empty matches");
    assert!(pairs.is_empty());
}

#[test]
fn portrait_and_landscape_coordinates_use_separate_pixel_transforms() {
    let reference = GrayImage::new(480, 640);
    let query = GrayImage::new(640, 480);
    let a = preprocessing::prepare(&reference, 640, 480, 1).unwrap();
    let b = preprocessing::prepare(&query, 640, 480, 1).unwrap();
    let pairs = decode(
        &[319.5, 239.5, 100.0, 239.5],
        &[319.5, 239.5, 100.0, 239.5],
        &[0.9, 0.9],
        [&a, &b],
        [&reference, &query],
    )
    .unwrap();
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].reference, nalgebra::Vector2::new(239.5, 319.5));
    assert_eq!(pairs[0].query, nalgebra::Vector2::new(319.5, 239.5));
}

#[test]
#[ignore = "requires NAVIGATE_LOFTR_PAIR with runtime, model, reference and query paths"]
fn native_portrait_reference_matches_a_landscape_query() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::var_os("NAVIGATE_LOFTR_PAIR").ok_or("set NAVIGATE_LOFTR_PAIR")?;
    let paths: std::collections::BTreeMap<String, std::path::PathBuf> =
        serde_json::from_slice(&std::fs::read(path)?)?;
    let get = |key: &str| paths.get(key).ok_or_else(|| format!("missing {key}"));
    crate::initialize_blocking(get("runtime")?)?;
    let reference = image::open(get("reference")?)?.to_luma8();
    let query = image::open(get("query")?)?.to_luma8();
    assert_ne!(reference.dimensions(), query.dimensions());
    let mut matcher = LoFtrMatcher::load_blocking(get("model")?, &ExecutionConfig::default())?;
    let pairs = matcher.match_images_blocking(&reference, &query)?;
    assert!(pairs.len() >= 20);
    assert!(
        pairs
            .iter()
            .all(|p| preprocessing::inside(p.reference.into(), &reference)
                && preprocessing::inside(p.query.into(), &query))
    );
    let mut matcher = matcher.with_refinement_patches();
    let repeated = matcher.match_images_blocking(&reference, &query)?;
    let pixels = |pairs: &[PixelMatch]| {
        pairs
            .iter()
            .map(|p| [p.reference.x, p.reference.y, p.query.x, p.query.y])
            .collect::<Vec<_>>()
    };
    assert_eq!(pixels(&pairs), pixels(&repeated));
    Ok(())
}
