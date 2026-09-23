//! Known pixel transformations check adapter output coordinates.
use super::Error;
use image::GrayImage;
use navigate_visual::ImageMatcher;
use serde_json::{Value, json};
pub(super) fn check_blocking(
    matcher: &mut impl ImageMatcher,
    image: &GrayImage,
) -> Result<Value, Error> {
    tracing::info!(check = "image identity", "matcher control");
    let identity = matcher.match_images_blocking(image, image)?;
    if identity.len() < 100
        || identity
            .iter()
            .any(|p| (p.reference - p.query).norm() > 1e-5)
    {
        return Err("adapter failed image identity".into());
    }
    let mut shifted = GrayImage::new(image.width(), image.height());
    for y in 16..image.height() {
        for x in 24..image.width() {
            shifted.put_pixel(x, y, *image.get_pixel(x - 24, y - 16));
        }
    }
    tracing::info!(check = "known translation", "matcher control");
    let pairs = matcher.match_images_blocking(image, &shifted)?;
    let mut errors: Vec<_> = pairs
        .iter()
        .map(|p| (p.query - p.reference - nalgebra::Vector2::new(24.0, 16.0)).norm())
        .collect();
    errors.sort_by(f64::total_cmp);
    let median = errors
        .get(errors.len() / 2)
        .copied()
        .unwrap_or(f64::INFINITY);
    if pairs.len() < 100 || median > 2.0 {
        return Err(format!(
            "adapter failed known translation: {} matches, median {median}",
            pairs.len()
        )
        .into());
    }
    let blank =
        matcher.match_images_blocking(image, &GrayImage::new(image.width(), image.height()))?;
    if !blank.is_empty() {
        return Err("blank image produced matches".into());
    }
    if matcher
        .match_images_blocking(image, &GrayImage::new(16, 16))
        .is_ok()
    {
        return Err("image dimension mismatch was not rejected".into());
    }
    Ok(
        json!({"identity_pairs":identity.len(),"translation_pairs":pairs.len(),"translation_median_error_px":median,"blank_pairs":0,"dimension_mismatch_rejected":true}),
    )
}
