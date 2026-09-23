use super::*;
#[test]
fn letterbox_preserves_pixel_centres_and_padding() -> Result<(), InferenceError> {
    let image = GrayImage::from_pixel(640, 360, image::Luma([255]));
    let input = prepare(&image, 800, 600, 3)?;
    assert_eq!(input.data[0], 0.0);
    assert_eq!(input.data[300 * 800 + 400], 1.0);
    let p = [211.25, 82.75];
    let mapped = input.pixel([
        p[0] * input.scale + input.offset[0],
        p[1] * input.scale + input.offset[1],
    ]);
    assert!((mapped[0] - p[0]).abs() < 1e-9 && (mapped[1] - p[1]).abs() < 1e-9);
    Ok(())
}
#[test]
fn invalid_dimensions_do_not_enter_runtime() {
    assert!(prepare(&GrayImage::new(0, 0), 800, 600, 3).is_err());
}
