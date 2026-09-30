use super::*;
use image::Rgb;

#[test]
fn rgb_planes_keep_raw_pixel_units() -> Result<(), InferenceError> {
    let image = RgbImage::from_pixel(3, 2, Rgb([7, 31, 211]));
    let values = prepare(&image, 4)?;
    assert_eq!(&values[..16], &[7.0; 16]);
    assert_eq!(&values[16..32], &[31.0; 16]);
    assert_eq!(&values[32..], &[211.0; 16]);
    Ok(())
}

#[test]
fn resize_uses_pixel_centres_and_clamps_edges() -> Result<(), InferenceError> {
    let image = RgbImage::from_fn(2, 1, |x, _| Rgb([(x * 100) as u8, 0, 0]));
    let values = prepare(&image, 4)?;
    for row in values[..16].as_chunks::<4>().0 {
        assert_eq!(*row, [0.0, 25.0, 75.0, 100.0]);
    }
    let average = prepare(&image, 1)?;
    assert_eq!(average, [50.0, 0.0, 0.0]);
    assert!(prepare(&RgbImage::new(0, 0), 384).is_err());
    Ok(())
}
