use super::*;
use std::io::Cursor;

fn request(max_width_px: u32) -> LocateRequest {
    LocateRequest {
        version: navigate_contract::locate::LOCATE_MESSAGES_VERSION,
        target_latitude_rad: 0.9247,
        target_longitude_rad: 0.6293,
        heading_rad: 4.8,
        focal_px: 1420.0,
        max_width_px,
        area_radius_m: 800.0,
    }
}

fn jpeg(width: u32, height: u32) -> Vec<u8> {
    let image = image::RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, 9])
    });
    let mut out = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image)
        .write_to(&mut out, image::ImageFormat::Jpeg)
        .unwrap();
    out.into_inner()
}

#[test]
fn a_wide_photo_is_reduced_and_the_focal_length_follows() {
    let photo = prepare_photo(&request(1024), &jpeg(2048, 1172)).unwrap();
    assert_eq!(photo.image.dimensions(), (1024, 586));
    assert_eq!((photo.camera.width, photo.camera.height), (1024, 586));
    assert!((photo.camera.fx - 710.0).abs() < 1e-9);
    assert!((photo.camera.cx - 511.5).abs() < 1e-9);
}

#[test]
fn a_narrow_photo_keeps_its_size() {
    let photo = prepare_photo(&request(1024), &jpeg(640, 480)).unwrap();
    assert_eq!(photo.image.dimensions(), (640, 480));
    assert!((photo.camera.fx - 1420.0).abs() < 1e-9);
}

#[test]
fn another_version_and_bad_bytes_are_refused() {
    let mut other = request(1024);
    other.version = 99;
    assert!(
        prepare_photo(&other, &jpeg(64, 64))
            .unwrap_err()
            .contains("version")
    );
    assert!(
        prepare_photo(&request(1024), b"not a photo")
            .unwrap_err()
            .contains("cannot be read")
    );
}
