use super::*;
use image::{Rgb, RgbImage};

#[test]
fn a_straight_down_view_sees_the_ground_below_the_camera() {
    let view = View {
        height_m: 100.0,
        pitch_deg: 90.0,
        heading_deg: 30.0,
        focal_px: 459.0,
    };
    let centre = view.footprint_centre((640, 360)).unwrap();
    assert!(centre[0].hypot(centre[1]) < 1.0, "{centre:?}");
    let photo = RgbImage::from_pixel(640, 360, Rgb([90, 120, 60]));
    let canvas = project::canvas(&photo, &view, centre, 160.0);
    let share = canvas.valid_share();
    // The frame covers about 139 × 78 m of the 160 m square.
    assert!((0.35..0.5).contains(&share), "{share}");
}

#[test]
fn a_forward_view_looks_along_its_heading() {
    let view = View {
        height_m: 800.0,
        pitch_deg: 25.0,
        heading_deg: 90.0,
        focal_px: 1200.0,
    };
    let [east, north] = view.footprint_centre((1600, 1200)).unwrap();
    assert!(east > 1_000.0 && north.abs() < 50.0, "{east} {north}");
}

#[test]
fn the_horizon_of_a_level_photo_is_at_its_centre_and_a_ground_photo_has_none() {
    let level = RgbImage::from_fn(800, 600, |_, y| {
        if y < 300 {
            Rgb([150, 180, 230])
        } else {
            Rgb([60, 80, 40])
        }
    });
    let h = horizon(&level, 700.0).unwrap();
    assert!(h.pitch_deg.abs() < 1.0 && h.roll_deg.abs() < 1.0, "{h:?}");
    let down = RgbImage::from_fn(800, 600, |_, y| {
        if y < 120 {
            Rgb([150, 180, 230])
        } else {
            Rgb([60, 80, 40])
        }
    });
    let looking_down = horizon(&down, 700.0).unwrap();
    assert!(
        (looking_down.pitch_deg - (180.0_f64 / 700.0).atan().to_degrees()).abs() < 1.0,
        "{looking_down:?}"
    );
    let ground = RgbImage::from_pixel(800, 600, Rgb([60, 80, 40]));
    assert!(horizon(&ground, 700.0).is_none());
}
