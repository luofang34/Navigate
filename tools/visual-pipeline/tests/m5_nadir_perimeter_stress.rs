//! Empirical stress tests for nadir image rotation and pixel unrotation.
//!
//! This module follows the ASD-STE100 Simplified Technical English standard.
//! It tests boundary perimeter pixels, clamping behavior, and subpixel accuracy.

use image::{GrayImage, Luma};
use navigate_visual_pipeline::corridor::nadir::{
    NADIR_YAW_BANDS_DEG, rotate_image_gray, unrotate_pixel,
};

#[test]
fn stress_test_perimeter_zero_degrees_all_dimensions() {
    let test_dimensions = [(20_u32, 20_u32), (35, 47), (80, 60), (3, 3), (2, 2)];

    for &(w, h) in &test_dimensions {
        let mut src = GrayImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let val = (((x * 13 + y * 17) % 200) + 50) as u8;
                src.put_pixel(x, y, Luma([val]));
            }
        }

        let rot0 = rotate_image_gray(&src, 0.0);

        // Verify top and bottom rows
        for x in 0..w {
            assert_eq!(
                rot0.get_pixel(x, 0),
                src.get_pixel(x, 0),
                "Top perimeter mismatch at ({x}, 0) for size {w}x{h}"
            );
            assert_eq!(
                rot0.get_pixel(x, h - 1),
                src.get_pixel(x, h - 1),
                "Bottom perimeter mismatch at ({x}, {}) for size {w}x{h}",
                h - 1
            );
        }

        // Verify left and right columns
        for y in 0..h {
            assert_eq!(
                rot0.get_pixel(0, y),
                src.get_pixel(0, y),
                "Left perimeter mismatch at (0, {y}) for size {w}x{h}"
            );
            assert_eq!(
                rot0.get_pixel(w - 1, y),
                src.get_pixel(w - 1, y),
                "Right perimeter mismatch at ({}, {y}) for size {w}x{h}",
                w - 1
            );
        }
    }
}

#[test]
fn stress_test_perimeter_no_gutter_at_cardinal_angles() {
    let cardinal_angles = [0.0, 90.0, 180.0, 270.0, 360.0];
    let (w, h) = (64_u32, 64_u32);
    let fill_value = 180_u8;
    let src = GrayImage::from_pixel(w, h, Luma([fill_value]));

    for &angle in &cardinal_angles {
        let rot = rotate_image_gray(&src, angle);

        // Check top and bottom rows
        for x in 0..w {
            assert_eq!(
                rot.get_pixel(x, 0).0[0],
                fill_value,
                "Gutter bug at top perimeter ({x}, 0) for angle {angle}"
            );
            assert_eq!(
                rot.get_pixel(x, h - 1).0[0],
                fill_value,
                "Gutter bug at bottom perimeter ({x}, {}) for angle {angle}",
                h - 1
            );
        }

        // Check left and right columns
        for y in 0..h {
            assert_eq!(
                rot.get_pixel(0, y).0[0],
                fill_value,
                "Gutter bug at left perimeter (0, {y}) for angle {angle}"
            );
            assert_eq!(
                rot.get_pixel(w - 1, y).0[0],
                fill_value,
                "Gutter bug at right perimeter ({}, {y}) for angle {angle}",
                w - 1
            );
        }
    }
}

#[test]
fn stress_test_perimeter_pixels_180_and_360_rectangular() {
    let (w, h) = (75_u32, 53_u32);
    let fill_val = 210_u8;
    let src = GrayImage::from_pixel(w, h, Luma([fill_val]));

    for &angle in &[180.0, 360.0] {
        let rot = rotate_image_gray(&src, angle);

        for x in 0..w {
            assert_eq!(
                rot.get_pixel(x, 0).0[0],
                fill_val,
                "Perimeter top zero gutter detected at angle {angle}"
            );
            assert_eq!(
                rot.get_pixel(x, h - 1).0[0],
                fill_val,
                "Perimeter bottom zero gutter detected at angle {angle}"
            );
        }
        for y in 0..h {
            assert_eq!(
                rot.get_pixel(0, y).0[0],
                fill_val,
                "Perimeter left zero gutter detected at angle {angle}"
            );
            assert_eq!(
                rot.get_pixel(w - 1, y).0[0],
                fill_val,
                "Perimeter right zero gutter detected at angle {angle}"
            );
        }
    }
}

#[test]
fn stress_test_unrotate_pixel_subpixel_roundtrip_dense_perimeter() {
    let (w, h) = (640_u32, 480_u32);
    let max_x = f64::from(w - 1);
    let max_y = f64::from(h - 1);

    let mut perimeter_points = Vec::new();

    // Top and bottom edges with subpixel steps
    let mut step_x = 0.0_f64;
    while step_x <= max_x {
        perimeter_points.push([step_x, 0.0]);
        perimeter_points.push([step_x, max_y]);
        step_x += 0.5;
    }

    // Left and right edges with subpixel steps
    let mut step_y = 0.0_f64;
    while step_y <= max_y {
        perimeter_points.push([0.0, step_y]);
        perimeter_points.push([max_x, step_y]);
        step_y += 0.5;
    }

    let angles = [
        0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 180.0, 240.0, 270.0, 300.0, 360.0,
    ];

    for &angle in &angles {
        for &pt in &perimeter_points {
            let forward = unrotate_pixel(pt, angle, w, h);
            assert!(
                forward[0].is_finite() && forward[1].is_finite(),
                "Forward coordinate must be finite for pt {pt:?} at angle {angle}"
            );

            let backward = unrotate_pixel(forward, -angle, w, h);
            let err_x = (backward[0] - pt[0]).abs();
            let err_y = (backward[1] - pt[1]).abs();

            assert!(
                err_x < 1e-8,
                "X roundtrip error too large ({err_x}) for pt {pt:?} at angle {angle}"
            );
            assert!(
                err_y < 1e-8,
                "Y roundtrip error too large ({err_y}) for pt {pt:?} at angle {angle}"
            );
        }
    }
}

#[test]
fn stress_test_unrotate_pixel_center_and_corner_invariance() {
    let test_sizes = [(100_u32, 100_u32), (800, 600), (1920, 1080), (1, 1), (3, 3)];

    for &(w, h) in &test_sizes {
        let cx = f64::from(w) * 0.5 - 0.5;
        let cy = f64::from(h) * 0.5 - 0.5;

        for &angle in &NADIR_YAW_BANDS_DEG {
            let unrot_c = unrotate_pixel([cx, cy], angle, w, h);
            let dcx = (unrot_c[0] - cx).abs();
            let dcy = (unrot_c[1] - cy).abs();
            assert!(
                dcx < 1e-10,
                "Center X must stay fixed for size {w}x{h} at angle {angle}"
            );
            assert!(
                dcy < 1e-10,
                "Center Y must stay fixed for size {w}x{h} at angle {angle}"
            );
        }

        // Test four exact corners
        let max_x = f64::from(w.saturating_sub(1));
        let max_y = f64::from(h.saturating_sub(1));
        let corners = [[0.0, 0.0], [max_x, 0.0], [0.0, max_y], [max_x, max_y]];

        for &c in &corners {
            for &angle in &[0.0, 60.0, 120.0, 180.0, 240.0, 300.0] {
                let forward = unrotate_pixel(c, angle, w, h);
                let backward = unrotate_pixel(forward, -angle, w, h);
                assert!((backward[0] - c[0]).abs() < 1e-8);
                assert!((backward[1] - c[1]).abs() < 1e-8);
            }
        }
    }
}

#[test]
fn stress_test_bilinear_subpixel_interpolation_smoothness() {
    let (w, h) = (32_u32, 32_u32);
    let mut src = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            src.put_pixel(x, y, Luma([(x * 8) as u8]));
        }
    }

    // Small angular steps from 0.0 to 10.0 degrees
    let mut prev_corner_val = f64::from(src.get_pixel(0, 0).0[0]);
    for step in 1..=10 {
        let angle = f64::from(step) * 0.5;
        let rot = rotate_image_gray(&src, angle);
        let corner_val = f64::from(rot.get_pixel(0, 0).0[0]);

        // Changes must be gradual and continuous
        let diff = (corner_val - prev_corner_val).abs();
        assert!(
            diff < 25.0,
            "Discontinuous jump {diff} in corner pixel at angle {angle}"
        );
        prev_corner_val = corner_val;
    }
}
