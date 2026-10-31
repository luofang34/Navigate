//! Independent forensic integrity stress tests for Milestone 5 features.
//!
//! This test module follows the ASD-STE100 Simplified Technical English standard.

use image::{GrayImage, Luma};
use nalgebra::{Matrix3, SMatrix, SVector};
use navigate_visual_pipeline::corridor::nadir::{
    is_nadir_pose, match_nadir_yaw_band, min_cos_orientation_for_pose, rotate_image_gray,
    synthesize_nadir_yaw_bands, unrotate_pixel,
};

#[test]
fn test_nadir_yaw_band_angles_are_exact() {
    let bands = synthesize_nadir_yaw_bands();
    assert_eq!(bands.len(), 6);
    assert_eq!(bands, [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]);
}

#[test]
fn test_nadir_pose_threshold() {
    // Identity rotation: down is +Z, so -rot[(2, 2)] = -1.0 <= 0.95 (not nadir)
    let eye = Matrix3::<f64>::identity();
    assert!(!is_nadir_pose(&eye));
    assert_eq!(min_cos_orientation_for_pose(&eye), 0.90);

    // Nadir pose: camera looks straight down along -Z, so rot[(2, 2)] = -1.0, -rot[(2, 2)] = 1.0 > 0.95
    let mut nadir_rot = Matrix3::<f64>::identity();
    nadir_rot[(2, 2)] = -1.0;
    assert!(is_nadir_pose(&nadir_rot));
    assert_eq!(min_cos_orientation_for_pose(&nadir_rot), 0.80);

    // Boundary check at exactly 0.95: -rot[(2, 2)] = 0.95 is false
    let mut b95 = Matrix3::<f64>::identity();
    b95[(2, 2)] = -0.95;
    assert!(!is_nadir_pose(&b95));

    // Boundary check slightly above 0.95: -rot[(2, 2)] = 0.9501 is true
    let mut b951 = Matrix3::<f64>::identity();
    b951[(2, 2)] = -0.9501;
    assert!(is_nadir_pose(&b951));
}

#[test]
fn test_match_nadir_yaw_band_full_coverage() {
    // Exact band headings match with cosine = 1.0
    for (i, &band) in [0.0, 60.0, 120.0, 180.0, 240.0, 300.0].iter().enumerate() {
        let (idx, cos_val) = match_nadir_yaw_band(band).expect("exact band must match");
        assert_eq!(idx, i);
        assert!((cos_val - 1.0).abs() < 1e-6);
    }

    // Wraparound 360 degrees matches band 0
    let (idx0, cos0) = match_nadir_yaw_band(360.0).expect("360 deg must match band 0");
    assert_eq!(idx0, 0);
    assert!((cos0 - 1.0).abs() < 1e-6);

    // Negative heading -60.0 degrees wraps to 300 degrees (index 5)
    let (idx5, cos5) = match_nadir_yaw_band(-60.0).expect("-60 deg must match band 5");
    assert_eq!(idx5, 5);
    assert!((cos5 - 1.0).abs() < 1e-6);

    // The 6 bands guarantee complete coverage: every angle has distance <= 30 deg to nearest band
    // cos(30 deg) = 0.8660 >= 0.80
    for deg in 0..360 {
        let (idx, cos_val) = match_nadir_yaw_band(deg as f64).expect("all angles covered");
        assert!(cos_val >= 0.8660 - 1e-4);
        let band_deg = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0][idx];
        let diff = (deg as f64 - band_deg).abs().rem_euclid(360.0);
        let min_diff = diff.min(360.0 - diff);
        assert!(min_diff <= 30.0 + 1e-6);
    }
}

#[test]
fn test_bilinear_rotation_identity_and_symmetry() {
    let (w, h) = (64_u32, 64_u32);
    let mut src = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            src.put_pixel(x, y, Luma([(x.wrapping_mul(7) ^ y.wrapping_mul(11)) as u8]));
        }
    }

    // Rotation by 0 degrees must preserve the inner image exactly
    let rot0 = rotate_image_gray(&src, 0.0);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            assert_eq!(src.get_pixel(x, y), rot0.get_pixel(x, y));
        }
    }

    // Rotation by 360 degrees must preserve the inner image exactly
    let rot360 = rotate_image_gray(&src, 360.0);
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            assert_eq!(src.get_pixel(x, y), rot360.get_pixel(x, y));
        }
    }
}

#[test]
fn test_impulse_rotation_and_unrotation_consistency() {
    let (w, h) = (200_u32, 200_u32);

    // Place an impulse peak at an off-center position
    let (px0, py0) = (130_u32, 120_u32);
    let mut src = GrayImage::new(w, h);
    src.put_pixel(px0, py0, Luma([255]));
    // Add 4-connected neighbors with lower values for a smooth peak
    src.put_pixel(px0 - 1, py0, Luma([128]));
    src.put_pixel(px0 + 1, py0, Luma([128]));
    src.put_pixel(px0, py0 - 1, Luma([128]));
    src.put_pixel(px0, py0 + 1, Luma([128]));

    for &angle_deg in &[60.0, 120.0, 180.0, 240.0, 300.0] {
        let rotated = rotate_image_gray(&src, angle_deg);

        // Find the maximum pixel in the rotated image
        let mut max_val = 0_u8;
        let mut max_pos = (0_u32, 0_u32);
        for y in 0..h {
            for x in 0..w {
                let v = rotated.get_pixel(x, y).0[0];
                if v > max_val {
                    max_val = v;
                    max_pos = (x, y);
                }
            }
        }

        assert!(max_val > 100, "rotated peak must remain visible");

        // Unrotate max_pos back to source coordinates
        let unrot = unrotate_pixel(
            [f64::from(max_pos.0), f64::from(max_pos.1)],
            angle_deg,
            w,
            h,
        );
        let dist =
            ((unrot[0] - f64::from(px0)).powi(2) + (unrot[1] - f64::from(py0)).powi(2)).sqrt();
        assert!(
            dist < 1.5,
            "unrotated peak at angle {angle_deg} must align with source peak, error was {dist} px"
        );
    }
}

#[test]
fn test_milliradian_scaling_algebraic_truth() {
    // Construct a synthetic 6x6 normal matrix H with translation in meters and rotation in milliradians
    let scale_diag = SVector::<f64, 6>::from_row_slice(&[1.0, 1.0, 1.0, 0.001, 0.001, 0.001]);
    let scale_mat = SMatrix::<f64, 6, 6>::from_diagonal(&scale_diag);

    // Well-behaved diagonally dominant matrix
    let mut h = SMatrix::<f64, 6, 6>::identity();
    for i in 0..6 {
        h[(i, i)] = (i + 1) as f64 * 100.0;
    }
    // Add small off-diagonal cross-coupling
    h[(0, 3)] = 1.0;
    h[(3, 0)] = 1.0;

    let h_inv = h.try_inverse().expect("invertible");
    let scaled_cov = scale_mat * h_inv * scale_mat;

    // Translation block: scaled by 1.0 * 1.0 = 1.0
    for r in 0..3 {
        for c in 0..3 {
            assert!((scaled_cov[(r, c)] - h_inv[(r, c)]).abs() < 1e-12);
        }
    }

    // Cross block: scaled by 1.0 * 0.001 = 0.001
    for r in 0..3 {
        for c in 3..6 {
            assert!((scaled_cov[(r, c)] - h_inv[(r, c)] * 0.001).abs() < 1e-12);
        }
    }

    // Rotation block: scaled by 0.001 * 0.001 = 1e-6
    for r in 3..6 {
        for c in 3..6 {
            assert!((scaled_cov[(r, c)] - h_inv[(r, c)] * 1e-6).abs() < 1e-12);
        }
    }
}
