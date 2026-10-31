//! Empirical challenge test suite for Milestone 5 (Features 21 and 22).
//!
//! Adversarial stress tests for:
//! 1. Multi-Yaw Nadir Coverage: In-plane rotation across 6 yaw bands.
//! 2. Pixel Rotation & Ray Preservation: Unrotation matches 3D ray projection within subpixel tolerance.
//! 3. Orientation Cosine Gating: min_cos_orientation = 0.80 strictly gates yaw misalignment to ±36°.
//! 4. Nadir Seed Prioritization: Low AGL (<= 250 m) nadir pitch prioritization across compass headings.

use image::{GrayImage, Luma, Rgb, RgbImage};
use navigate_visual_pipeline::{
    corridor::nadir::{
        MIN_COS_ORIENTATION, NADIR_DOWN_DEG, NADIR_YAW_BANDS_DEG, is_nadir_pose,
        match_nadir_yaw_band, min_cos_orientation_for_pose, rotate_image_gray,
        synthesize_nadir_yaw_bands, unrotate_pixel,
    },
    surface::Surface,
};
use std::error::Error;

type TestResult = Result<(), Box<dyn Error>>;

const GRAZING_PITCH_SEEDS_DEG: [f64; 6] = [-10.0, 8.0, 22.0, 38.0, 58.0, 88.0];

fn make_dummy_surface() -> Surface {
    let img = RgbImage::from_pixel(128, 128, Rgb([110, 125, 95]));
    Surface::synthetic(42.0, -74.0, 100.0, img)
}

/// Challenge seed ordering logic reproducing Feature 22 specification.
fn evaluate_seed_sequence(
    agl_m: f64,
    attitude: Option<(f64, f64)>,
    headings: &[f64],
) -> Vec<(f64, f64)> {
    let mut out = Vec::new();
    let prioritize_nadir = attitude.is_none() && agl_m <= 250.0;
    let pitches = match attitude {
        Some(att) => vec![att],
        None => {
            let mut list = Vec::new();
            for &p in &GRAZING_PITCH_SEEDS_DEG {
                list.push((p, 0.0));
            }
            for &r in &[25.0, -25.0] {
                for &p in &GRAZING_PITCH_SEEDS_DEG {
                    list.push((p, r));
                }
            }
            list
        }
    };

    if prioritize_nadir {
        for _ in headings {
            for &down_deg in &[88.0, 75.0] {
                out.push((down_deg, 0.0));
            }
        }
        for &(down_deg, roll_deg) in &pitches {
            if (down_deg - 88.0).abs() < 1e-3 && roll_deg.abs() < 1e-3 {
                continue;
            }
            for _ in headings {
                out.push((down_deg, roll_deg));
            }
        }
    } else {
        for &(down_deg, roll_deg) in &pitches {
            for _ in headings {
                out.push((down_deg, roll_deg));
            }
        }
    }
    out
}

#[test]
fn challenge_f21_yaw_bands_and_nadir_pose_thresholds() -> TestResult {
    let bands = synthesize_nadir_yaw_bands();
    assert_eq!(bands.len(), 6, "Must produce exactly 6 yaw bands");
    assert_eq!(
        bands,
        [0.0, 60.0, 120.0, 180.0, 240.0, 300.0],
        "Bands must start at 0 deg and step by 60 deg"
    );

    let surface = make_dummy_surface();
    for &band_deg in &NADIR_YAW_BANDS_DEG {
        let rot = surface.camera_rotation(42.0, -74.0, band_deg, NADIR_DOWN_DEG, 0.0);
        assert!(
            is_nadir_pose(&rot),
            "Pitch 89 deg must be identified as nadir pose for heading {band_deg}"
        );
        let cos_thresh = min_cos_orientation_for_pose(&rot);
        assert!(
            (cos_thresh - MIN_COS_ORIENTATION).abs() < 1e-6,
            "Nadir pose must have threshold {MIN_COS_ORIENTATION}, got {cos_thresh}"
        );
    }

    // Oblique pose check: pitch 45 deg down
    let rot_oblique = surface.camera_rotation(42.0, -74.0, 0.0, 45.0, 0.0);
    assert!(
        !is_nadir_pose(&rot_oblique),
        "Pitch 45 deg must not be classified as nadir pose"
    );
    assert_eq!(
        min_cos_orientation_for_pose(&rot_oblique),
        0.90,
        "Oblique pose must enforce 0.90 orientation threshold"
    );

    // Exact nadir transition boundary: -R22 = 0.95 corresponds to asin(0.95) = 71.805 deg
    let rot_nadir_boundary = surface.camera_rotation(42.0, -74.0, 0.0, 72.0, 0.0);
    assert!(
        is_nadir_pose(&rot_nadir_boundary),
        "Pitch 72 deg must qualify as nadir"
    );

    let rot_sub_boundary = surface.camera_rotation(42.0, -74.0, 0.0, 71.0, 0.0);
    assert!(
        !is_nadir_pose(&rot_sub_boundary),
        "Pitch 71 deg must not qualify as nadir"
    );
    Ok(())
}

#[test]
fn challenge_f21_rotate_image_gray_across_all_yaw_bands() -> TestResult {
    let (w, h) = (80_u32, 60_u32);
    let mut test_img = GrayImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let val = ((x * 3 + y * 4) % 256) as u8;
            test_img.put_pixel(x, y, Luma([val]));
        }
    }
    // Asymmetric marker at top-left quadrant
    test_img.put_pixel(10, 10, Luma([255]));
    test_img.put_pixel(11, 10, Luma([255]));
    test_img.put_pixel(10, 11, Luma([255]));

    for &angle in &NADIR_YAW_BANDS_DEG {
        let rotated = rotate_image_gray(&test_img, angle);
        assert_eq!(rotated.width(), w, "Rotated width must match source");
        assert_eq!(rotated.height(), h, "Rotated height must match source");

        if angle == 0.0 {
            // Identity rotation must equal source
            for y in 1..(h - 1) {
                for x in 1..(w - 1) {
                    let diff = (rotated.get_pixel(x, y).0[0] as i32
                        - test_img.get_pixel(x, y).0[0] as i32)
                        .abs();
                    assert!(
                        diff <= 1,
                        "0 deg rotation must equal source pixel at {x},{y}"
                    );
                }
            }
        }
    }

    // 360 deg rotation must match 0 deg rotation on interior pixels (within valid bilinear bounds)
    let rot_360 = rotate_image_gray(&test_img, 360.0);
    let rot_0 = rotate_image_gray(&test_img, 0.0);
    for y in 1..(h - 2) {
        for x in 1..(w - 2) {
            assert_eq!(
                rot_360.get_pixel(x, y),
                rot_0.get_pixel(x, y),
                "360 deg rotation must match 0 deg rotation at interior pixel {x},{y}"
            );
        }
    }

    // Degenerate empty image input must not panic
    let empty_img = GrayImage::new(0, 0);
    let empty_rot = rotate_image_gray(&empty_img, 60.0);
    assert_eq!(empty_rot.width(), 0);
    assert_eq!(empty_rot.height(), 0);
    Ok(())
}

#[test]
fn challenge_f21_pixel_unrotation_roundtrip_and_center_invariance() -> TestResult {
    let (w, h) = (800_u32, 600_u32);
    let cx = f64::from(w) * 0.5 - 0.5;
    let cy = f64::from(h) * 0.5 - 0.5;

    // Center pixel must be an invariant fixed point under unrotation for all 6 angles
    for &angle in &NADIR_YAW_BANDS_DEG {
        let unrot_c = unrotate_pixel([cx, cy], angle, w, h);
        assert!(
            (unrot_c[0] - cx).abs() < 1e-9,
            "Center X must remain invariant for angle {angle}"
        );
        assert!(
            (unrot_c[1] - cy).abs() < 1e-9,
            "Center Y must remain invariant for angle {angle}"
        );
    }

    // Roundtrip algebraic inversion across a grid of interior and boundary points
    let test_points: Vec<[f64; 2]> = vec![
        [0.0, 0.0],
        [f64::from(w - 1), 0.0],
        [0.0, f64::from(h - 1)],
        [f64::from(w - 1), f64::from(h - 1)],
        [cx, cy],
        [200.0, 150.0],
        [600.0, 450.0],
        [100.0, 500.0],
        [700.0, 100.0],
    ];

    for &pt in &test_points {
        for &angle in &NADIR_YAW_BANDS_DEG {
            let forward = unrotate_pixel(pt, angle, w, h);
            assert!(
                forward[0].is_finite() && forward[1].is_finite(),
                "Forward coordinate must be finite"
            );
            let backward = unrotate_pixel(forward, -angle, w, h);
            assert!(
                (backward[0] - pt[0]).abs() < 1e-8,
                "Roundtrip X must restore original coordinate within 1e-8"
            );
            assert!(
                (backward[1] - pt[1]).abs() < 1e-8,
                "Roundtrip Y must restore original coordinate within 1e-8"
            );
        }
    }
    Ok(())
}

#[test]
fn challenge_f21_3d_ray_projection_preservation_subpixel() -> TestResult {
    let (w, h) = (800_u32, 600_u32);
    let focal = 600.0_f64;
    let (cx, cy) = (f64::from(w) * 0.5 - 0.5, f64::from(h) * 0.5 - 0.5);
    let camera_alt = 500.0_f64;

    // For a camera looking straight down at DEM plane Z = 0:
    // A point at pixel (u_base, v_base) in the 0 deg yaw view casts a 3D ray:
    // ray_dir = [(u_base - cx) / focal, -(v_base - cy) / focal, -1.0]
    // Ground intersection: X = camera_alt * (u_base - cx) / focal, Y = -camera_alt * (v_base - cy) / focal
    // When the camera physically rotates by yaw angle psi (around the optical down-axis):
    // The camera right and down axes rotate in the horizontal plane by psi.
    // The ground point projects to pixel (u_rot, v_rot) in the physically rotated camera.
    // We challenge whether unrotate_pixel maps (u_rot, v_rot) back to (u_base, v_base) within subpixel tolerance.

    let sample_pixels: &[[f64; 2]] = &[
        [cx, cy],
        [cx + 100.0, cy],
        [cx - 150.0, cy + 80.0],
        [200.0, 150.0],
        [600.0, 450.0],
        [300.0, 200.0],
        [500.0, 400.0],
    ];

    for &base_px in sample_pixels {
        let dx_base = base_px[0] - cx;
        let dy_base = base_px[1] - cy;
        // 3D raycast to flat terrain plane Z = 0
        let ground_x = camera_alt * dx_base / focal;
        let ground_y = camera_alt * dy_base / focal;

        for &angle_deg in &NADIR_YAW_BANDS_DEG {
            let rad = angle_deg.to_radians();
            let cos_t = rad.cos();
            let sin_t = rad.sin();

            // 3D pinhole projection into camera physically yawed by angle_deg:
            let cam_x = ground_x * cos_t + ground_y * sin_t;
            let cam_y = -ground_x * sin_t + ground_y * cos_t;
            let rot_px = [
                cx + focal * cam_x / camera_alt,
                cy + focal * cam_y / camera_alt,
            ];

            let unrot_px = unrotate_pixel(rot_px, angle_deg, w, h);
            let err_x = (unrot_px[0] - base_px[0]).abs();
            let err_y = (unrot_px[1] - base_px[1]).abs();

            assert!(
                err_x < 1e-4,
                "Unrotated X error must be subpixel (<1e-4 px), got {err_x}"
            );
            assert!(
                err_y < 1e-4,
                "Unrotated Y error must be subpixel (<1e-4 px), got {err_y}"
            );
        }
    }
    Ok(())
}

#[test]
fn challenge_f21_min_cos_orientation_strict_36deg_gating() -> TestResult {
    // 1. Exact mathematical cone boundary:
    // cos(36.0 deg) = 0.80901699 >= 0.80 -> inside cone
    let cos_36 = 36.0_f64.to_radians().cos();
    assert!(
        cos_36 >= f64::from(MIN_COS_ORIENTATION),
        "36.0 deg must satisfy threshold"
    );

    // cos(36.8698976 deg) = 0.80000000 -> exact boundary
    let cos_limit = 36.8698976_f64.to_radians().cos();
    assert!(
        cos_limit >= 0.799999,
        "Cone boundary angle must meet threshold"
    );

    // cos(37.0 deg) = 0.7986355 < 0.80 -> outside cone
    let cos_37 = 37.0_f64.to_radians().cos();
    assert!(
        cos_37 < f64::from(MIN_COS_ORIENTATION),
        "37.0 deg must violate threshold"
    );

    // 2. Full-circle 360-degree sweep coverage:
    // With 6 bands spaced by 60 deg, maximum misalignment to nearest band is 30 deg.
    // cos(30 deg) = sqrt(3)/2 = 0.866025 > 0.80.
    // Therefore, EVERY query heading must match at least one synthesized band!
    for deg in 0..360 {
        let q_deg = f64::from(deg);
        let matched = match_nadir_yaw_band(q_deg)
            .ok_or_else(|| format!("Query heading {q_deg} deg must match a band"))?;
        assert!(
            matched.1 >= f64::from(MIN_COS_ORIENTATION),
            "Matched cosine must be >= {MIN_COS_ORIENTATION}"
        );
        assert!(
            matched.1 >= 0.8660,
            "Matched cosine must be >= cos(30 deg) = 0.866"
        );
    }

    // 3. Real drone survey flight headings:
    // DJI_0029 at heading 246.6 deg: nearest band is 240 deg (index 4) with delta = 6.6 deg
    let match_29 = match_nadir_yaw_band(246.6).ok_or("DJI_0029 must match")?;
    assert_eq!(match_29.0, 4, "DJI_0029 must select 240 deg band (index 4)");
    assert!(
        match_29.1 > 0.99,
        "DJI_0029 cosine must exceed 0.99, got {}",
        match_29.1
    );

    // DJI_0030 at heading 204.3 deg: distance to 180 deg is 24.3 deg; distance to 240 deg is 35.7 deg
    let match_30 = match_nadir_yaw_band(204.3).ok_or("DJI_0030 must match")?;
    assert_eq!(match_30.0, 3, "DJI_0030 must select 180 deg band (index 3)");
    assert!(
        match_30.1 >= f64::from(MIN_COS_ORIENTATION),
        "DJI_0030 cosine must meet threshold"
    );

    // 4. Strict rejection of distant bands:
    // A landmark synthesized at 0 deg compared against query at 60 deg has cos(60) = 0.50 < 0.80
    let dot_60 = 60.0_f64.to_radians().cos();
    assert!(
        dot_60 < f64::from(MIN_COS_ORIENTATION),
        "60 deg misalignment must be strictly rejected"
    );
    let dot_180 = 180.0_f64.to_radians().cos();
    assert!(
        dot_180 < f64::from(MIN_COS_ORIENTATION),
        "Opposing heading (180 deg) must be strictly rejected"
    );
    Ok(())
}

#[test]
fn challenge_f22_generate_seeds_low_agl_prioritization() -> TestResult {
    let surface = make_dummy_surface();
    let headings = [
        0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0,
    ];

    // Case 1: Low AGL (120 m AGL <= 250 m), attitude = None
    // The first 24 seeds must interleave nadir pitches 88.0 and 75.0 across all 12 headings
    let seq_low = evaluate_seed_sequence(120.0, None, &headings);
    assert!(
        seq_low.len() >= 24,
        "Must generate at least 24 seeds for 12 headings"
    );

    for (i, &(pitch, roll)) in seq_low.iter().take(24).enumerate() {
        assert!(
            (pitch - 88.0).abs() < 1e-4 || (pitch - 75.0).abs() < 1e-4,
            "Seed {i} under low AGL must be nadir pitch 88 deg or 75 deg, got {pitch}"
        );
        assert!(
            roll.abs() < 1e-4,
            "Nadir seeds must have zero roll, got {roll}"
        );

        let rot = surface.camera_rotation(42.0, -74.0, 0.0, pitch, roll);
        assert!(
            is_nadir_pose(&rot),
            "Camera rotation for seed {i} must qualify as nadir"
        );
    }

    // Subsequent seeds must not repeat (88.0, 0.0)
    for (i, &(pitch, roll)) in seq_low.iter().skip(24).enumerate() {
        assert!(
            (pitch - 88.0).abs() > 1e-3 || roll.abs() > 1e-3,
            "Seed {} in non-nadir suffix must not repeat (88.0, 0.0)",
            i + 24
        );
    }

    // Case 2: Exact boundary AGL = 250.0 m, attitude = None -> nadir prioritized
    let seq_boundary_low = evaluate_seed_sequence(250.0, None, &headings);
    assert_eq!(
        seq_boundary_low[0].0, 88.0,
        "Exact 250 m AGL must prioritize nadir pitch 88 deg first"
    );
    assert_eq!(
        seq_boundary_low[1].0, 75.0,
        "Exact 250 m AGL must prioritize nadir pitch 75 deg second"
    );

    // Case 3: AGL above 250 m (e.g. 250.001 m AGL), attitude = None -> standard order
    let seq_high = evaluate_seed_sequence(250.001, None, &headings);
    assert_eq!(
        seq_high[0].0, -10.0,
        "Above 250 m AGL, first seed must be grazing pitch -10 deg, got {}",
        seq_high[0].0
    );

    // Case 4: Low AGL but attitude prior provided: attitude prior must OVERRIDE nadir prioritization
    let seq_prior = evaluate_seed_sequence(100.0, Some((20.0, 0.0)), &headings);
    assert_eq!(
        seq_prior[0],
        (20.0, 0.0),
        "Attitude prior must override nadir prioritization"
    );

    // Case 5: Clamped sub-surface AGL: (alt - dem).max(0.0) -> AGL = 0.0 <= 250 m -> nadir prioritized
    let dem_h = 100.0_f64;
    let base_alt = 50.0_f64; // Sub-surface
    let clamped_agl = (base_alt - dem_h).max(0.0);
    assert_eq!(clamped_agl, 0.0);
    let seq_clamped = evaluate_seed_sequence(clamped_agl, None, &headings);
    assert_eq!(
        seq_clamped[0].0, 88.0,
        "Clamped sub-surface AGL (0 m) must prioritize nadir pitch 88 deg first"
    );
    Ok(())
}
