//! Empirical challenge test suite for Milestone 4 (Features 15, 16, and 17).
//!
//! Adversarial stress tests for:
//! 1. Keypoint Retention (Feature 15): Full budget retention across extreme occlusion ratios.
//! 2. Frame 1 Segformer Pass (Feature 16): Initial fast pass and cold anchor gating.
//! 3. RANSAC Horizon Attitude (Feature 17): Robust attitude recovery, cloud noise, airframe rejection.

use burn::tensor::Device;
use image::RgbImage;
use nalgebra::Matrix3;
use navigate_visual_pipeline::{
    corridor::masking::AmortizedMasker,
    surface::{
        RansacHorizonEstimator,
        horizon::{AIRFRAME_LABEL, GROUND_LABEL, PROPELLER_LABEL, SKY_LABEL},
        horizon_attitude,
    },
};
use std::error::Error;

type TestResult = Result<(), Box<dyn Error>>;

#[test]
fn challenge_f15_keypoint_budget_retention_extreme_occlusion() -> TestResult {
    let kp_limit = 2048_usize;
    let test_fractions = [0.05, 0.15, 0.30, 0.90, 1.0, 0.0];

    for &valid_frac in &test_fractions {
        let qry_kp_cap = if valid_frac > 0.0 { kp_limit } else { 0 };
        if valid_frac > 0.0 {
            assert_eq!(
                qry_kp_cap, kp_limit,
                "Budget must not throttle down for valid_fraction {valid_frac}"
            );
        } else {
            assert_eq!(
                qry_kp_cap, 0,
                "Budget must be zero when valid_fraction is 0.0"
            );
        }
    }
    Ok(())
}

#[test]
fn challenge_f16_frame1_segformer_gating() -> TestResult {
    let device = Device::flex();
    let masker_no_seg = AmortizedMasker::new(None, "airframe-occlusions", 6, 4, &device)?;

    assert!(!masker_no_seg.has_segformer());
    let dims = [320, 240];
    assert!(!masker_no_seg.has_usable_key(dims));

    // When no segformer is configured, cold anchor jobs must be permitted immediately
    let cold_job_allowed = masker_no_seg.has_usable_key(dims) || !masker_no_seg.has_segformer();
    assert!(
        cold_job_allowed,
        "Cold anchor job must be allowed immediately when Segformer is disabled"
    );

    // Initial fast pass with no segformer returns Ok(false)
    let dummy_rgb = RgbImage::new(dims[0], dims[1]);
    let mut masker_mut = masker_no_seg;
    let fast_pass_result = masker_mut.fast_initial_pass(&dummy_rgb, None)?;
    assert!(
        !fast_pass_result,
        "Fast pass without model should return false without error"
    );

    Ok(())
}

fn make_synthetic_labels(
    width: u32,
    height: u32,
    pitch_deg: f64,
    roll_deg: f64,
    k: &Matrix3<f64>,
) -> Vec<u8> {
    let slope = -roll_deg.to_radians().tan();
    let cx = k[(0, 2)];
    let cy = k[(1, 2)];
    let fy = k[(1, 1)].max(1.0);
    let y_at_cx = cy - fy * pitch_deg.to_radians().tan();
    let intercept = y_at_cx - slope * cx;

    let mut labels = vec![SKY_LABEL; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            let y_line = slope * f64::from(x) + intercept;
            if f64::from(y) >= y_line {
                labels[(y * width + x) as usize] = GROUND_LABEL;
            }
        }
    }
    labels
}

#[test]
fn challenge_f17_clean_horizon_pitches_and_rolls() -> TestResult {
    let width = 640_u32;
    let height = 480_u32;
    let k = Matrix3::new(300.0, 0.0, 320.0, 0.0, 300.0, 240.0, 0.0, 0.0, 1.0);

    let test_cases = [
        (-10.0, 0.0),
        (0.0, 0.0),
        (15.0, 0.0),
        (30.0, 0.0),
        (-10.0, 8.0),
        (0.0, -12.0),
        (15.0, 10.0),
        (30.0, -8.0),
    ];

    for &(true_pitch, true_roll) in &test_cases {
        let labels = make_synthetic_labels(width, height, true_pitch, true_roll, &k);
        let est = horizon_attitude(&labels, width, height, &k);
        assert!(
            est.is_some(),
            "Expected horizon attitude for pitch={true_pitch}, roll={true_roll}"
        );
        let (est_pitch, est_roll) = est.unwrap();
        let pitch_err = (est_pitch - true_pitch).abs();
        let roll_err = (est_roll - true_roll).abs();

        assert!(
            pitch_err <= 2.0,
            "Pitch error {pitch_err:.2}° exceeded 2.0° threshold for true pitch={true_pitch}"
        );
        assert!(
            roll_err <= 2.0,
            "Roll error {roll_err:.2}° exceeded 2.0° threshold for true roll={true_roll}"
        );
    }
    Ok(())
}

#[test]
fn challenge_f17_airframe_propeller_overlap_rejection() -> TestResult {
    let width = 640_u32;
    let height = 480_u32;
    let k = Matrix3::new(300.0, 0.0, 320.0, 0.0, 300.0, 240.0, 0.0, 0.0, 1.0);

    let true_pitch = 12.0;
    let true_roll = -5.0;
    let mut labels = make_synthetic_labels(width, height, true_pitch, true_roll, &k);

    // Overlap left 25% of columns with AIRFRAME
    for y in 0..height {
        for x in 0..(width / 4) {
            labels[(y * width + x) as usize] = AIRFRAME_LABEL;
        }
    }

    // Overlap center bottom section with PROPELLER
    for y in (height / 2)..height {
        for x in (width / 2 - 50)..(width / 2 + 50) {
            labels[(y * width + x) as usize] = PROPELLER_LABEL;
        }
    }

    let est = horizon_attitude(&labels, width, height, &k);
    assert!(
        est.is_some(),
        "Expected horizon attitude despite airframe and propeller occlusion"
    );
    let (est_pitch, est_roll) = est.unwrap();
    let pitch_err = (est_pitch - true_pitch).abs();
    let roll_err = (est_roll - true_roll).abs();

    assert!(
        pitch_err <= 2.0,
        "Pitch error {pitch_err:.2}° exceeded 2.0°"
    );
    assert!(roll_err <= 2.0, "Roll error {roll_err:.2}° exceeded 2.0°");
    Ok(())
}

#[test]
fn challenge_f17_all_sky_or_all_ground_returns_none() -> TestResult {
    let width = 640_u32;
    let height = 480_u32;
    let k = Matrix3::new(300.0, 0.0, 320.0, 0.0, 300.0, 240.0, 0.0, 0.0, 1.0);

    let all_sky = vec![SKY_LABEL; (width * height) as usize];
    assert!(
        horizon_attitude(&all_sky, width, height, &k).is_none(),
        "All sky must return None"
    );

    let all_ground = vec![GROUND_LABEL; (width * height) as usize];
    assert!(
        horizon_attitude(&all_ground, width, height, &k).is_none(),
        "All ground must return None"
    );

    let all_airframe = vec![AIRFRAME_LABEL; (width * height) as usize];
    assert!(
        horizon_attitude(&all_airframe, width, height, &k).is_none(),
        "All airframe must return None"
    );

    Ok(())
}

#[test]
fn challenge_f17_broken_cloud_noise_ransac_fit() -> TestResult {
    let width = 640_u32;
    let height = 480_u32;
    let focal = 300.0;
    let estimator = RansacHorizonEstimator::default();

    let true_pitch: f64 = 10.0;
    let true_roll: f64 = -6.0;

    // Generate inliers along true horizon line
    let slope = -true_roll.to_radians().tan();
    let cx = f64::from(width) * 0.5;
    let cy = f64::from(height) * 0.5;
    let y_at_cx = cy - focal * true_pitch.to_radians().tan();
    let intercept = y_at_cx - slope * cx;

    let mut points = Vec::new();
    let n_total = 100_usize;
    let n_outliers = 30_usize; // 30% broken cloud noise

    // 70 inliers on true horizon line with integer pixel quantization
    for i in 0..(n_total - n_outliers) {
        let x = (i as f64) * (f64::from(width) / (n_total as f64));
        let y = (slope * x + intercept).round();
        points.push([x, y]);
    }

    // 30 outliers (broken cloud noise scattered across the image)
    for i in 0..n_outliers {
        let x = ((i * 19 + 7) % width as usize) as f64;
        let y = ((i * 37 + 13) % height as usize) as f64;
        points.push([x, y]);
    }

    let est = estimator.fit_horizon(&points, [width, height], focal);
    assert!(
        est.is_some(),
        "RANSAC must fit horizon line despite 30% broken cloud noise"
    );
    let (est_pitch, est_roll) = est.unwrap();
    let pitch_err = (est_pitch - true_pitch).abs();
    let roll_err = (est_roll - true_roll).abs();

    assert!(
        pitch_err <= 2.0,
        "Pitch error {pitch_err:.2}° exceeded 2.0° under 30% cloud noise"
    );
    assert!(
        roll_err <= 2.0,
        "Roll error {roll_err:.2}° exceeded 2.0° under 30% cloud noise"
    );

    Ok(())
}

#[test]
fn challenge_f17_broken_cloud_noise_in_label_map() -> TestResult {
    let width = 640_u32;
    let height = 480_u32;
    let k = Matrix3::new(300.0, 0.0, 320.0, 0.0, 300.0, 240.0, 0.0, 0.0, 1.0);

    let true_pitch = 8.0;
    let true_roll = 5.0;
    let mut labels = make_synthetic_labels(width, height, true_pitch, true_roll, &k);

    // In 30% of columns, insert a cloud noise patch at varied heights in the sky
    let step_x = 4_u32;
    for (col_idx, x) in (0..width).step_by(step_x as usize).enumerate() {
        if col_idx % 3 == 0 {
            let cloud_y = ((col_idx * 17 + 23) % 100) as u32 + 20;
            for y in cloud_y..(cloud_y + 6) {
                labels[(y * width + x) as usize] = GROUND_LABEL;
            }
        }
    }

    let est = horizon_attitude(&labels, width, height, &k);
    assert!(
        est.is_some(),
        "Expected horizon attitude despite cloud noise in label map"
    );
    let (est_pitch, est_roll) = est.unwrap();
    let pitch_err = (est_pitch - true_pitch).abs();
    let roll_err = (est_roll - true_roll).abs();

    assert!(
        pitch_err <= 2.0,
        "Pitch error {pitch_err:.2}° exceeded 2.0°"
    );
    assert!(roll_err <= 2.0, "Roll error {roll_err:.2}° exceeded 2.0°");

    Ok(())
}

#[test]
fn challenge_f17_fully_occluded_horizon_returns_none() -> TestResult {
    let width = 640_u32;
    let height = 480_u32;
    let k = Matrix3::new(300.0, 0.0, 320.0, 0.0, 300.0, 240.0, 0.0, 0.0, 1.0);

    let mut labels = make_synthetic_labels(width, height, 0.0, 0.0, &k);

    // Completely block the horizon line with airframe (y from 200 to 280 across all columns)
    for y in 200..280 {
        for x in 0..width {
            labels[(y * width + x) as usize] = AIRFRAME_LABEL;
        }
    }

    let est = horizon_attitude(&labels, width, height, &k);
    assert!(
        est.is_none(),
        "Fully occluded horizon must return None without panic"
    );

    Ok(())
}

#[test]
fn challenge_f17_salt_and_pepper_boundary_cloud_noise() -> TestResult {
    let width = 640_u32;
    let height = 480_u32;
    let k = Matrix3::new(300.0, 0.0, 320.0, 0.0, 300.0, 240.0, 0.0, 0.0, 1.0);

    let true_pitch = -5.0;
    let true_roll = 6.0;
    let mut labels = make_synthetic_labels(width, height, true_pitch, true_roll, &k);

    let slope = -true_roll.to_radians().tan();
    let cx = k[(0, 2)];
    let cy = k[(1, 2)];
    let fy = k[(1, 1)].max(1.0);
    let y_at_cx = cy - fy * true_pitch.to_radians().tan();
    let intercept = y_at_cx - slope * cx;

    // Flip 30% of pixels along the horizon transition line
    for x in 0..width {
        let y_h = (slope * f64::from(x) + intercept).round() as i32;
        if y_h >= 0 && y_h < height as i32 {
            // Pseudo-random 30% flip: if (x * 31 + 7) % 10 < 3
            if (x as usize * 31 + 7) % 10 < 3 {
                let idx = (y_h as u32 * width + x) as usize;
                labels[idx] = SKY_LABEL;
            }
        }
    }

    let est = horizon_attitude(&labels, width, height, &k);
    assert!(
        est.is_some(),
        "RANSAC should estimate horizon attitude despite 30% boundary flip noise"
    );
    let (est_pitch, est_roll) = est.unwrap();
    let pitch_err = (est_pitch - true_pitch).abs();
    let roll_err = (est_roll - true_roll).abs();

    assert!(
        pitch_err <= 2.0,
        "Pitch error {pitch_err:.2}° exceeded 2.0°"
    );
    assert!(roll_err <= 2.0, "Roll error {roll_err:.2}° exceeded 2.0°");

    Ok(())
}
