//! Empirical challenge test suite for Milestone 3 Feature 12 (Decoupled Physical Plausibility):
//! 1. Surface clearance gating (AGL >= 5.0 m threshold, sub-surface rejections, extreme terrain).
//! 2. High-altitude operations (SR-71, mesosphere, LEO orbital) without upper ceiling.
//! 3. Cheirality verification (positive depth fraction >= 0.98).
//! 4. Fisher normal matrix conditioning (lambda_min >= 1e-6, singular/rank-deficient handling).
//! 5. SearchPrior covariance bounding (bypassed when untrusted, enforced when trusted, singular covariance safety).
//! 6. Non-finite coordinate rejection (NaN, Inf) without panic.
//! 7. Holdout reprojection RMSE boundary checks and fallback safety.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use navigate_visual::{CameraPose, SearchPrior};
use navigate_visual_pipeline::corridor::anchor::plausibility::{
    PhysicalPlausibilityConfig, PlausibilityError, compute_fisher_min_eigenvalue,
    compute_holdout_rmse_px, verify_physical_plausibility, verify_prior_bounding,
};
use navigate_visual_pipeline::pose::{PoseCandidate, PoseScore};
use std::error::Error;

type TestResult = Result<(), Box<dyn Error>>;

fn dummy_pose_score() -> PoseScore {
    PoseScore {
        focal_px: 1000.0,
        geometry_accepted: true,
        holdout_count: 8,
        holdout_inlier_fraction: 0.95,
        holdout_median_px: 1.2,
        image_support_fraction: 0.15,
        inliers: 16,
        pairs: 16,
        positive_depth_fraction: 0.99,
        train_inliers: 12,
        train_rms_px: 1.1,
    }
}

fn make_pinhole_k(fx: f64, fy: f64, cx: f64, cy: f64) -> Matrix3<f64> {
    Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)
}

#[test]
fn test_surface_clearance_boundaries_and_extremes() -> TestResult {
    let cfg = PhysicalPlausibilityConfig::default();

    // Exact 5.0 m AGL passes
    let res_exact = verify_physical_plausibility(505.0, 500.0, 0.99, 1e-4, 1.5, &cfg);
    assert!(res_exact.is_ok(), "AGL = 5.0 m must pass clearance check");

    // Epsilon above threshold passes
    let res_above = verify_physical_plausibility(505.0001, 500.0, 0.99, 1e-4, 1.5, &cfg);
    assert!(
        res_above.is_ok(),
        "AGL = 5.0001 m must pass clearance check"
    );

    // Epsilon below threshold fails with SubSurfaceClearance
    let res_below = verify_physical_plausibility(504.9999, 500.0, 0.99, 1e-4, 1.5, &cfg);
    assert!(
        matches!(res_below, Err(PlausibilityError::SubSurfaceClearance { agl_m }) if (agl_m - 4.9999).abs() < 1e-6),
        "AGL = 4.9999 m must fail with SubSurfaceClearance"
    );

    // Negative AGL (sub-surface / underground) fails
    let res_sub = verify_physical_plausibility(490.0, 500.0, 0.99, 1e-4, 1.5, &cfg);
    assert!(
        matches!(res_sub, Err(PlausibilityError::SubSurfaceClearance { agl_m }) if (agl_m - (-10.0)).abs() < 1e-6),
        "Underground eye must fail with SubSurfaceClearance"
    );

    // Extreme terrain: Dead Sea depression (-430 m MSL)
    let res_dead_sea_pass = verify_physical_plausibility(-425.0, -430.0, 0.99, 1e-4, 1.5, &cfg);
    assert!(
        res_dead_sea_pass.is_ok(),
        "AGL = 5.0 m at Dead Sea depression must pass"
    );

    let res_dead_sea_fail = verify_physical_plausibility(-425.01, -430.0, 0.99, 1e-4, 1.5, &cfg);
    assert!(
        matches!(
            res_dead_sea_fail,
            Err(PlausibilityError::SubSurfaceClearance { .. })
        ),
        "AGL = 4.99 m at Dead Sea depression must fail"
    );

    // Extreme terrain: Mt Everest summit (8848.86 m MSL)
    let res_everest_pass = verify_physical_plausibility(8853.86, 8848.86, 0.99, 1e-4, 1.5, &cfg);
    assert!(
        res_everest_pass.is_ok(),
        "AGL = 5.0 m at Everest summit must pass"
    );

    let res_everest_fail = verify_physical_plausibility(8853.85, 8848.86, 0.99, 1e-4, 1.5, &cfg);
    assert!(
        matches!(
            res_everest_fail,
            Err(PlausibilityError::SubSurfaceClearance { .. })
        ),
        "AGL = 4.99 m at Everest summit must fail"
    );

    Ok(())
}

#[test]
fn test_high_altitude_no_upper_ceiling() -> TestResult {
    let cfg = PhysicalPlausibilityConfig::default();

    // 11 km: Commercial airline cruise altitude
    assert!(verify_physical_plausibility(11_000.0, 100.0, 0.99, 1e-4, 1.0, &cfg).is_ok());

    // 25 km: Stratospheric / SR-71 / U-2 reconnaissance flight
    assert!(verify_physical_plausibility(25_000.0, 200.0, 0.99, 1e-4, 1.0, &cfg).is_ok());

    // 80 km: Mesosphere / high-altitude suborbital boundary
    assert!(verify_physical_plausibility(80_000.0, 0.0, 0.99, 1e-4, 1.0, &cfg).is_ok());

    // 100 km: Kármán space line
    assert!(verify_physical_plausibility(100_000.0, 0.0, 0.99, 1e-4, 1.0, &cfg).is_ok());

    // 200 km: Low Earth Orbit (LEO) satellite perigee
    assert!(verify_physical_plausibility(200_000.0, 0.0, 0.99, 1e-4, 1.0, &cfg).is_ok());

    // 408 km: International Space Station (ISS) orbital altitude
    assert!(verify_physical_plausibility(408_000.0, 0.0, 0.99, 1e-4, 1.0, &cfg).is_ok());

    // 35,786 km: Geostationary orbit (GEO)
    assert!(verify_physical_plausibility(35_786_000.0, 0.0, 0.99, 1e-4, 1.0, &cfg).is_ok());

    // 384,400 km: Lunar distance orbit
    assert!(verify_physical_plausibility(384_400_000.0, 0.0, 0.99, 1e-4, 1.0, &cfg).is_ok());

    Ok(())
}

#[test]
fn test_cheirality_depth_fraction_thresholds() -> TestResult {
    let cfg = PhysicalPlausibilityConfig::default();

    // 100% positive depth passes
    assert!(verify_physical_plausibility(500.0, 0.0, 1.0, 1e-4, 1.0, &cfg).is_ok());

    // Exact threshold: 0.98 passes
    assert!(verify_physical_plausibility(500.0, 0.0, 0.98, 1e-4, 1.0, &cfg).is_ok());

    // Epsilon above: 0.980001 passes
    assert!(verify_physical_plausibility(500.0, 0.0, 0.980001, 1e-4, 1.0, &cfg).is_ok());

    // Epsilon below: 0.979999 fails with LowCheiralityFraction
    let res_below = verify_physical_plausibility(500.0, 0.0, 0.979999, 1e-4, 1.0, &cfg);
    assert!(
        matches!(res_below, Err(PlausibilityError::LowCheiralityFraction { fraction }) if (fraction - 0.979999).abs() < 1e-6),
        "Positive depth fraction 0.979999 must fail"
    );

    // 95% passes standard PnP but fails decoupled plausibility
    let res_95 = verify_physical_plausibility(500.0, 0.0, 0.95, 1e-4, 1.0, &cfg);
    assert!(matches!(
        res_95,
        Err(PlausibilityError::LowCheiralityFraction { .. })
    ));

    // 0.0% (all correspondences behind camera) fails
    let res_zero = verify_physical_plausibility(500.0, 0.0, 0.0, 1e-4, 1.0, &cfg);
    assert!(matches!(
        res_zero,
        Err(PlausibilityError::LowCheiralityFraction { .. })
    ));

    // Negative fraction fails
    let res_neg = verify_physical_plausibility(500.0, 0.0, -0.5, 1e-4, 1.0, &cfg);
    assert!(matches!(
        res_neg,
        Err(PlausibilityError::LowCheiralityFraction { .. })
    ));

    Ok(())
}

#[test]
fn test_fisher_matrix_conditioning_thresholds() -> TestResult {
    let cfg = PhysicalPlausibilityConfig::default();

    // lambda_min = 1e-6 passes
    assert!(verify_physical_plausibility(500.0, 0.0, 0.99, 1e-6, 1.0, &cfg).is_ok());

    // lambda_min = 1.0001e-6 passes
    assert!(verify_physical_plausibility(500.0, 0.0, 0.99, 1.0001e-6, 1.0, &cfg).is_ok());

    // lambda_min = 9.999e-7 fails with IllConditionedFisher
    let res_below = verify_physical_plausibility(500.0, 0.0, 0.99, 9.999e-7, 1.0, &cfg);
    assert!(
        matches!(res_below, Err(PlausibilityError::IllConditionedFisher { min_eigenvalue }) if (min_eigenvalue - 9.999e-7).abs() < 1e-12),
        "lambda_min = 9.999e-7 must fail with IllConditionedFisher"
    );

    // Singular Fisher matrix (lambda_min = 0.0) fails
    let res_zero = verify_physical_plausibility(500.0, 0.0, 0.99, 0.0, 1.0, &cfg);
    assert!(matches!(
        res_zero,
        Err(PlausibilityError::IllConditionedFisher { .. })
    ));

    // Negative eigenvalue (indefinite normal matrix) fails
    let res_neg = verify_physical_plausibility(500.0, 0.0, 0.99, -1e-5, 1.0, &cfg);
    assert!(matches!(
        res_neg,
        Err(PlausibilityError::IllConditionedFisher { .. })
    ));

    Ok(())
}

#[test]
fn test_fisher_matrix_empirical_degeneration() -> TestResult {
    let k = make_pinhole_k(1000.0, 1000.0, 500.0, 500.0);
    let eye = [0.0, 0.0, 100.0];
    let rot = Matrix3::identity();

    // Scenario 1: Fewer than 6 inliers produces zero eigenvalue
    let mut cand_few = PoseCandidate {
        eye,
        rotation: rot,
        k,
        size: [1000, 1000],
        points: vec![[0.0, 0.0, 150.0]; 4],
        pixels: vec![[500.0, 500.0]; 4],
        inliers: vec![true; 4],
        score: dummy_pose_score(),
    };
    assert_eq!(
        compute_fisher_min_eigenvalue(&cand_few),
        0.0,
        "Fewer than 6 inliers must produce 0.0 Fisher eigenvalue"
    );

    // Scenario 2: Collinear points along optical axis produce rank-deficient Fisher matrix
    let collinear_points: Vec<[f64; 3]> = (0..10)
        .map(|i| [0.0, 0.0, 120.0 + f64::from(i) * 10.0])
        .collect();
    let collinear_pixels: Vec<[f64; 2]> = vec![[500.0, 500.0]; 10];
    let cand_collinear = PoseCandidate {
        eye,
        rotation: rot,
        k,
        size: [1000, 1000],
        points: collinear_points,
        pixels: collinear_pixels,
        inliers: vec![true; 10],
        score: dummy_pose_score(),
    };
    let lambda_collinear = compute_fisher_min_eigenvalue(&cand_collinear);
    assert!(
        lambda_collinear < 1e-6,
        "Collinear points along optical axis must be rank-deficient (lambda_min: {lambda_collinear:?})"
    );

    // Scenario 3: Full 3D distribution with parallax produces well-conditioned Fisher matrix
    let non_coplanar_points = vec![
        [15.0, 15.0, 130.0],
        [-15.0, 15.0, 135.0],
        [15.0, -15.0, 140.0],
        [-15.0, -15.0, 145.0],
        [30.0, 0.0, 125.0],
        [-30.0, 0.0, 150.0],
        [0.0, 30.0, 120.0],
        [0.0, -30.0, 155.0],
    ];
    let non_coplanar_pixels: Vec<[f64; 2]> = non_coplanar_points
        .iter()
        .map(|p| {
            let z = p[2] - eye[2];
            [1000.0 * p[0] / z + 500.0, 1000.0 * p[1] / z + 500.0]
        })
        .collect();
    let cand_well_conditioned = PoseCandidate {
        eye,
        rotation: rot,
        k,
        size: [1000, 1000],
        points: non_coplanar_points,
        pixels: non_coplanar_pixels,
        inliers: vec![true; 8],
        score: dummy_pose_score(),
    };
    let lambda_well = compute_fisher_min_eigenvalue(&cand_well_conditioned);
    assert!(
        lambda_well >= 1e-6,
        "Well-distributed 3D points must have lambda_min >= 1e-6 (got {lambda_well})"
    );

    // Scenario 4: Empty candidate with geometry_accepted = false returns 0.0
    cand_few.points.clear();
    cand_few.pixels.clear();
    cand_few.inliers.clear();
    cand_few.score.geometry_accepted = false;
    assert_eq!(compute_fisher_min_eigenvalue(&cand_few), 0.0);

    // Scenario 5: Empty candidate with geometry_accepted = true still returns 0.0 (no shortcut)
    cand_few.score.geometry_accepted = true;
    assert_eq!(
        compute_fisher_min_eigenvalue(&cand_few),
        0.0,
        "Empty candidate must return 0.0 even if geometry_accepted is true"
    );

    Ok(())
}

#[test]
fn test_search_prior_covariance_bounding_logic() -> TestResult {
    let center = CameraPose {
        position: Vector3::new(1000.0, 2000.0, 500.0),
        orientation: UnitQuaternion::identity(),
    };

    // Diagonal covariance: sigma_x = 10m, sigma_y = 10m, sigma_z = 2m
    // var_x = 100, var_y = 100, var_z = 4
    let cov = Matrix3::new(100.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 4.0);
    let prior = SearchPrior {
        center,
        position_covariance_m2: cov,
        attitude_sigma_rad: 0.05,
    };

    // When trust_prior == false, bounding is completely bypassed
    // Even an extreme candidate 500 km away is accepted
    let far_eye = [500_000.0, 2000.0, 500.0];
    assert!(
        verify_prior_bounding(far_eye, Some(&prior), false, 3.0),
        "Untrusted prior must bypass bounding for any candidate distance"
    );

    // When trust_prior == false and prior is None
    assert!(
        verify_prior_bounding(far_eye, None, false, 3.0),
        "Untrusted prior with None prior must return true"
    );

    // When trust_prior == true, Mahalanobis gate is strictly enforced
    // Candidate at center passes (Mahalanobis distance = 0)
    let center_eye = [1000.0, 2000.0, 500.0];
    assert!(verify_prior_bounding(center_eye, Some(&prior), true, 3.0));

    // Candidate at 2.9 sigma along X passes (dx = 29.0 m, var_x = 100.0 -> d^2 = 8.41 <= 9.0)
    let eye_2_9_sigma = [1029.0, 2000.0, 500.0];
    assert!(verify_prior_bounding(
        eye_2_9_sigma,
        Some(&prior),
        true,
        3.0
    ));

    // Candidate at 3.1 sigma along X fails (dx = 31.0 m, var_x = 100.0 -> d^2 = 9.61 > 9.0)
    let eye_3_1_sigma = [1031.0, 2000.0, 500.0];
    assert!(!verify_prior_bounding(
        eye_3_1_sigma,
        Some(&prior),
        true,
        3.0
    ));

    // Candidate at 3.1 sigma along Z fails (dz = 6.2 m, var_z = 4.0 -> d^2 = 38.44 / 4 = 9.61 > 9.0)
    let eye_3_1_z = [1000.0, 2000.0, 506.2];
    assert!(!verify_prior_bounding(eye_3_1_z, Some(&prior), true, 3.0));

    // Singular / uninvertible prior covariance falls back to Euclidean norm safely without panic
    let singular_cov = Matrix3::zeros();
    let singular_prior = SearchPrior {
        center,
        position_covariance_m2: singular_cov,
        attitude_sigma_rad: 0.05,
    };

    // Euclidean fallback threshold = sigma_scale * 100.0 = 300.0 m
    let near_singular = [1200.0, 2000.0, 500.0]; // diff = 200.0 m <= 300.0 m
    assert!(
        verify_prior_bounding(near_singular, Some(&singular_prior), true, 3.0),
        "Singular covariance must fallback to Euclidean distance safely"
    );

    let far_singular = [1400.0, 2000.0, 500.0]; // diff = 400.0 m > 300.0 m
    assert!(
        !verify_prior_bounding(far_singular, Some(&singular_prior), true, 3.0),
        "Singular covariance must reject candidates beyond fallback gate"
    );

    Ok(())
}

#[test]
fn test_non_finite_coordinates_rejection_without_panic() -> TestResult {
    let cfg = PhysicalPlausibilityConfig::default();
    for (eye, dem, depth, fisher, rmse) in [
        (f64::NAN, 500.0, 0.99, 1e-4, 1.0),
        (505.0, f64::NAN, 0.99, 1e-4, 1.0),
        (f64::INFINITY, 500.0, 0.99, 1e-4, 1.0),
        (f64::NEG_INFINITY, 500.0, 0.99, 1e-4, 1.0),
        (505.0, f64::INFINITY, 0.99, 1e-4, 1.0),
        (f64::NAN, f64::NAN, 0.99, 1e-4, 1.0),
        (505.0, 500.0, f64::NAN, 1e-4, 1.0),
        (505.0, 500.0, 0.99, f64::NAN, 1.0),
        (505.0, 500.0, 0.99, 1e-4, f64::NAN),
        (505.0, 500.0, f64::INFINITY, 1e-4, 1.0),
        (505.0, 500.0, 0.99, f64::INFINITY, 1.0),
        (505.0, 500.0, 0.99, 1e-4, f64::INFINITY),
    ] {
        assert_eq!(
            verify_physical_plausibility(eye, dem, depth, fisher, rmse, &cfg),
            Err(PlausibilityError::NonFinitePose)
        );
    }

    // Prior bounding with NaN candidate coordinates fails safely when trusted
    let center = CameraPose {
        position: Vector3::new(0.0, 0.0, 0.0),
        orientation: UnitQuaternion::identity(),
    };
    let prior = SearchPrior {
        center,
        position_covariance_m2: Matrix3::identity() * 100.0,
        attitude_sigma_rad: 0.05,
    };
    let nan_cand = [f64::NAN, 0.0, 0.0];
    assert!(
        !verify_prior_bounding(nan_cand, Some(&prior), true, 3.0),
        "NaN candidate coordinate must return false under trusted prior"
    );

    // Inf candidate coordinates fail safely when trusted
    let inf_cand = [0.0, f64::INFINITY, 0.0];
    assert!(
        !verify_prior_bounding(inf_cand, Some(&prior), true, 3.0),
        "Inf candidate coordinate must return false under trusted prior"
    );

    Ok(())
}

#[test]
fn test_holdout_reprojection_rmse_gating_and_fallback() -> TestResult {
    let cfg = PhysicalPlausibilityConfig::default();

    // Exactly 5.25 px passes
    assert!(verify_physical_plausibility(500.0, 0.0, 0.99, 1e-4, 5.25, &cfg).is_ok());

    // Epsilon below (5.2499 px) passes
    assert!(verify_physical_plausibility(500.0, 0.0, 0.99, 1e-4, 5.2499, &cfg).is_ok());

    // Epsilon above (5.2501 px) fails with ExcessiveReprojectionError
    let res_above = verify_physical_plausibility(500.0, 0.0, 0.99, 1e-4, 5.2501, &cfg);
    assert!(
        matches!(res_above, Err(PlausibilityError::ExcessiveReprojectionError { rmse_px }) if (rmse_px - 5.2501).abs() < 1e-6),
        "RMSE = 5.2501 px must fail with ExcessiveReprojectionError"
    );

    // Empirical compute_holdout_rmse_px fallback when no holdout points exist
    let mut cand = PoseCandidate {
        eye: [0.0, 0.0, 100.0],
        rotation: Matrix3::identity(),
        k: make_pinhole_k(1000.0, 1000.0, 500.0, 500.0),
        size: [1000, 1000],
        points: Vec::new(),
        pixels: Vec::new(),
        inliers: Vec::new(),
        score: dummy_pose_score(),
    };
    cand.score.train_rms_px = 2.75;

    // With empty correspondences, falls back to train_rms_px safely without panic
    let fallback_rmse = compute_holdout_rmse_px(&cand);
    assert!(
        (fallback_rmse - 2.75).abs() < 1e-6,
        "Fallback must return train_rms_px (got {fallback_rmse})"
    );

    Ok(())
}
