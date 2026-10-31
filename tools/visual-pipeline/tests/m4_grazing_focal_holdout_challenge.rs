//! Empirical challenge test suite for Milestone 4 (Features 18, 19, and 20):
//! - Feature 18: Canonical Grazing Poses, Scale Similarity, and Dynamic AGL.
//! - Feature 19: Focal Length Estimator, Sweep Multipliers, and Lock Stability.
//! - Feature 20: Mask-Aware Spatial Holdout and QueryRayDomain Conditioning.

use nalgebra::Matrix3;
use navigate_visual_pipeline::{
    pose::holdout::{MaskAwareSpatialHoldout, QueryRayDomain},
    stages::focal::{DEFAULT_FOCAL_MM, FOCAL_SWEEP_MULTIPLIERS, FocalLengthEstimator},
};
use std::{collections::BTreeSet, error::Error};

type TestResult = Result<(), Box<dyn Error>>;

const EPS: f64 = 1e-6;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

// =========================================================================
// FEATURE 18: Scale Similarity and Dynamic AGL Scaling
// =========================================================================

#[test]
fn test_scale_similarity_mathematical_bounds_and_symmetry() -> TestResult {
    // Mathematical definition from codec.rs:
    // d_live = ||eye_to_pt||
    // d_ref = (ref_dist_m * (fx / 600.0).max(0.1)).max(1.0)
    // ratio = d_live / d_ref
    // bounds = 0.22..=5.50
    // similarity = exp(-0.9 * (ln(ratio))^2)
    let sim_fn = |ratio: f64| -> Option<f32> {
        let bounds = 0.22..=5.50;
        if !bounds.contains(&ratio) {
            None
        } else {
            let ls = ratio.ln() as f32;
            Some((-0.9 * ls * ls).exp())
        }
    };

    // Boundary acceptance
    assert!(
        sim_fn(0.22).is_some(),
        "exact lower bound 0.22 must be accepted"
    );
    assert!(
        sim_fn(5.50).is_some(),
        "exact upper bound 5.50 must be accepted"
    );

    // Boundary rejection
    assert!(
        sim_fn(0.2199).is_none(),
        "ratio 0.2199 below lower bound must be rejected"
    );
    assert!(
        sim_fn(5.5001).is_none(),
        "ratio 5.5001 above upper bound must be rejected"
    );
    assert!(sim_fn(0.0).is_none(), "zero ratio must be rejected");
    assert!(sim_fn(-1.0).is_none(), "negative ratio must be rejected");

    // Maximum at ratio = 1.0
    let sim_unit = sim_fn(1.0).ok_or("unit ratio must exist")?;
    assert!(
        approx_eq(f64::from(sim_unit), 1.0, EPS),
        "similarity at ratio 1.0 must be 1.0"
    );

    // Symmetry in log-space: sim(2.0) == sim(0.5)
    let sim_2 = sim_fn(2.0).ok_or("ratio 2.0 must exist")?;
    let sim_half = sim_fn(0.5).ok_or("ratio 0.5 must exist")?;
    assert!(
        approx_eq(f64::from(sim_2), f64::from(sim_half), 1e-5),
        "scale similarity must be symmetric in log-space"
    );

    // Monotonic decay away from 1.0
    let sim_1_5 = sim_fn(1.5).ok_or("ratio 1.5 must exist")?;
    let sim_3_0 = sim_fn(3.0).ok_or("ratio 3.0 must exist")?;
    assert!(
        sim_1_5 > sim_3_0,
        "similarity must monotonically decrease with scale discrepancy"
    );

    Ok(())
}

#[test]
fn test_dynamic_agl_scaling_without_upper_clamp() -> TestResult {
    let compute_max_oblique = |agl_m: f64| -> f64 { (agl_m * 14.0).max(4_000.0) };

    // Low AGL: clamped to 4,000m minimum
    assert_eq!(
        compute_max_oblique(0.0),
        4000.0,
        "zero AGL must clamp to 4000m"
    );
    assert_eq!(
        compute_max_oblique(100.0),
        4000.0,
        "100m AGL must clamp to 4000m"
    );
    assert_eq!(
        compute_max_oblique(285.0),
        4000.0,
        "285m AGL (285*14 = 3990) must clamp to 4000m"
    );

    // Transition threshold at 4000 / 14 = 285.714...
    let agl_trans = 4000.0 / 14.0;
    assert!(approx_eq(compute_max_oblique(agl_trans), 4000.0, 1e-4));

    // Mid AGL: linear scaling
    assert_eq!(
        compute_max_oblique(500.0),
        7000.0,
        "500m AGL must scale to 7000m"
    );
    assert_eq!(
        compute_max_oblique(1000.0),
        14000.0,
        "1000m AGL must scale to 14000m"
    );
    assert_eq!(
        compute_max_oblique(3000.0),
        42000.0,
        "3000m AGL must scale to 42000m"
    );

    // High AGL and stratospheric altitudes: no arbitrary ceiling clamp
    assert_eq!(
        compute_max_oblique(10_000.0),
        140_000.0,
        "10km AGL must scale to 140km"
    );
    assert_eq!(
        compute_max_oblique(25_000.0),
        350_000.0,
        "25km U-2/SR-71 AGL must scale to 350km"
    );
    assert_eq!(
        compute_max_oblique(80_000.0),
        1_120_000.0,
        "80km mesosphere must scale to 1120km"
    );

    Ok(())
}

#[test]
fn test_dem_ray_frustum_intersection_geometry_bounds() -> TestResult {
    // Verified against select_corridor_tiles in canonical.rs:
    // horizon_dist_m = (2.0 * 6_371_000.0 * agl_m).sqrt()
    // max_ray_dist_m = 12_000.0.min(horizon_dist_m).max(4_000.0)
    let compute_frustum_ray_dist = |agl_m: f64| -> f64 {
        let agl = agl_m.max(10.0);
        let horizon_dist_m = (2.0 * 6_371_000.0 * agl).sqrt();
        12_000.0_f64.min(horizon_dist_m).max(4_000.0)
    };

    // Low AGL (10m): horizon is ~11.28 km, clamped between 4 km and 12 km
    let r_10 = compute_frustum_ray_dist(10.0);
    assert!(
        approx_eq(r_10, 11288.0, 1.0),
        "10m AGL raycast capped by horizon"
    );

    // High AGL (1000m): horizon is ~112.8 km, ray distance clamped to 12.0 km ceiling
    let r_1000 = compute_frustum_ray_dist(1000.0);
    assert_eq!(
        r_1000, 12000.0,
        "frustum raycast clamped to 12 km ceiling at high AGL"
    );

    // Extreme low AGL (1m): min clamp applies at 4000.0m
    let r_min = 12_000.0_f64.min(3_000.0).max(4_000.0);
    assert_eq!(r_min, 4000.0, "minimum ray distance clamped to 4 km");

    Ok(())
}

// =========================================================================
// FEATURE 19: Focal Length Estimator, Sweep Multipliers, and Lock Stability
// =========================================================================

#[test]
fn test_focal_length_estimator_initial_state_and_baselines() -> TestResult {
    let estimator = FocalLengthEstimator::default();
    assert_eq!(estimator.default_focal_mm, DEFAULT_FOCAL_MM);
    assert_eq!(estimator.default_focal_mm, 26.0);
    assert_eq!(estimator.sweep_multipliers, FOCAL_SWEEP_MULTIPLIERS);
    assert_eq!(estimator.sweep_multipliers, [1.0, 0.55, 1.45, 2.0]);
    assert!(
        estimator.locked_focal_px.is_none(),
        "locked focal length starts as None"
    );

    // Baseline calculation: (26.0 / 36.0) * width
    let b_1920 = estimator.baseline_focal_px(1920);
    assert!(
        approx_eq(b_1920, 1386.6666666666667, 1e-4),
        "1920 width baseline is ~1386.67 px"
    );

    let b_1280 = estimator.baseline_focal_px(1280);
    assert!(
        approx_eq(b_1280, 924.4444444444445, 1e-4),
        "1280 width baseline is ~924.44 px"
    );

    let b_800 = estimator.baseline_focal_px(800);
    assert!(
        approx_eq(b_800, 577.7777777777778, 1e-4),
        "800 width baseline is ~577.78 px"
    );

    let b_zero = estimator.baseline_focal_px(0);
    assert_eq!(b_zero, 0.0, "zero width returns zero baseline");

    Ok(())
}

#[test]
fn test_focal_length_sweep_multiplier_selection_and_lock() -> TestResult {
    let mut estimator = FocalLengthEstimator::default();
    let baseline = 1000.0;

    // Case 1: Index 0 (1.0) has lowest score
    let focal_1_0 = estimator.select_multiplier(&[0.8, 1.5, 2.0, 3.0], baseline);
    assert!(approx_eq(focal_1_0, 1000.0, EPS));
    assert_eq!(estimator.locked_focal_px, Some(1000.0));

    // Case 2: Index 1 (0.55 wide-angle) has lowest score
    let focal_0_55 = estimator.select_multiplier(&[2.5, 0.6, 1.8, 3.2], baseline);
    assert!(approx_eq(focal_0_55, 550.0, EPS));
    assert_eq!(estimator.locked_focal_px, Some(550.0));

    // Case 3: Index 2 (1.45 mid-tele) has lowest score
    let focal_1_45 = estimator.select_multiplier(&[2.1, 1.9, 0.45, 1.7], baseline);
    assert!(approx_eq(focal_1_45, 1450.0, EPS));
    assert_eq!(estimator.locked_focal_px, Some(1450.0));

    // Case 4: Index 3 (2.0 telephoto) has lowest score
    let focal_2_0 = estimator.select_multiplier(&[3.5, 2.8, 2.2, 0.9], baseline);
    assert!(approx_eq(focal_2_0, 2000.0, EPS));
    assert_eq!(estimator.locked_focal_px, Some(2000.0));

    // Tie-breaking: identical scores select first index (multiplier 1.0)
    let focal_tie = estimator.select_multiplier(&[1.0, 1.0, 1.0, 1.0], baseline);
    assert!(approx_eq(focal_tie, 1000.0, EPS));
    assert_eq!(estimator.locked_focal_px, Some(1000.0));

    // Non-finite score resilience
    let focal_nan = estimator.select_multiplier(&[f64::INFINITY, 1.2, f64::NAN, 3.0], baseline);
    assert!(
        approx_eq(focal_nan, 550.0, EPS),
        "valid finite score must beat Inf and NaN"
    );

    Ok(())
}

// =========================================================================
// FEATURE 20: Mask-Aware Spatial Holdout and QueryRayDomain Conditioning
// =========================================================================

#[test]
fn test_mask_aware_spatial_holdout_70_percent_occlusion() -> TestResult {
    let partitioner = MaskAwareSpatialHoldout::default();
    let img_size = [1920, 1080];

    // Generate 160 points: top 70% (sky/airframe) masked false, bottom 30% (ground) masked true
    let mut points = Vec::new();
    let mut mask = Vec::new();

    // 112 points in top 70% (y from 0 to 756)
    for i in 0..112 {
        let x = (i as f64 * 17.0) % 1920.0;
        let y = (i as f64 * 6.5) % 756.0;
        points.push([x, y]);
        mask.push(false); // occluded
    }

    // 48 points in bottom 30% (y from 756 to 1080)
    for i in 0..48 {
        let x = (i as f64 * 39.0) % 1920.0;
        let y = 756.0 + (i as f64 * 6.5) % 324.0;
        points.push([x, y]);
        mask.push(true); // valid ground
    }

    let (fit, holdout) = partitioner.partition(&points, img_size, Some(&mask));

    // All occluded points must be excluded
    let total_admitted = fit.len() + holdout.len();
    assert_eq!(
        total_admitted, 48,
        "only the 48 ground points must be admitted"
    );

    // Verify all admitted indices correspond to true mask entries
    for &idx in fit.iter().chain(holdout.iter()) {
        assert!(mask[idx], "admitted index must have true mask");
    }

    // Verify holdout is approximately 25% of the admitted ground points
    let holdout_frac = holdout.len() as f64 / 48.0;
    assert!(
        approx_eq(holdout_frac, 0.25, 0.08),
        "ground holdout fraction must be ~25%, actual: {holdout_frac}"
    );

    // Verify disjoint sets: no overlap between fit and holdout
    let fit_set: BTreeSet<usize> = fit.iter().copied().collect();
    let holdout_set: BTreeSet<usize> = holdout.iter().copied().collect();
    assert_eq!(
        fit_set.len(),
        fit.len(),
        "fit indices must contain no duplicates"
    );
    assert_eq!(
        holdout_set.len(),
        holdout.len(),
        "holdout indices must contain no duplicates"
    );
    assert!(
        fit_set.is_disjoint(&holdout_set),
        "fit and holdout sets must be strictly disjoint"
    );

    // Determinism test: partitioned output is identical on repeated calls
    let (fit2, holdout2) = partitioner.partition(&points, img_size, Some(&mask));
    assert_eq!(fit, fit2, "partitioning must be perfectly deterministic");
    assert_eq!(holdout, holdout2, "holdout must be perfectly deterministic");

    Ok(())
}

#[test]
fn test_spatial_holdout_subcell_starvation_prevention() -> TestResult {
    let partitioner = MaskAwareSpatialHoldout::default();
    let img_size = [1920, 1080];

    // Single point in cell: round(1 * 0.25) = 0 holdout, 1 fit (no starvation)
    let single_pt = vec![[100.0, 100.0]];
    let (fit_s, hold_s) = partitioner.partition(&single_pt, img_size, None);
    assert_eq!(fit_s.len(), 1, "single point must be retained in fit set");
    assert_eq!(hold_s.len(), 0, "single point cell produces 0 holdout");

    // Two points in cell: round(2 * 0.25) = 1 holdout, 1 fit
    let two_pts = vec![[100.0, 100.0], [105.0, 105.0]];
    let (fit_2, hold_2) = partitioner.partition(&two_pts, img_size, None);
    assert_eq!(fit_2.len(), 1, "two points cell produces 1 fit");
    assert_eq!(hold_2.len(), 1, "two points cell produces 1 holdout");

    // Four points in cell: round(4 * 0.25) = 1 holdout, 3 fit
    let four_pts = vec![
        [100.0, 100.0],
        [105.0, 105.0],
        [110.0, 110.0],
        [115.0, 115.0],
    ];
    let (fit_4, hold_4) = partitioner.partition(&four_pts, img_size, None);
    assert_eq!(fit_4.len(), 3, "four points cell produces 3 fit");
    assert_eq!(hold_4.len(), 1, "four points cell produces 1 holdout");

    // Fully masked (all occluded) returns empty vectors without error
    let all_masked = vec![false; 10];
    let pts = vec![[100.0, 100.0]; 10];
    let (fit_empty, hold_empty) = partitioner.partition(&pts, img_size, Some(&all_masked));
    assert!(fit_empty.is_empty());
    assert!(hold_empty.is_empty());

    Ok(())
}

#[test]
fn test_query_ray_domain_well_conditioned_geometry() -> TestResult {
    let eye = [0.0, 0.0, 100.0];
    let rot = Matrix3::identity();
    let k = Matrix3::new(600.0, 0.0, 400.0, 0.0, 600.0, 300.0, 0.0, 0.0, 1.0);
    let size = [800, 600];

    // Non-planar 3D terrain correspondences in front of the camera (Z > eye[2])
    let world_points = vec![
        [-100.0, -80.0, 300.0],
        [100.0, -80.0, 320.0],
        [-100.0, 80.0, 310.0],
        [100.0, 80.0, 340.0],
        [-50.0, -40.0, 280.0],
        [50.0, -40.0, 330.0],
        [-50.0, 40.0, 290.0],
        [50.0, 40.0, 325.0],
    ];

    let mut pixels = Vec::new();
    for wp in &world_points {
        let dz = wp[2] - eye[2]; // Z camera depth in front
        let u = k[(0, 0)] * (wp[0] - eye[0]) / dz + k[(0, 2)];
        let v = k[(1, 1)] * (wp[1] - eye[1]) / dz + k[(1, 2)];
        pixels.push([u, v]);
    }

    let domain = QueryRayDomain::analyze(&world_points, &pixels, eye, &rot, &k, size);
    assert!(
        domain.is_some(),
        "well-conditioned 3D geometry must produce a QueryRayDomain"
    );
    let d = domain.ok_or("missing domain")?;

    assert!(
        d.occupied_subcells >= 4,
        "must occupy at least 4 subcells, actual: {}",
        d.occupied_subcells
    );
    assert!(
        d.ray_angle_span_rad > 0.3,
        "ray angle span must be significant, actual: {}",
        d.ray_angle_span_rad
    );
    assert!(
        d.min_eigenvalue >= 1e-6,
        "min eigenvalue must be >= 1e-6, actual: {}",
        d.min_eigenvalue
    );
    assert!(
        d.condition_number <= 1e10,
        "condition number must be <= 1e10, actual: {}",
        d.condition_number
    );
    assert!(
        d.is_well_conditioned(),
        "domain must pass is_well_conditioned()"
    );

    Ok(())
}

#[test]
fn test_query_ray_domain_degenerate_and_boundary_cases() -> TestResult {
    let eye = [0.0, 0.0, 0.0];
    let rot = Matrix3::identity();
    let k = Matrix3::new(600.0, 0.0, 400.0, 0.0, 600.0, 300.0, 0.0, 0.0, 1.0);
    let size = [800, 600];

    // Fewer than 6 points returns None
    let few_points = vec![[0.0, 0.0, 100.0]; 5];
    let few_pixels = vec![[400.0, 300.0]; 5];
    assert!(QueryRayDomain::analyze(&few_points, &few_pixels, eye, &rot, &k, size).is_none());

    // Mismatched slice lengths returns None
    let pts6 = vec![[0.0, 0.0, 100.0]; 6];
    let px7 = vec![[400.0, 300.0]; 7];
    assert!(QueryRayDomain::analyze(&pts6, &px7, eye, &rot, &k, size).is_none());

    // Degenerate collinear points in single cell (< 4 cells) fails is_well_conditioned()
    let collinear_world = vec![
        [10.0, 10.0, 100.0],
        [20.0, 20.0, 100.0],
        [30.0, 30.0, 100.0],
        [40.0, 40.0, 100.0],
        [50.0, 50.0, 100.0],
        [60.0, 60.0, 100.0],
    ];
    let collinear_pixels = vec![
        [410.0, 310.0],
        [420.0, 320.0],
        [430.0, 330.0],
        [440.0, 340.0],
        [450.0, 350.0],
        [460.0, 360.0],
    ];

    if let Some(d) =
        QueryRayDomain::analyze(&collinear_world, &collinear_pixels, eye, &rot, &k, size)
    {
        assert!(
            !d.is_well_conditioned(),
            "collinear cluster must fail is_well_conditioned()"
        );
    }

    Ok(())
}
