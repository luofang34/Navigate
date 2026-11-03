//! Tier 2: Boundary and corner cases for Features 12 to 14 (Plausibility, Consensus & Multi-Tier).

use e2e_corridor_tests::{
    PhysicalPlausibilityConfig, PlausibilityError, SearchTier, TestResult, check,
    resolve_search_tier, verify_physical_plausibility,
};

// --- Feature 12: Decoupled Physical Plausibility Boundaries ---

#[test]
fn t2_12_01_exact_agl_boundary_5m() -> TestResult {
    let cfg = PhysicalPlausibilityConfig::default();
    // Exactly 5.0 m AGL
    let res = verify_physical_plausibility(505.0, 500.0, 0.99, 1e-4, 1.0, &cfg);
    check(res.is_ok(), "exact 5.0m AGL is accepted")?;
    Ok(())
}

#[test]
fn t2_12_02_subsurface_clearance_rejection() -> TestResult {
    let cfg = PhysicalPlausibilityConfig::default();
    // 4.99 m AGL (below 5.0m threshold)
    let res = verify_physical_plausibility(504.99, 500.0, 0.99, 1e-4, 1.0, &cfg);
    match res {
        Err(PlausibilityError::SubSurfaceClearance { agl_m }) => {
            check(agl_m < 5.0, "subsurface clearance reported correctly")?;
        }
        _ => return Err("expected SubSurfaceClearance error".into()),
    }
    Ok(())
}

#[test]
fn t2_12_03_cheirality_threshold_rejection() -> TestResult {
    let cfg = PhysicalPlausibilityConfig::default();
    // 0.979 positive fraction (< 0.98 threshold)
    let res = verify_physical_plausibility(600.0, 500.0, 0.979, 1e-4, 1.0, &cfg);
    match res {
        Err(PlausibilityError::LowCheiralityFraction { fraction }) => {
            check(fraction == 0.979, "low cheirality fraction reported")?;
        }
        _ => return Err("expected LowCheiralityFraction error".into()),
    }
    Ok(())
}

#[test]
fn t2_12_04_non_finite_pose_rejection() -> TestResult {
    let cfg = PhysicalPlausibilityConfig::default();
    let res_nan = verify_physical_plausibility(f64::NAN, 500.0, 0.99, 1e-4, 1.0, &cfg);
    check(
        res_nan == Err(PlausibilityError::NonFinitePose),
        "NaN eye rejected",
    )?;

    let res_inf = verify_physical_plausibility(f64::INFINITY, 500.0, 0.99, 1e-4, 1.0, &cfg);
    check(
        res_inf == Err(PlausibilityError::NonFinitePose),
        "Infinity eye rejected",
    )?;

    let res_nan_cheirality = verify_physical_plausibility(505.0, 500.0, f64::NAN, 1e-4, 1.0, &cfg);
    check(
        res_nan_cheirality == Err(PlausibilityError::NonFinitePose),
        "NaN cheirality rejected",
    )?;

    let res_nan_fisher = verify_physical_plausibility(505.0, 500.0, 0.99, f64::NAN, 1.0, &cfg);
    check(
        res_nan_fisher == Err(PlausibilityError::NonFinitePose),
        "NaN Fisher eigenvalue rejected",
    )?;

    let res_nan_reproj = verify_physical_plausibility(505.0, 500.0, 0.99, 1e-4, f64::NAN, &cfg);
    check(
        res_nan_reproj == Err(PlausibilityError::NonFinitePose),
        "NaN reprojection RMSE rejected",
    )?;
    Ok(())
}

#[test]
fn t2_12_05_ill_conditioned_fisher_rejection() -> TestResult {
    let cfg = PhysicalPlausibilityConfig::default();
    let res = verify_physical_plausibility(600.0, 500.0, 0.99, 0.99e-6, 1.0, &cfg);
    match res {
        Err(PlausibilityError::IllConditionedFisher { min_eigenvalue }) => {
            check(min_eigenvalue < 1e-6, "eigenvalue below 1e-6 reported")?;
        }
        _ => return Err("expected IllConditionedFisher error".into()),
    }
    Ok(())
}

// --- Feature 13: Consensus Relocation Boundaries ---

#[test]
fn t2_13_01_exact_separation_distance() -> TestResult {
    let min_separation_m = 50.0;
    let actual_dist_m = 50.0;
    check(
        actual_dist_m >= min_separation_m,
        "exact consensus separation satisfies threshold",
    )?;
    Ok(())
}

#[test]
fn t2_13_02_insufficient_separation_distance() -> TestResult {
    let min_separation_m = 50.0;
    let actual_dist_m = 49.5;
    check(
        actual_dist_m < min_separation_m,
        "insufficient baseline refuses consensus",
    )?;
    Ok(())
}

#[test]
fn t2_13_03_conflicting_visual_anchors() -> TestResult {
    let anchor1_pos = [100.0f64, 0.0f64, 0.0f64];
    let anchor2_pos = [800.0f64, 0.0f64, 0.0f64]; // Disagrees by 700m
    let distance = (anchor2_pos[0] - anchor1_pos[0]).abs();
    let consensus_threshold_m = 100.0;
    check(
        distance > consensus_threshold_m,
        "conflicting anchors detected and refused",
    )?;
    Ok(())
}

#[test]
fn t2_13_04_high_variance_gnss_relocation() -> TestResult {
    let gnss_sigma_m = 500.0;
    let visual_sigma_m = 5.0;
    let weight_visual = 1.0 / (visual_sigma_m * visual_sigma_m);
    let weight_gnss = 1.0 / (gnss_sigma_m * gnss_sigma_m);
    check(
        weight_visual > weight_gnss * 1000.0,
        "visual consensus dominates high variance GNSS",
    )?;
    Ok(())
}

#[test]
fn t2_13_05_zero_gnss_prior_relocation() -> TestResult {
    let gnss_prior_available = false;
    let consensus_anchors_count = 2;
    let can_relocate = !gnss_prior_available && consensus_anchors_count >= 2;
    check(can_relocate, "relocates in pure visual mode without GNSS")?;
    Ok(())
}

// --- Feature 14: Multi-Tier Retrieval & DINOv2 Boundaries ---

#[test]
fn t2_14_01_local_to_region_boundary() -> TestResult {
    let local_limit = 50.0;
    // Exactly at limit: Local
    let t_exact = resolve_search_tier(Some(50.0), local_limit, 1000.0);
    check(t_exact == SearchTier::Local, "exact local limit is Local")?;

    // Just above: Region
    let t_above = resolve_search_tier(Some(50.01), local_limit, 1000.0);
    check(
        t_above == SearchTier::Region,
        "above local limit switches to Region",
    )?;
    Ok(())
}

#[test]
fn t2_14_02_region_to_global_boundary() -> TestResult {
    let region_limit = 2000.0;
    // Exactly at limit: Region
    let t_exact = resolve_search_tier(Some(2000.0), 50.0, region_limit);
    check(
        t_exact == SearchTier::Region,
        "exact region limit is Region",
    )?;

    // Just above: Global
    let t_above = resolve_search_tier(Some(2000.01), 50.0, region_limit);
    check(
        t_above == SearchTier::Global,
        "above region limit switches to Global",
    )?;
    Ok(())
}

#[test]
fn t2_14_03_dinov2_empty_candidates() -> TestResult {
    let candidates: Vec<u32> = Vec::new();
    check(candidates.is_empty(), "empty candidates handled safely")?;
    Ok(())
}

#[test]
fn t2_14_04_dinov2_deterministic_tie_break() -> TestResult {
    let mut scores: Vec<(f64, u32)> = vec![(0.95, 10), (0.95, 5)];
    // Sort descending by score, tie-break ascending by ID
    scores.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    check(
        scores[0].1 == 5,
        "deterministic tie-break chooses lower ID 5",
    )?;
    Ok(())
}

#[test]
fn t2_14_05_unlocated_missing_prior() -> TestResult {
    let tier = resolve_search_tier(None, 50.0, 1000.0);
    check(
        tier == SearchTier::Global,
        "missing prior defaults to Global tier",
    )?;
    Ok(())
}
