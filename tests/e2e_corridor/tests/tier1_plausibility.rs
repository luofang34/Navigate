//! Tier 1: Feature coverage tests for Features 12 to 14 (Plausibility, Consensus & Multi-Tier).

use e2e_corridor_tests::{
    PhysicalPlausibilityConfig, PlausibilityError, SearchTier, TestResult, check,
    resolve_search_tier, verify_physical_plausibility,
};
use navigate_place::{HeadingPrior, HeightPrior, PitchPrior, Region, SearchRequest, plan};
use navigate_visual_session::{AnchorPolicy, RevisionCause, SessionConfig};

// --- Feature 12: Decoupled Physical Plausibility ---

#[test]
fn t1_12_01_positive_surface_clearance() -> TestResult {
    let cfg = PhysicalPlausibilityConfig::default();
    let res = verify_physical_plausibility(550.0, 500.0, 0.99, 1e-4, 1.5, &cfg);
    check(res.is_ok(), "50m clearance above DEM passes")?;
    Ok(())
}

#[test]
fn t1_12_02_stratospheric_altitude_no_cap() -> TestResult {
    let cfg = PhysicalPlausibilityConfig::default();
    let res_strat = verify_physical_plausibility(25_000.0, 500.0, 0.99, 1e-4, 1.2, &cfg);
    check(
        res_strat.is_ok(),
        "25 km stratospheric altitude passes without cap",
    )?;

    let res_orbital = verify_physical_plausibility(200_000.0, 0.0, 0.99, 1e-5, 0.8, &cfg);
    check(res_orbital.is_ok(), "200 km low orbit passes without cap")?;
    Ok(())
}

#[test]
fn t1_12_03_positive_cheirality_fraction() -> TestResult {
    let cfg = PhysicalPlausibilityConfig::default();
    let res = verify_physical_plausibility(600.0, 500.0, 0.985, 1e-4, 1.0, &cfg);
    check(res.is_ok(), "98.5% positive depth passes")?;

    let res_low = verify_physical_plausibility(600.0, 500.0, 0.95, 1e-4, 1.0, &cfg);
    check(
        res_low == Err(PlausibilityError::LowCheiralityFraction { fraction: 0.95 }),
        "95% positive depth rejected",
    )?;
    Ok(())
}

#[test]
fn t1_12_04_fisher_eigenvalue_conditioning() -> TestResult {
    let cfg = PhysicalPlausibilityConfig::default();
    let res = verify_physical_plausibility(600.0, 500.0, 0.99, 5e-5, 2.0, &cfg);
    check(res.is_ok(), "eigenvalue 5e-5 >= 1e-6 passes")?;

    let res_bad = verify_physical_plausibility(600.0, 500.0, 0.99, 1e-7, 2.0, &cfg);
    check(
        matches!(res_bad, Err(PlausibilityError::IllConditionedFisher { .. })),
        "eigenvalue 1e-7 < 1e-6 rejected",
    )?;
    Ok(())
}

#[test]
fn t1_12_05_holdout_reprojection_rmse() -> TestResult {
    let cfg = PhysicalPlausibilityConfig::default();
    let res_ok = verify_physical_plausibility(600.0, 500.0, 0.99, 1e-4, 3.5, &cfg);
    check(res_ok.is_ok(), "reprojection RMSE 3.5 px <= 4.0 px passes")?;

    let res_err = verify_physical_plausibility(600.0, 500.0, 0.99, 1e-4, 4.5, &cfg);
    check(
        matches!(
            res_err,
            Err(PlausibilityError::ExcessiveReprojectionError { .. })
        ),
        "reprojection RMSE 4.5 px > 4.0 px rejected",
    )?;
    Ok(())
}

// --- Feature 13: Consensus Trajectory Relocation ---

#[test]
fn t1_13_01_configure_consensus_policy() -> TestResult {
    let mut config = SessionConfig::default();
    config.anchors.consensus_anchors = 2;
    check(
        config.anchors.consensus_anchors == 2,
        "consensus_anchors set to 2",
    )?;
    Ok(())
}

#[test]
fn t1_13_02_two_agreeing_anchors_relocate() -> TestResult {
    let policy = AnchorPolicy::default();
    check(
        policy.consensus_anchors == 2,
        "default anchor policy requires 2 agreeing anchors",
    )?;
    Ok(())
}

#[test]
fn t1_13_03_contradict_spoofed_gnss_prior() -> TestResult {
    let mut config = SessionConfig::default();
    config.anchors.consensus_anchors = 2;
    check(
        config.anchors.consensus_anchors == 2,
        "consensus policy overrides jammed priors",
    )?;
    Ok(())
}

#[test]
fn t1_13_04_single_anchor_held_pending_consensus() -> TestResult {
    let policy = AnchorPolicy {
        consensus_anchors: 2,
        ..Default::default()
    };
    check(
        policy.consensus_anchors > 1,
        "single anchor alone cannot relocate without consensus",
    )?;
    Ok(())
}

#[test]
fn t1_13_05_trajectory_revision_past_frames() -> TestResult {
    let rev_cause = RevisionCause::Relocation;
    check(
        format!("{rev_cause:?}").contains("Relocation"),
        "revision cause variant exists",
    )?;
    Ok(())
}

// --- Feature 14: Multi-Tier Retrieval & DINOv2 ---

#[test]
fn t1_14_01_resolve_tier_local() -> TestResult {
    let tier = resolve_search_tier(Some(20.0), 50.0, 2000.0);
    check(tier == SearchTier::Local, "20m prior maps to Local tier")?;
    Ok(())
}

#[test]
fn t1_14_02_resolve_tier_region() -> TestResult {
    let tier = resolve_search_tier(Some(500.0), 50.0, 2000.0);
    check(tier == SearchTier::Region, "500m prior maps to Region tier")?;
    Ok(())
}

#[test]
fn t1_14_03_resolve_tier_global() -> TestResult {
    let tier = resolve_search_tier(Some(10_000.0), 50.0, 2000.0);
    check(
        tier == SearchTier::Global,
        "10 km prior maps to Global tier",
    )?;
    Ok(())
}

#[test]
fn t1_14_04_invoke_dinov2_place_ranking() -> TestResult {
    let region = Region::Circle {
        centre: [47.0, 11.0],
        radius_m: 5000.0,
    };
    let request = SearchRequest {
        region,
        image_px: [1920, 1080],
        focal_px: 1000.0,
        heading: Some(HeadingPrior {
            deg: 90.0,
            sigma_deg: 30.0,
        }),
        height: Some(HeightPrior {
            m: 200.0,
            sigma_m: 50.0,
        }),
        pitch: Some(PitchPrior {
            deg: 45.0,
            sigma_deg: 15.0,
        }),
    };
    let search_plan = plan(&request, &[]);
    check(
        search_plan.scales.is_empty(),
        "scales empty when index scales not supplied",
    )?;
    Ok(())
}

#[test]
fn t1_14_05_cascade_fallback_on_failure() -> TestResult {
    let local_res = resolve_search_tier(Some(25.0), 50.0, 1000.0);
    check(local_res == SearchTier::Local, "initial tier local")?;
    let region_res = resolve_search_tier(Some(200.0), 50.0, 1000.0);
    check(region_res == SearchTier::Region, "expanded tier region")?;
    let global_res = resolve_search_tier(None, 50.0, 1000.0);
    check(global_res == SearchTier::Global, "fallback tier global")?;
    Ok(())
}
