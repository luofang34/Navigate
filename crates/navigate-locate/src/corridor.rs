//! Sans-IO visual corridor tracking, kinematics, and session fusion state.
//!
//! A corridor tracker maintains 3D map landmarks, inter-frame visual odometry,
//! platform dynamics gates, and `VisualSession` fix admission without filesystem,
//! network, or thread operations.

pub mod kinematics;
pub mod multitier;
pub mod plausibility;
pub mod session;
pub mod tracker;

pub use kinematics::{
    CHI2_GATE_3DOF_99, KinematicState, MAX_DELTA_T_S, MIN_DELTA_T_S, TrackingSearchStrategy,
    VehicleDynamicsProfile, approx_dot, can_update_vertical_velocity, compute_dt_s,
    determine_search_strategy, match_tracks_kashin, match_virtual_kashin, orthonormalize_so3,
    project_track_strategy, should_force_anchor,
};
pub use multitier::{
    AnchorMatch, GlobalFallbackSeed, GlobalFallbackSeeds, MultiTierSearchConfig, SearchTierMatcher,
    SearchTierOutcome, build_search_prior_from_hint, descriptor_from_tokens_and_canvas,
    global_fallback_seeds_for_places, gray_mask_to_bools, ground_crop_to_canvas,
    load_candidate_tiles, load_candidate_tiles_geodetic, rank_candidate_regions,
    resolve_global_fallback_seeds, resolve_global_fallback_seeds_with_mask, resolve_search_tier,
    resolve_search_tier_from_prior,
};
pub use plausibility::{
    CorridorTerrain, PhysicalPlausibilityConfig, PlausibilityError,
    compute_fisher_geometry_covariance, compute_fisher_min_eigenvalue, compute_holdout_rmse_px,
    is_plausible_candidate, local_camera_rotation, verify_candidate_plausibility,
    verify_physical_plausibility, verify_prior_bounding,
};
pub use session::{
    AnchorWorkerState, DeadReckonEstimate, DeadReckonPath, DualPathDeadReckoner,
    FilterIntegritySnapshot, SessionAnchorReport, VisualCorridorSession,
};
pub use tracker::{
    CorridorTracker, OdometryStep, WarmTrackAttempt, kabsch_3d, relative_rotation_ransac,
    try_warm_track,
};
