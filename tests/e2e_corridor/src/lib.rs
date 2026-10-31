//! End-to-end corridor testing harness, reference models, and validation utilities.

pub mod harness;
pub mod kinematics_model;
pub mod nadir_model;
pub mod plausibility_model;
pub mod session_model;

pub use harness::{TestResult, approx_eq, check, synthetic_dem_tile, synthetic_photo_case};
pub use kinematics_model::{
    KinematicState, SearchMode, TriggerConfig, TriggerState, VehicleDynamicsProfile,
    determine_search_mode,
};
pub use nadir_model::{
    DynamicMockSession, FocalLengthEstimator, GRAZING_PITCH_SEEDS_DEG, MilliradianCovarianceScaler,
    RansacHorizonEstimator, condition_number_6x6, direction_cosine, generate_corridor_seeds,
    is_nadir_pose, match_nadir_yaw_band, min_cos_orientation_for_pose, rotate_image_gray,
    synthesize_nadir_yaw_bands, unrotate_pixel,
};
pub use plausibility_model::{
    MaskAwareSpatialHoldout, PhysicalPlausibilityConfig, PlausibilityError, SearchTier,
    SearchTierOutcome, resolve_search_tier, verify_physical_plausibility,
};
pub use session_model::{
    AnchorWorkerState, DeadReckonEstimate, DeadReckonPath, DualPathDeadReckoner,
    VisualCorridorSession, can_update_vertical_velocity,
};
