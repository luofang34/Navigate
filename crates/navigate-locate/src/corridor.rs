//! Sans-IO visual corridor tracking, kinematics, and session fusion state.
//!
//! A corridor tracker maintains 3D map landmarks, inter-frame visual odometry,
//! platform dynamics gates, and `VisualSession` fix admission without filesystem,
//! network, or thread operations.

pub mod kinematics;
pub mod session;
pub mod tracker;

pub use kinematics::{
    CHI2_GATE_3DOF_99, KinematicState, MAX_DELTA_T_S, MIN_DELTA_T_S, TrackingSearchStrategy,
    VehicleDynamicsProfile, approx_dot, can_update_vertical_velocity, compute_dt_s,
    determine_search_strategy, match_tracks_kashin, match_virtual_kashin, orthonormalize_so3,
    project_track_strategy, should_force_anchor,
};
pub use session::{
    AnchorWorkerState, DeadReckonEstimate, DeadReckonPath, DualPathDeadReckoner,
    VisualCorridorSession,
};
pub use tracker::{
    CorridorTracker, OdometryStep, WarmTrackAttempt, kabsch_3d, relative_rotation_ransac,
    try_warm_track,
};
