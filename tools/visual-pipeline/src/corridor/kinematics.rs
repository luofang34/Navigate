//! Re-exports of sans-IO vehicle kinematics and uncertainty projection from `navigate-locate`.

pub use navigate_locate::corridor::kinematics::{
    CHI2_GATE_3DOF_99, KinematicState, MAX_DELTA_T_S, MIN_DELTA_T_S, TrackingSearchStrategy,
    VehicleDynamicsProfile, approx_dot, can_update_vertical_velocity, compute_dt_s,
    determine_search_strategy, match_tracks_kashin, match_virtual_kashin, orthonormalize_so3,
    project_track_strategy, should_force_anchor,
};
