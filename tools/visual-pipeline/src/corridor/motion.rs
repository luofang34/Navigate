//! CLI adapter around [`navigate_locate::corridor::CorridorTracker`].

use super::kinematics::{TrackingSearchStrategy, VehicleDynamicsProfile};
use crate::{PipelineError, pose::PoseCandidate, realtime::helpers::log_attempt};
use nalgebra::Matrix3;
use navigate_match::{ActiveTrackSet, PreparedKashinQuery};
use serde_json::Value;

pub(super) type TrackState = navigate_locate::corridor::CorridorTracker;
#[allow(unused_imports)]
pub(super) use navigate_locate::corridor::OdometryStep;

pub(super) fn record_odometry_step(
    state: &mut TrackState,
    cur_q: &PreparedKashinQuery,
    k_render: &Matrix3<f64>,
    prior_rot_hint: Option<Matrix3<f64>>,
    nominal_agl_m: f64,
    dt_s: f64,
    profile: &VehicleDynamicsProfile,
) {
    state.record_odometry_step(
        cur_q,
        k_render,
        prior_rot_hint,
        nominal_agl_m,
        dt_s,
        profile,
    );
}

pub(super) fn propagate_pose_from(
    state: &TrackState,
    sent_frame_seq: usize,
    init_eye: [f64; 3],
    init_rot: Matrix3<f64>,
) -> ([f64; 3], Matrix3<f64>) {
    state.propagate_pose_from(sent_frame_seq, init_eye, init_rot)
}

pub(super) fn update_motion_state(
    state: &mut TrackState,
    cand: &PoseCandidate,
    prepared: PreparedKashinQuery,
    dt_s: f64,
    profile: &VehicleDynamicsProfile,
) {
    state.update_motion_state(cand, prepared, dt_s, profile);
}

pub(super) fn advance_unsolved_prepared(
    state: &mut TrackState,
    prepared: PreparedKashinQuery,
    dt_s: f64,
    profile: &VehicleDynamicsProfile,
) {
    state.advance_unsolved_prepared(prepared, dt_s, profile);
}

pub(super) fn predict_interframe_pose(
    state: &TrackState,
    cur_q: &PreparedKashinQuery,
    k: &Matrix3<f64>,
    dt_s: f64,
    profile: &VehicleDynamicsProfile,
) -> Option<([f64; 3], Matrix3<f64>)> {
    state.predict_interframe_pose(cur_q, k, dt_s, profile)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn try_warm_track(
    tracks: &ActiveTrackSet,
    prepared: &PreparedKashinQuery,
    pred: ([f64; 3], Matrix3<f64>),
    k_render: &Matrix3<f64>,
    render_size: [u32; 2],
    min_inliers: usize,
    attempts_log: &mut Vec<Value>,
    strategy: TrackingSearchStrategy,
) -> Result<Option<PoseCandidate>, PipelineError> {
    let (attempt, accepted) = navigate_locate::corridor::try_warm_track(
        tracks,
        prepared,
        pred,
        k_render,
        render_size,
        min_inliers,
        strategy,
    )?;
    log_attempt(
        attempts_log,
        attempt.method,
        0,
        0,
        1.0,
        attempt.matched_landmarks,
        attempt.candidate.as_ref(),
    );
    Ok(accepted)
}
