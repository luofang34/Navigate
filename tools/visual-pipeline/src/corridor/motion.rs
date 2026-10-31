//! Inter-frame motion prediction and warm 3D track solving for `corridor-track`.

use crate::{
    PipelineError,
    pose::{PoseCandidate, fit_pose_hybrid, fit_pose_l1_gnc},
    realtime::helpers::{log_attempt, rot_to_row_major},
    video_ops::relative_rotation_ransac,
};
use nalgebra::{Matrix3, Vector3};
use navigate_match::{ActiveTrackSet, PreparedKashinQuery, TrackMatchConfig};
use navigate_visual_session::MapCell;
use serde_json::Value;
use std::collections::BTreeMap;

pub(super) struct TrackState {
    pub(super) active_tracks: ActiveTrackSet,
    pub(super) prev_eye: Option<[f64; 3]>,
    pub(super) prev_vel: [f64; 3],
    pub(super) prev_rot: Option<Matrix3<f64>>,
    pub(super) prev_rot_step: Matrix3<f64>,
    pub(super) prev_prepared: Option<PreparedKashinQuery>,
    pub(super) frames_since_anchor: usize,
    pub(super) visited_cells: BTreeMap<MapCell, String>,
}

impl TrackState {
    pub(super) fn new() -> Self {
        Self {
            active_tracks: ActiveTrackSet::empty(),
            prev_eye: None,
            prev_vel: [0.0; 3],
            prev_rot: None,
            prev_rot_step: Matrix3::identity(),
            prev_prepared: None,
            frames_since_anchor: 0,
            visited_cells: BTreeMap::new(),
        }
    }
}

pub(super) fn update_motion_state(
    state: &mut TrackState,
    cand: &PoseCandidate,
    prepared: PreparedKashinQuery,
) {
    if let Some(pe) = state.prev_eye {
        let dv = Vector3::from(cand.eye) - Vector3::from(pe);
        if dv.norm() <= 500.0 {
            state.prev_vel = [dv.x, dv.y, dv.z];
        }
    }
    if let Some(pr) = state.prev_rot {
        state.prev_rot_step = cand.rotation * pr.transpose();
    }
    state.prev_eye = Some(cand.eye);
    state.prev_rot = Some(cand.rotation);
    state.prev_prepared = Some(prepared);
}

pub(super) fn predict_interframe_pose(
    state: &TrackState,
    cur_q: &PreparedKashinQuery,
    k: &Matrix3<f64>,
) -> Option<([f64; 3], Matrix3<f64>)> {
    let (prev_eye, prev_rot) = (state.prev_eye?, state.prev_rot?);
    let pred_eye = [
        prev_eye[0] + 0.85 * state.prev_vel[0],
        prev_eye[1] + 0.85 * state.prev_vel[1],
        prev_eye[2] + 0.85 * state.prev_vel[2],
    ];
    let mut pred_rot = state.prev_rot_step * prev_rot;
    if let Some(prev_q) = &state.prev_prepared {
        let (px_a, px_b) = match_interframe_top(prev_q, cur_q, 192);
        if px_a.len() >= 12
            && let Some((r_delta, inliers)) = relative_rotation_ransac(&px_a, &px_b, k, 4.0)
        {
            let inl_n = inliers.iter().filter(|&&b| b).count();
            let trace_cos = ((r_delta.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
            if inl_n >= 10 && trace_cos >= 15.0_f64.to_radians().cos() {
                pred_rot = r_delta * prev_rot;
            }
        }
    }
    Some((pred_eye, pred_rot))
}

fn match_interframe_top(
    prev_q: &PreparedKashinQuery,
    cur_q: &PreparedKashinQuery,
    cap: usize,
) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let na = prev_q.valid_indices().len().min(cap);
    let nb = cur_q.valid_indices().len().min(cap);
    let (sa, sb) = (prev_q.sign_bits(), cur_q.sign_bits());
    let (pa, pb) = (prev_q.pixels(), cur_q.pixels());
    let (ia, ib) = (prev_q.valid_indices(), cur_q.valid_indices());
    let mut out_a = Vec::new();
    let mut out_b = Vec::new();
    for a in 0..na {
        let [ax, ay] = pa[ia[a]];
        let mut best = (usize::MAX, 65_u32, 65_u32);
        for b in 0..nb {
            let [bx, by] = pb[ib[b]];
            if (ax - bx).abs() > 96.0 || (ay - by).abs() > 96.0 {
                continue;
            }
            let hd = (sa[a] ^ sb[b]).count_ones();
            if hd < best.1 {
                best = (b, hd, best.1);
            } else if hd < best.2 {
                best.2 = hd;
            }
        }
        if best.0 != usize::MAX && best.1 <= 14 && best.1 + 3 <= best.2 {
            out_a.push([ax, ay]);
            out_b.push(pb[ib[best.0]]);
        }
    }
    (out_a, out_b)
}

pub(super) fn try_warm_track(
    tracks: &ActiveTrackSet,
    prepared: &PreparedKashinQuery,
    pred: ([f64; 3], Matrix3<f64>),
    k_render: &Matrix3<f64>,
    render_size: [u32; 2],
    min_inliers: usize,
    attempts_log: &mut Vec<Value>,
) -> Result<Option<PoseCandidate>, PipelineError> {
    let cfg = TrackMatchConfig {
        search_radius_px: 36.0,
        hamming_gate: 22,
        min_dot: 0.64,
        ratio_limit: 0.90,
    };
    let intrinsics = [
        k_render[(0, 0)],
        k_render[(1, 1)],
        k_render[(0, 2)],
        k_render[(1, 2)],
    ];
    let matches = tracks.match_radius(
        prepared,
        pred.0,
        &rot_to_row_major(&pred.1),
        intrinsics,
        render_size,
        &cfg,
    )?;
    let mut world = Vec::with_capacity(matches.len());
    let mut px = Vec::with_capacity(matches.len());
    for m in &matches {
        world.push(m.world_point_enu);
        px.push(m.query_pixel);
    }
    let cand = fit_pose_l1_gnc(&world, &px, k_render, render_size, pred)
        .or_else(|| fit_pose_hybrid(&world, &px, k_render, render_size, Some(pred)));
    log_attempt(
        attempts_log,
        "kashin-3d-radius-track+l1-gnc",
        0,
        0,
        1.0,
        world.len(),
        cand.as_ref(),
    );
    Ok(cand.filter(|c| c.score.geometry_accepted && c.score.inliers >= min_inliers))
}
