//! Inter-frame motion prediction, relative odometry history, and warm 3D track solving.

use super::kinematics::{
    KinematicState, TrackingSearchStrategy, VehicleDynamicsProfile, can_update_vertical_velocity,
    match_tracks_kashin, orthonormalize_so3,
};
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
use std::collections::{BTreeMap, VecDeque};

const MAX_ODOMETRY_STEPS: usize = 256;

#[derive(Clone, Copy, Debug)]
pub(super) struct OdometryStep {
    pub frame_seq: usize,
    pub rot_step: Matrix3<f64>,
    pub eye_delta: [f64; 3],
}

pub(super) struct TrackState {
    pub(super) active_tracks: ActiveTrackSet,
    pub(super) prev_eye: Option<[f64; 3]>,
    pub(super) prev_vel: [f64; 3],
    pub(super) prev_rot: Option<Matrix3<f64>>,
    pub(super) prev_rot_step: Matrix3<f64>,
    pub(super) prev_prepared: Option<PreparedKashinQuery>,
    pub(super) vo_rot: Option<Matrix3<f64>>,
    pub(super) frame_seq: usize,
    pub(super) pose_steps: VecDeque<OdometryStep>,
    pub(super) frames_since_anchor: usize,
    pub(super) visited_cells: BTreeMap<MapCell, String>,
    pub(super) prev_time_s: Option<f64>,
    pub(super) time_since_anchor_s: f64,
    pub(super) kinematic: Option<KinematicState>,
    #[allow(dead_code)]
    pub(super) focal_estimator: crate::stages::focal::FocalLengthEstimator,
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
            vo_rot: None,
            frame_seq: 0,
            pose_steps: VecDeque::with_capacity(MAX_ODOMETRY_STEPS),
            frames_since_anchor: 0,
            visited_cells: BTreeMap::new(),
            prev_time_s: None,
            time_since_anchor_s: 0.0,
            kinematic: None,
            focal_estimator: crate::stages::focal::FocalLengthEstimator::default(),
        }
    }
}

pub(super) fn record_odometry_step(
    state: &mut TrackState,
    cur_q: &PreparedKashinQuery,
    k_render: &Matrix3<f64>,
    prior_rot_hint: Option<Matrix3<f64>>,
    nominal_agl_m: f64,
    dt_s: f64,
    profile: &VehicleDynamicsProfile,
) {
    state.frame_seq += 1;
    let base_rot = state.prev_rot.or(state.vo_rot).or(prior_rot_hint);
    let mut rot_step = Matrix3::identity();
    let mut matched_pairs = (Vec::new(), Vec::new());

    if let Some(prev_q) = &state.prev_prepared {
        let fx = k_render[(0, 0)];
        let fy = k_render[(1, 1)];
        let max_px = profile
            .max_pixel_displacement(dt_s, fx, fy, nominal_agl_m)
            .clamp(96.0, 400.0);
        matched_pairs = match_interframe_top(prev_q, cur_q, 192, max_px);
        if matched_pairs.0.len() >= 12
            && let Some((r_delta, inliers)) =
                relative_rotation_ransac(&matched_pairs.0, &matched_pairs.1, k_render, 4.0)
        {
            let inl_n = inliers.iter().filter(|&&b| b).count();
            let trace_cos = ((r_delta.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
            let min_cos = profile
                .min_trace_cosine(dt_s)
                .min(18.0_f64.to_radians().cos());
            if inl_n >= 8 && trace_cos >= min_cos {
                rot_step = orthonormalize_so3(&r_delta);
            }
        }
    }

    let eye_delta = if state.prev_eye.is_some() {
        [
            state.prev_vel[0] * dt_s,
            state.prev_vel[1] * dt_s,
            state.prev_vel[2] * dt_s,
        ]
    } else if let Some(r_world_to_cam) = base_rot {
        estimate_ground_vo_delta(
            &matched_pairs.0,
            &matched_pairs.1,
            &rot_step,
            &r_world_to_cam,
            k_render,
            nominal_agl_m,
            dt_s,
            profile,
        )
    } else {
        [0.0; 3]
    };

    state.vo_rot = base_rot.map(|r| orthonormalize_so3(&(rot_step * r)));
    state.prev_rot_step = rot_step;
    if state.pose_steps.len() >= MAX_ODOMETRY_STEPS {
        let _ = state.pose_steps.pop_front();
    }
    state.pose_steps.push_back(OdometryStep {
        frame_seq: state.frame_seq,
        rot_step,
        eye_delta,
    });
}

#[allow(clippy::too_many_arguments)]
fn estimate_ground_vo_delta(
    px_a: &[[f64; 2]],
    px_b: &[[f64; 2]],
    rot_step: &Matrix3<f64>,
    world_to_cam: &Matrix3<f64>,
    k: &Matrix3<f64>,
    agl_m: f64,
    dt_s: f64,
    profile: &VehicleDynamicsProfile,
) -> [f64; 3] {
    if px_a.len() < 8 || !agl_m.is_finite() || agl_m < 15.0 {
        return [0.0; 3];
    }
    let fx = k[(0, 0)].max(1e-6);
    let fy = k[(1, 1)].max(1e-6);
    let cx = k[(0, 2)];
    let cy = k[(1, 2)];
    let cam_to_world = world_to_cam.transpose();
    let mut east_samples = Vec::with_capacity(px_a.len());
    let mut north_samples = Vec::with_capacity(px_a.len());
    let (filter_gate, clamp_gate) = profile.ground_vo_bounds(dt_s);

    for (&[ax, ay], &[bx, by]) in px_a.iter().zip(px_b.iter()) {
        let ray_a = Vector3::new((ax - cx) / fx, (ay - cy) / fy, 1.0);
        let dir_world = cam_to_world * (ray_a / ray_a.norm().max(1e-9));
        if dir_world.z >= -0.15 {
            continue;
        }
        let rot_ray = rot_step * ray_a;
        if rot_ray.z <= 1e-3 {
            continue;
        }
        let ax_derot = fx * (rot_ray.x / rot_ray.z) + cx;
        let ay_derot = fy * (rot_ray.y / rot_ray.z) + cy;
        let slant_m = (agl_m / (-dir_world.z)).clamp(15.0, 8_000.0);
        let z_cam = slant_m / ray_a.norm().max(1.0);
        let d_cam = Vector3::new(
            -z_cam * (bx - ax_derot) / fx,
            -z_cam * (by - ay_derot) / fy,
            0.0,
        );
        let d_enu = cam_to_world * d_cam;
        if d_enu.x.hypot(d_enu.y) <= filter_gate {
            east_samples.push(d_enu.x);
            north_samples.push(d_enu.y);
        }
    }
    if east_samples.len() < 6 {
        return [0.0; 3];
    }
    east_samples.sort_by(f64::total_cmp);
    north_samples.sort_by(f64::total_cmp);
    let mid = east_samples.len() / 2;
    [
        east_samples[mid].clamp(-clamp_gate, clamp_gate),
        north_samples[mid].clamp(-clamp_gate, clamp_gate),
        0.0,
    ]
}

pub(super) fn propagate_pose_from(
    state: &TrackState,
    sent_frame_seq: usize,
    init_eye: [f64; 3],
    init_rot: Matrix3<f64>,
) -> ([f64; 3], Matrix3<f64>) {
    let mut eye = init_eye;
    let mut rot = init_rot;
    for step in &state.pose_steps {
        if step.frame_seq > sent_frame_seq && step.frame_seq <= state.frame_seq {
            eye[0] += step.eye_delta[0];
            eye[1] += step.eye_delta[1];
            eye[2] += step.eye_delta[2];
            rot = orthonormalize_so3(&(step.rot_step * rot));
        }
    }
    (eye, rot)
}

pub(super) fn update_motion_state(
    state: &mut TrackState,
    cand: &PoseCandidate,
    prepared: PreparedKashinQuery,
    dt_s: f64,
    profile: &VehicleDynamicsProfile,
) {
    let mut exact_delta = [
        state.prev_vel[0] * dt_s,
        state.prev_vel[1] * dt_s,
        state.prev_vel[2] * dt_s,
    ];
    let allow_vert = can_update_vertical_velocity(cand.score.holdout_count, cand.score.inliers);
    if let Some(pe) = state.prev_eye {
        let dv = Vector3::from(cand.eye) - Vector3::from(pe);
        let max_disp = profile.max_displacement(dt_s);
        if dv.norm() <= (max_disp * 1.5).max(10.0) {
            exact_delta = [dv.x, dv.y, dv.z];
            let v_mps = dv / dt_s;
            let vz = if allow_vert {
                v_mps.z
            } else {
                state.prev_vel[2]
            };
            state.prev_vel = [v_mps.x, v_mps.y, vz];
        }
    }
    let mut exact_rot = state.prev_rot_step;
    if let Some(pr) = state.prev_rot {
        exact_rot = orthonormalize_so3(&(cand.rotation * pr.transpose()));
        state.prev_rot_step = exact_rot;
    }
    if state.prev_eye.is_some()
        && let Some(last) = state.pose_steps.back_mut()
        && last.frame_seq == state.frame_seq
    {
        last.eye_delta = exact_delta;
        last.rot_step = exact_rot;
    }
    if let Some(kin) = state.kinematic.as_mut() {
        kin.update_from_pose_gated(cand.eye, cand.rotation, dt_s, profile, allow_vert);
    } else {
        state.kinematic = Some(KinematicState::new(cand.eye, cand.rotation));
    }
    state.prev_eye = Some(cand.eye);
    state.prev_rot = Some(cand.rotation);
    state.vo_rot = Some(cand.rotation);
    state.prev_prepared = Some(prepared);
}

pub(super) fn advance_unsolved_prepared(
    state: &mut TrackState,
    prepared: PreparedKashinQuery,
    dt_s: f64,
    profile: &VehicleDynamicsProfile,
) {
    if let Some(pe) = state.prev_eye {
        let step_delta = state
            .pose_steps
            .back()
            .filter(|s| s.frame_seq == state.frame_seq)
            .map_or(
                [
                    state.prev_vel[0] * dt_s,
                    state.prev_vel[1] * dt_s,
                    state.prev_vel[2] * dt_s,
                ],
                |s| s.eye_delta,
            );
        state.prev_eye = Some([
            pe[0] + step_delta[0],
            pe[1] + step_delta[1],
            pe[2] + step_delta[2],
        ]);
    }
    if let Some(pr) = state.prev_rot {
        state.prev_rot = Some(orthonormalize_so3(&(state.prev_rot_step * pr)));
    }
    if let Some(kin) = state.kinematic.as_mut() {
        kin.propagate_covariance(
            dt_s,
            profile.rms_accel_mps2,
            profile.max_angular_rate_rad_s * 0.1,
        );
    }
    state.prev_prepared = Some(prepared);
}

pub(super) fn predict_interframe_pose(
    state: &TrackState,
    cur_q: &PreparedKashinQuery,
    k: &Matrix3<f64>,
    dt_s: f64,
    profile: &VehicleDynamicsProfile,
) -> Option<([f64; 3], Matrix3<f64>)> {
    let (prev_eye, prev_rot) = (state.prev_eye?, state.prev_rot?);
    let pred_eye = if let Some(kin) = &state.kinematic {
        kin.predict_position(dt_s)
    } else {
        [
            prev_eye[0] + state.prev_vel[0] * dt_s,
            prev_eye[1] + state.prev_vel[1] * dt_s,
            prev_eye[2] + state.prev_vel[2] * dt_s,
        ]
    };
    let mut pred_rot = state.prev_rot_step * prev_rot;
    if let Some(prev_q) = &state.prev_prepared {
        let fx = k[(0, 0)];
        let fy = k[(1, 1)];
        let max_px = profile
            .max_pixel_displacement(dt_s, fx, fy, 480.0)
            .clamp(96.0, 400.0);
        let (px_a, px_b) = match_interframe_top(prev_q, cur_q, 192, max_px);
        if px_a.len() >= 12
            && let Some((r_delta, inliers)) = relative_rotation_ransac(&px_a, &px_b, k, 4.0)
        {
            let inl_n = inliers.iter().filter(|&&b| b).count();
            let trace_cos = ((r_delta.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
            let min_cos = profile
                .min_trace_cosine(dt_s)
                .min(15.0_f64.to_radians().cos());
            if inl_n >= 10 && trace_cos >= min_cos {
                pred_rot = r_delta * prev_rot;
            }
        }
    }
    Some((pred_eye, orthonormalize_so3(&pred_rot)))
}

fn match_interframe_top(
    prev_q: &PreparedKashinQuery,
    cur_q: &PreparedKashinQuery,
    cap: usize,
    max_pixel_disp: f64,
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
            if (ax - bx).abs() > max_pixel_disp || (ay - by).abs() > max_pixel_disp {
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
    let (world, px) = match strategy {
        TrackingSearchStrategy::RadiusSearch { radius_px } => {
            let cfg = TrackMatchConfig {
                search_radius_px: radius_px as f32,
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
            let mut w = Vec::with_capacity(matches.len());
            let mut p = Vec::with_capacity(matches.len());
            for m in &matches {
                w.push(m.world_point_enu);
                p.push(m.query_pixel);
            }
            (w, p)
        }
        TrackingSearchStrategy::GlobalFrustumMatch => match_tracks_kashin(tracks, prepared, 384),
    };
    let cand = fit_pose_l1_gnc(&world, &px, k_render, render_size, pred)
        .or_else(|| fit_pose_hybrid(&world, &px, k_render, render_size, Some(pred)));
    let label = match strategy {
        TrackingSearchStrategy::RadiusSearch { .. } => "kashin-3d-radius-track+l1-gnc",
        TrackingSearchStrategy::GlobalFrustumMatch => "kashin-3d-global-frustum+l1-gnc",
    };
    log_attempt(attempts_log, label, 0, 0, 1.0, world.len(), cand.as_ref());
    Ok(cand.filter(|c| c.score.geometry_accepted && c.score.inliers >= min_inliers))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Rotation3;

    #[test]
    fn propagate_pose_from_accumulates_interframe_steps_after_sent_frame() {
        let mut state = TrackState::new();
        let step_rot =
            Rotation3::from_axis_angle(&Vector3::z_axis(), 2.0_f64.to_radians()).into_inner();
        state.frame_seq = 1;
        state.pose_steps.push_back(OdometryStep {
            frame_seq: 1,
            rot_step: step_rot,
            eye_delta: [100.0, 0.0, 0.0],
        });
        state.frame_seq = 2;
        state.pose_steps.push_back(OdometryStep {
            frame_seq: 2,
            rot_step: step_rot,
            eye_delta: [4.0, -1.5, 0.25],
        });
        state.frame_seq = 3;
        state.pose_steps.push_back(OdometryStep {
            frame_seq: 3,
            rot_step: step_rot,
            eye_delta: [5.0, -2.5, 0.75],
        });

        let (eye, rot) = propagate_pose_from(&state, 1, [10.0, 20.0, 500.0], Matrix3::identity());
        assert!((eye[0] - 19.0).abs() < 1e-9);
        assert!((eye[1] - 16.0).abs() < 1e-9);
        assert!((eye[2] - 501.0).abs() < 1e-9);
        let expected_rot =
            Rotation3::from_axis_angle(&Vector3::z_axis(), 4.0_f64.to_radians()).into_inner();
        assert!((rot - expected_rot).norm() < 1e-9);
    }
}
