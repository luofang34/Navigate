//! Sans-IO inter-frame visual odometry and warm 3D landmark tracking.

mod odometry;
mod planar_vo;
mod session_ops;

pub use odometry::{ColdDeadReckonPrior, kabsch_3d, relative_rotation_ransac};
pub use planar_vo::{WarmTrackAttempt, nadir_yaw_translation_ransac, try_warm_track};

use super::{
    kinematics::{
        KinematicState, VehicleDynamicsProfile, can_update_vertical_velocity, orthonormalize_so3,
    },
    session::VisualCorridorSession,
};
use nalgebra::{Matrix3, Vector3};
use navigate_match::{ActiveTrackSet, PreparedKashinQuery};
use navigate_visual::{holdout_pnp::PoseCandidate, session::MapCell};
use odometry::{estimate_derotated_pixel_flow, estimate_ground_vo_delta, match_interframe_top};
use std::collections::{BTreeMap, VecDeque};

pub(super) const MAX_ODOMETRY_STEPS: usize = 256;

/// One inter-frame odometry step stored for background anchor arrival propagation.
#[derive(Clone, Copy, Debug)]
pub struct OdometryStep {
    /// Monotonic frame sequence index.
    pub frame_seq: usize,
    /// Relative camera rotation matrix for this step.
    pub rot_step: Matrix3<f64>,
    /// Translational position delta `[east, north, up]` in meters.
    pub eye_delta: [f64; 3],
    /// Median derotated inter-frame pixel flow `[du, dv]` in render-space pixels.
    pub derotated_flow_px: Option<[f64; 2]>,
    /// Elapsed step interval in seconds.
    pub dt_s: f64,
    /// True when `eye_delta` came from a solved 3D world pose rather than cold ground `VO`.
    pub has_world_pose: bool,
}

/// Sans-IO 3D landmark, visual odometry, session, and dead-reckoning tracker state machine.
pub struct CorridorTracker {
    /// Active 3D map landmarks tracked across frames.
    pub active_tracks: ActiveTrackSet,
    /// Previous camera optical center in `ENU` meters.
    pub prev_eye: Option<[f64; 3]>,
    /// Previous translational velocity in meters per second.
    pub prev_vel: [f64; 3],
    /// Previous camera rotation from world `ENU` to camera coordinates.
    pub prev_rot: Option<Matrix3<f64>>,
    /// Most recent inter-frame relative rotation step.
    pub prev_rot_step: Matrix3<f64>,
    /// Previous frame's Kashin-projected query features.
    pub prev_prepared: Option<PreparedKashinQuery>,
    /// Accumulated visual-odometry orientation estimate.
    pub vo_rot: Option<Matrix3<f64>>,
    /// Processed frame sequence counter.
    pub frame_seq: usize,
    /// Bounded history of inter-frame odometry steps.
    pub pose_steps: VecDeque<OdometryStep>,
    /// Frames elapsed since the last accepted map anchor.
    pub frames_since_anchor: usize,
    /// Visited map cells synchronized with [`VisualSession`](navigate_visual::session::VisualSession).
    pub visited_cells: BTreeMap<MapCell, String>,
    /// Previous frame timestamp in seconds.
    pub prev_time_s: Option<f64>,
    /// Elapsed seconds since the last accepted map anchor.
    pub time_since_anchor_s: f64,
    /// Elapsed seconds since the last accepted visual pose (anchor or warm track).
    pub time_since_accepted_pose_s: f64,
    /// Previous frame's search-prior position in `ENU` meters.
    pub prev_prior_enu: Option<[f64; 3]>,
    /// Continuous-time kinematic state and covariance.
    pub kinematic: Option<KinematicState>,
    /// Count of inter-frame ground optical-flow matches on the current step.
    pub last_flow_vectors: usize,
    /// Most recent caller-supplied monotonic capture timestamp.
    pub last_monotonic_ns: Option<navigate_contract::MonotonicNanos>,
    /// Dead-reckoned search center maintained during cold acquisition (`prev_eye.is_none()`).
    pub cold_prior: Option<ColdDeadReckonPrior>,
    /// Embedded visual session, fix source, navigation filter, and dual-path dead reckoner.
    pub corridor_session: Option<VisualCorridorSession>,
}

impl Default for CorridorTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl CorridorTracker {
    /// Create an empty corridor tracker with an initialized visual session and dead reckoner.
    #[must_use]
    pub fn new() -> Self {
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
            time_since_accepted_pose_s: f64::INFINITY,
            prev_prior_enu: None,
            kinematic: None,
            last_flow_vectors: 0,
            last_monotonic_ns: None,
            cold_prior: None,
            corridor_session: None,
        }
    }

    /// Record one inter-frame visual-odometry step for `cur_q`.
    pub fn record_odometry_step(
        &mut self,
        cur_q: &PreparedKashinQuery,
        k_render: &Matrix3<f64>,
        prior_rot_hint: Option<Matrix3<f64>>,
        nominal_agl_m: f64,
        dt_s: f64,
        profile: &VehicleDynamicsProfile,
    ) {
        self.frame_seq = self.frame_seq.wrapping_add(1);
        let base_rot = self.prev_rot.or(self.vo_rot).or(prior_rot_hint);
        let mut rot_step = Matrix3::identity();
        let mut matched_pairs = (Vec::new(), Vec::new());

        if let Some(prev_q) = &self.prev_prepared {
            let fx = k_render[(0, 0)];
            let fy = k_render[(1, 1)];
            let max_px = profile
                .max_pixel_displacement(dt_s, fx, fy, nominal_agl_m)
                .clamp(96.0, 400.0);
            let vo_cap = if self.prev_eye.is_some() && self.active_tracks.len() >= 96 {
                96
            } else {
                192
            };
            matched_pairs = match_interframe_top(prev_q, cur_q, vo_cap, max_px);
            let use_planar_nadir = base_rot.map_or(nominal_agl_m <= 250.0, |r| -r[(2, 2)] > 0.75);
            let ransac_fit = if matched_pairs.0.len() >= 12 {
                if use_planar_nadir {
                    nadir_yaw_translation_ransac(&matched_pairs.0, &matched_pairs.1, k_render, 4.0)
                        .or_else(|| {
                            relative_rotation_ransac(
                                &matched_pairs.0,
                                &matched_pairs.1,
                                k_render,
                                4.0,
                            )
                        })
                } else {
                    relative_rotation_ransac(&matched_pairs.0, &matched_pairs.1, k_render, 4.0)
                }
            } else {
                None
            };
            if let Some((r_delta, inliers)) = ransac_fit {
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
        self.last_flow_vectors = matched_pairs.0.len();

        let eye_delta = if let Some(pe) = self.prev_eye {
            let pred_eye = self.kinematic.as_ref().map_or_else(
                || {
                    [
                        pe[0] + self.prev_vel[0] * dt_s,
                        pe[1] + self.prev_vel[1] * dt_s,
                        pe[2] + self.prev_vel[2] * dt_s,
                    ]
                },
                |k| k.predict_position(dt_s),
            );
            [
                pred_eye[0] - pe[0],
                pred_eye[1] - pe[1],
                pred_eye[2] - pe[2],
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

        let derotated_flow_px =
            estimate_derotated_pixel_flow(&matched_pairs.0, &matched_pairs.1, &rot_step, k_render);
        self.vo_rot = base_rot.map(|r| orthonormalize_so3(&(rot_step * r)));
        self.prev_rot_step = rot_step;
        if self.prev_eye.is_none() {
            self.advance_cold_prior(
                eye_delta,
                &rot_step,
                derotated_flow_px,
                k_render,
                nominal_agl_m,
                dt_s,
                profile,
                base_rot.is_some(),
            );
        }
        if self.pose_steps.len() >= MAX_ODOMETRY_STEPS {
            let _ = self.pose_steps.pop_front();
        }
        self.pose_steps.push_back(OdometryStep {
            frame_seq: self.frame_seq,
            rot_step,
            eye_delta,
            derotated_flow_px,
            dt_s: dt_s.max(0.0),
            has_world_pose: self.prev_eye.is_some(),
        });
    }

    /// Propagate an anchor pose forward across odometry steps recorded after `sent_frame_seq`.
    #[must_use]
    pub fn propagate_pose_from(
        &self,
        sent_frame_seq: usize,
        init_eye: [f64; 3],
        init_rot: Matrix3<f64>,
    ) -> ([f64; 3], Matrix3<f64>) {
        let mut eye = init_eye;
        let mut rot = init_rot;
        for step in &self.pose_steps {
            if step.frame_seq > sent_frame_seq && step.frame_seq <= self.frame_seq {
                eye[0] += step.eye_delta[0];
                eye[1] += step.eye_delta[1];
                eye[2] += step.eye_delta[2];
                rot = orthonormalize_so3(&(step.rot_step * rot));
            }
        }
        (eye, rot)
    }

    /// Update tracker motion, kinematic state, and dead reckoner from an accepted pose candidate.
    pub fn update_motion_state(
        &mut self,
        cand: &PoseCandidate,
        prepared: PreparedKashinQuery,
        dt_s: f64,
        profile: &VehicleDynamicsProfile,
    ) {
        let mut exact_delta = [
            self.prev_vel[0] * dt_s,
            self.prev_vel[1] * dt_s,
            self.prev_vel[2] * dt_s,
        ];
        let allow_vert = can_update_vertical_velocity(cand.score.holdout_count, cand.score.inliers);
        if let Some(pe) = self.prev_eye {
            let dv = Vector3::from(cand.eye) - Vector3::from(pe);
            let max_disp = profile.max_displacement(dt_s);
            if dv.norm() <= (max_disp * 1.5).max(10.0) {
                exact_delta = [dv.x, dv.y, dv.z];
            }
        }
        let mut exact_rot = self.prev_rot_step;
        if let Some(pr) = self.prev_rot {
            exact_rot = orthonormalize_so3(&(cand.rotation * pr.transpose()));
            self.prev_rot_step = exact_rot;
        }
        if self.prev_eye.is_some()
            && let Some(last) = self.pose_steps.back_mut()
            && last.frame_seq == self.frame_seq
        {
            last.eye_delta = exact_delta;
            last.rot_step = exact_rot;
        }
        let fisher_pos = super::plausibility::compute_fisher_geometry_covariance(cand)
            .fixed_view::<3, 3>(0, 0)
            .into_owned();
        if let Some(kin) = self.kinematic.as_mut() {
            kin.update_from_pose_gated(
                cand.eye,
                cand.rotation,
                dt_s,
                profile,
                allow_vert,
                Some(fisher_pos),
            );
            self.prev_vel = kin.vel_enu_mps;
        } else {
            let mut kin = KinematicState::new(cand.eye, cand.rotation);
            kin.pos_cov_enu_m2 = fisher_pos + Matrix3::identity() * 2.25;
            self.kinematic = Some(kin);
        }
        if let (Some(session), Some(kin)) = (self.corridor_session.as_mut(), self.kinematic.clone())
        {
            session.dead_reckoner.kinematic = kin;
        }
        self.prev_eye = Some(cand.eye);
        self.prev_rot = Some(cand.rotation);
        self.vo_rot = Some(cand.rotation);
        self.cold_prior = None;
        self.prev_prepared = Some(prepared);
        self.time_since_accepted_pose_s = 0.0;
    }

    /// Coast tracker state across an unsolved frame via [`DualPathDeadReckoner`](super::session::DualPathDeadReckoner).
    pub fn advance_unsolved_prepared(
        &mut self,
        prepared: PreparedKashinQuery,
        dt_s: f64,
        profile: &VehicleDynamicsProfile,
    ) {
        let a_rms = profile.rms_accel_mps2;
        let omega_rms = profile.max_angular_rate_rad_s * 0.1;

        if let Some(pe) = self.prev_eye {
            if let Some(session) = self.corridor_session.as_mut() {
                if let Some(kin) = self.kinematic.clone() {
                    session.dead_reckoner.kinematic = kin;
                } else {
                    let mut kin =
                        KinematicState::new(pe, self.prev_rot.unwrap_or_else(Matrix3::identity));
                    kin.vel_enu_mps = self.prev_vel;
                    session.dead_reckoner.kinematic = kin;
                }
                let est = session.dead_reckon(dt_s, self.last_flow_vectors, a_rms, omega_rms);
                self.prev_eye = Some(est.eye_enu);
                self.kinematic = Some(session.dead_reckoner.kinematic.clone());
                if !session.dead_reckoner.external_sensors_available && self.last_flow_vectors < 6 {
                    session.advance_epoch();
                }
                if let Some(last) = self.pose_steps.back_mut()
                    && last.frame_seq == self.frame_seq
                {
                    last.eye_delta = [
                        est.eye_enu[0] - pe[0],
                        est.eye_enu[1] - pe[1],
                        est.eye_enu[2] - pe[2],
                    ];
                }
            } else if let Some(kin) = self.kinematic.as_mut() {
                kin.propagate_covariance(dt_s, a_rms, omega_rms);
                let pred = kin.predict_position(dt_s);
                kin.eye_enu = pred;
                self.prev_eye = Some(pred);
            }
        } else if let Some(session) = self.corridor_session.as_mut()
            && !session.dead_reckoner.external_sensors_available
            && self.last_flow_vectors < 6
        {
            session.advance_epoch();
        }

        if let Some(pr) = self.prev_rot {
            self.prev_rot = Some(orthonormalize_so3(&(self.prev_rot_step * pr)));
            if let Some(kin) = self.kinematic.as_mut()
                && let Some(r_now) = self.prev_rot
            {
                kin.rot_world_to_cam = r_now;
            }
        }
        self.prev_prepared = Some(prepared);
        self.time_since_accepted_pose_s += dt_s.max(0.0);
    }

    /// Predict the next frame camera pose from dead-reckoned kinematics and inter-frame rotation.
    #[must_use]
    pub fn predict_interframe_pose(
        &self,
        cur_q: &PreparedKashinQuery,
        k: &Matrix3<f64>,
        dt_s: f64,
        profile: &VehicleDynamicsProfile,
    ) -> Option<([f64; 3], Matrix3<f64>)> {
        if !dt_s.is_finite() || dt_s > 5.0 {
            return None;
        }
        let (prev_eye, prev_rot) = (self.prev_eye?, self.prev_rot?);
        let pred_eye = if let Some(kin) = &self.kinematic {
            kin.predict_position(dt_s)
        } else if let Some(session) = &self.corridor_session {
            session.dead_reckoner.kinematic.predict_position(dt_s)
        } else {
            [
                prev_eye[0] + self.prev_vel[0] * dt_s,
                prev_eye[1] + self.prev_vel[1] * dt_s,
                prev_eye[2] + self.prev_vel[2] * dt_s,
            ]
        };
        let mut pred_rot = self.prev_rot_step * prev_rot;
        let already_stepped = self
            .pose_steps
            .back()
            .is_some_and(|s| s.frame_seq == self.frame_seq);
        if !already_stepped && let Some(prev_q) = &self.prev_prepared {
            let fx = k[(0, 0)];
            let fy = k[(1, 1)];
            let max_px = profile
                .max_pixel_displacement(dt_s, fx, fy, 480.0)
                .clamp(96.0, 400.0);
            let (px_a, px_b) = match_interframe_top(prev_q, cur_q, 192, max_px);
            let ransac_fit = if px_a.len() >= 12 {
                if -prev_rot[(2, 2)] > 0.75 {
                    nadir_yaw_translation_ransac(&px_a, &px_b, k, 4.0)
                        .or_else(|| relative_rotation_ransac(&px_a, &px_b, k, 4.0))
                } else {
                    relative_rotation_ransac(&px_a, &px_b, k, 4.0)
                }
            } else {
                None
            };
            if let Some((r_delta, inliers)) = ransac_fit {
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
}

#[cfg(test)]
mod tests;
