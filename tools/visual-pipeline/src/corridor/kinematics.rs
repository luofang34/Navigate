//! Physical vehicle kinematics, dynamics profiles, and uncertainty projection.
//!
//! This module provides continuous-time motion prediction parameterized by elapsed time,
//! dynamic parameter gating for diverse aerial platforms, covariance propagation, and
//! adaptive feature search selection.

use nalgebra::{Matrix3, Vector3};
use navigate_match::{ActiveTrackSet, PreparedKashinQuery, VirtualCorridorView};
use navigate_visual_session::MapCell;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Maximum valid time step in seconds between consecutive processed frames.
pub const MAX_DELTA_T_S: f64 = 60.0;

/// Minimum valid time step in seconds between consecutive processed frames.
pub const MIN_DELTA_T_S: f64 = 1e-4;

/// Chi-squared innovation gate threshold for 3 degrees of freedom at 99% confidence level.
pub const CHI2_GATE_3DOF_99: f64 = 11.345;

/// Evaluate whether vertical velocity can update based on holdout count or inlier count.
#[must_use]
pub fn can_update_vertical_velocity(holdout_count: usize, inliers: usize) -> bool {
    holdout_count >= 12 || inliers >= 48
}

/// Compute the time interval in seconds between two video frames.
#[must_use]
pub fn compute_dt_s(cur_time_s: Option<f64>, prev_time_s: Option<f64>, default_fps: f64) -> f64 {
    let dt = match (cur_time_s, prev_time_s) {
        (Some(cur), Some(prev)) if cur > prev => cur - prev,
        _ => 1.0 / default_fps.max(0.1),
    };
    dt.clamp(MIN_DELTA_T_S, MAX_DELTA_T_S)
}

/// Dynamic physical motion constraints for an aerial platform.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct VehicleDynamicsProfile {
    /// Maximum expected horizontal and 3D translational speed in meters per second.
    pub max_speed_mps: f64,
    /// Maximum expected vertical speed in meters per second.
    pub max_vertical_speed_mps: f64,
    /// Root mean square acceleration noise in meters per second squared.
    pub rms_accel_mps2: f64,
    /// Maximum expected angular rotation rate in radians per second.
    pub max_angular_rate_rad_s: f64,
}

impl VehicleDynamicsProfile {
    /// Preset configuration for survey quadcopters and slow rotorcraft.
    #[must_use]
    pub fn rotary_survey() -> Self {
        Self {
            max_speed_mps: 25.0,
            max_vertical_speed_mps: 5.0,
            rms_accel_mps2: 3.0,
            max_angular_rate_rad_s: 1.5,
        }
    }

    /// Preset configuration for fixed-wing mapping and reconnaissance aircraft.
    #[must_use]
    pub fn fixed_wing() -> Self {
        Self {
            max_speed_mps: 80.0,
            max_vertical_speed_mps: 15.0,
            rms_accel_mps2: 15.0,
            max_angular_rate_rad_s: 2.5,
        }
    }

    /// Preset configuration for high-speed, high-g aerial vehicles.
    #[must_use]
    pub fn high_performance() -> Self {
        Self {
            max_speed_mps: 800.0,
            max_vertical_speed_mps: 100.0,
            rms_accel_mps2: 100.0,
            max_angular_rate_rad_s: 6.0,
        }
    }

    /// Select a preset dynamics profile by name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "rotary-survey" => Some(Self::rotary_survey()),
            "fixed-wing" => Some(Self::fixed_wing()),
            "high-performance" => Some(Self::high_performance()),
            _ => None,
        }
    }

    /// Compute the maximum plausible 3D displacement across interval `dt_s`.
    #[must_use]
    pub fn max_displacement(&self, dt_s: f64) -> f64 {
        self.max_speed_mps * dt_s + 0.5 * self.rms_accel_mps2 * dt_s * dt_s
    }

    /// Compute the maximum plausible vertical displacement across interval `dt_s`.
    #[must_use]
    pub fn max_vertical_displacement(&self, dt_s: f64) -> f64 {
        self.max_vertical_speed_mps * dt_s + 0.5 * self.rms_accel_mps2 * dt_s * dt_s
    }

    /// Compute the minimum trace cosine of a valid rotation step across interval `dt_s`.
    #[must_use]
    pub fn min_trace_cosine(&self, dt_s: f64) -> f64 {
        let max_angle = (self.max_angular_rate_rad_s * dt_s + 0.15).min(std::f64::consts::PI);
        max_angle.cos()
    }

    /// Compute the maximum plausible pixel shift for inter-frame correspondence matching.
    #[must_use]
    pub fn max_pixel_displacement(&self, dt_s: f64, fx: f64, fy: f64, min_depth_m: f64) -> f64 {
        let f_max = fx.max(fy).max(100.0);
        let rot_disp = f_max * (self.max_angular_rate_rad_s * dt_s);
        let lin_disp = f_max * (self.max_displacement(dt_s) / min_depth_m.max(5.0));
        rot_disp + lin_disp
    }

    /// Derive ground visual odometry sample rejection and clamp thresholds.
    #[must_use]
    pub fn ground_vo_bounds(&self, dt_s: f64) -> (f64, f64) {
        let disp = self.max_displacement(dt_s);
        let filter_gate = (disp * 1.5).clamp(10.0, 1500.0);
        let clamp_gate = disp.clamp(5.0, 1000.0);
        (filter_gate, clamp_gate)
    }
}

impl Default for VehicleDynamicsProfile {
    fn default() -> Self {
        Self::rotary_survey()
    }
}

/// Continuous-time kinematic tracking state in local East-North-Up coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct KinematicState {
    /// Camera optical center position `[east, north, up]` in meters.
    pub eye_enu: [f64; 3],
    /// Translational velocity `[v_east, v_north, v_up]` in meters per second.
    pub vel_enu_mps: [f64; 3],
    /// Orthonormal orientation mapping world ENU coordinates to camera coordinates.
    pub rot_world_to_cam: Matrix3<f64>,
    /// Camera body angular velocity `[wx, wy, wz]` in radians per second.
    pub omega_cam_rad_s: [f64; 3],
    /// 3x3 position error covariance matrix in meters squared.
    pub pos_cov_enu_m2: Matrix3<f64>,
    /// Attitude orientation uncertainty standard deviation in radians.
    pub att_sigma_rad: f64,
}

impl KinematicState {
    /// Initialize a new kinematic state from an accepted camera pose.
    #[must_use]
    pub fn new(eye_enu: [f64; 3], rot_world_to_cam: Matrix3<f64>) -> Self {
        Self {
            eye_enu,
            vel_enu_mps: [0.0; 3],
            rot_world_to_cam,
            omega_cam_rad_s: [0.0; 3],
            pos_cov_enu_m2: Matrix3::identity() * 1.0,
            att_sigma_rad: 0.01,
        }
    }

    /// Predict the camera position after elapsed interval `dt_s`.
    #[must_use]
    pub fn predict_position(&self, dt_s: f64) -> [f64; 3] {
        [
            self.eye_enu[0] + self.vel_enu_mps[0] * dt_s,
            self.eye_enu[1] + self.vel_enu_mps[1] * dt_s,
            self.eye_enu[2] + self.vel_enu_mps[2] * dt_s,
        ]
    }

    /// Propagate state covariance across interval `dt_s` using platform noise parameters.
    pub fn propagate_covariance(&mut self, dt_s: f64, a_rms_mps2: f64, omega_rms_rad_s: f64) {
        let q_pos = 0.5 * a_rms_mps2 * dt_s * dt_s;
        self.pos_cov_enu_m2 += Matrix3::identity() * q_pos;
        self.att_sigma_rad += omega_rms_rad_s * dt_s;
    }

    /// Project position covariance and attitude uncertainty to pixel standard deviation.
    #[must_use]
    pub fn project_pixel_uncertainty(&self, fx: f64, fy: f64, median_depth_m: f64) -> f64 {
        let cov_cam =
            self.rot_world_to_cam * self.pos_cov_enu_m2 * self.rot_world_to_cam.transpose();
        let depth = median_depth_m.max(1.0);
        let depth_sq = depth * depth;
        let att_var = self.att_sigma_rad * self.att_sigma_rad;

        let var_u = (cov_cam[(0, 0)].max(0.0) / depth_sq) + att_var;
        let var_v = (cov_cam[(1, 1)].max(0.0) / depth_sq) + att_var;

        let sigma_u = fx.abs() * var_u.sqrt();
        let sigma_v = fy.abs() * var_v.sqrt();
        sigma_u.max(sigma_v)
    }

    /// Test whether a candidate position falls within the Mahalanobis gate.
    #[must_use]
    pub fn mahalanobis_innovation_gate(
        &self,
        cand_eye: [f64; 3],
        geom_pos_cov: Matrix3<f64>,
        chi2_gate: f64,
    ) -> bool {
        let diff = Vector3::new(
            cand_eye[0] - self.eye_enu[0],
            cand_eye[1] - self.eye_enu[1],
            cand_eye[2] - self.eye_enu[2],
        );
        (self.pos_cov_enu_m2 + geom_pos_cov)
            .try_inverse()
            .is_some_and(|inv| {
                let d2 = diff.dot(&(inv * diff));
                d2.is_finite() && d2 <= chi2_gate
            })
    }

    /// Update kinematic state from an observation pose.
    pub fn update_from_pose(
        &mut self,
        new_eye: [f64; 3],
        new_rot: Matrix3<f64>,
        dt_s: f64,
        profile: &VehicleDynamicsProfile,
    ) {
        self.update_from_pose_gated(new_eye, new_rot, dt_s, profile, true);
    }

    /// Update kinematic state from an observation pose with vertical velocity gating.
    pub fn update_from_pose_gated(
        &mut self,
        new_eye: [f64; 3],
        new_rot: Matrix3<f64>,
        dt_s: f64,
        profile: &VehicleDynamicsProfile,
        allow_vertical: bool,
    ) {
        let dt = dt_s.clamp(MIN_DELTA_T_S, MAX_DELTA_T_S);
        let de = Vector3::new(
            new_eye[0] - self.eye_enu[0],
            new_eye[1] - self.eye_enu[1],
            new_eye[2] - self.eye_enu[2],
        );
        let v_cand = de / dt;
        if v_cand.norm() <= profile.max_speed_mps * 1.5 {
            let vz = if allow_vertical {
                v_cand.z
            } else {
                self.vel_enu_mps[2]
            };
            self.vel_enu_mps = [v_cand.x, v_cand.y, vz];
        }

        let r_delta = new_rot * self.rot_world_to_cam.transpose();
        let cos_theta = ((r_delta.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
        let theta = cos_theta.acos();
        if theta > 1e-5 && theta.sin().abs() > 1e-6 {
            let axis = Vector3::new(
                r_delta[(2, 1)] - r_delta[(1, 2)],
                r_delta[(0, 2)] - r_delta[(2, 0)],
                r_delta[(1, 0)] - r_delta[(0, 1)],
            ) / (2.0 * theta.sin());
            let omega = axis * (theta / dt);
            if omega.norm() <= profile.max_angular_rate_rad_s * 1.5 {
                self.omega_cam_rad_s = [omega.x, omega.y, omega.z];
            }
        }

        self.eye_enu = new_eye;
        self.rot_world_to_cam = new_rot;
        self.pos_cov_enu_m2 = Matrix3::identity() * 1.0;
        self.att_sigma_rad = 0.01;
    }
}

/// Strategy for feature correspondence matching based on pixel uncertainty.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TrackingSearchStrategy {
    /// Bounded radius search around projected 3D landmark positions.
    RadiusSearch {
        /// Search radius in pixels.
        radius_px: f64,
    },
    /// Global landmark matching across the entire camera frustum.
    GlobalFrustumMatch,
}

/// Determine the tracking search strategy from the projected pixel uncertainty.
#[must_use]
pub fn determine_search_strategy(sigma_px: f64) -> TrackingSearchStrategy {
    let gate_px = 3.0 * sigma_px;
    if gate_px <= 64.0 {
        TrackingSearchStrategy::RadiusSearch {
            radius_px: gate_px.clamp(24.0, 96.0),
        }
    } else {
        TrackingSearchStrategy::GlobalFrustumMatch
    }
}

/// Project active track landmarks to derive median depth and select feature matching strategy.
#[must_use]
pub fn project_track_strategy(
    kin: Option<&KinematicState>,
    pred_pose: Option<([f64; 3], Matrix3<f64>)>,
    active_tracks: &ActiveTrackSet,
    k_render: &Matrix3<f64>,
    nominal_agl_m: f64,
) -> TrackingSearchStrategy {
    if let (Some(kin), Some((pred_eye, pred_rot))) = (kin, pred_pose) {
        let depth = if !active_tracks.is_empty() {
            let mut depths: Vec<f64> = active_tracks
                .world_points_enu
                .iter()
                .map(|w| {
                    pred_rot[(2, 0)] * (w[0] - pred_eye[0])
                        + pred_rot[(2, 1)] * (w[1] - pred_eye[1])
                        + pred_rot[(2, 2)] * (w[2] - pred_eye[2])
                })
                .filter(|&z| z > 1.0)
                .collect();
            if depths.is_empty() {
                nominal_agl_m.max(15.0)
            } else {
                depths.sort_by(f64::total_cmp);
                depths[depths.len() / 2]
            }
        } else {
            nominal_agl_m.max(15.0)
        };
        let sigma_px = kin.project_pixel_uncertainty(k_render[(0, 0)], k_render[(1, 1)], depth);
        determine_search_strategy(sigma_px)
    } else {
        TrackingSearchStrategy::RadiusSearch { radius_px: 36.0 }
    }
}

/// Evaluate multi-criteria triggers to determine if an anchor refresh must occur.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn should_force_anchor(
    opts_anchor_interval: usize,
    frames_since_anchor: usize,
    opts_anchor_interval_s: Option<f64>,
    time_since_anchor_s: f64,
    opts_anchor_uncertainty_m: Option<f64>,
    kin: Option<&KinematicState>,
    anchor_on_cell_boundary: bool,
    pred_pose: Option<&([f64; 3], Matrix3<f64>)>,
    visited_cells: &BTreeMap<MapCell, String>,
    cell_m: f64,
) -> bool {
    let frame_hit = opts_anchor_interval > 0 && frames_since_anchor >= opts_anchor_interval;
    let time_hit = opts_anchor_interval_s.is_some_and(|s| time_since_anchor_s >= s);
    let cell_hit = anchor_on_cell_boundary
        && pred_pose.is_some_and(|(eye, _)| {
            let east = (eye[0] / cell_m).floor() as i64;
            let north = (eye[1] / cell_m).floor() as i64;
            !visited_cells
                .keys()
                .any(|c| c.east == east && c.north == north)
        });
    let unc_hit = opts_anchor_uncertainty_m.is_some_and(|th| {
        kin.is_some_and(|k| (k.pos_cov_enu_m2[(0, 0)] + k.pos_cov_enu_m2[(1, 1)]).sqrt() >= th)
    });
    frame_hit || time_hit || cell_hit || unc_hit
}

/// Project an arbitrary 3x3 matrix onto the SO(3) rotation group via singular value decomposition.
#[must_use]
pub fn orthonormalize_so3(m: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = m.svd(true, true);
    if let (Some(u), Some(v_t)) = (svd.u, svd.v_t) {
        let mut r = u * v_t;
        if r.determinant() < 0.0 {
            let mut u_fix = u;
            u_fix.set_column(2, &(-u.column(2)));
            r = u_fix * v_t;
        }
        r
    } else {
        Matrix3::identity()
    }
}

/// Approximate cosine dot product between a Kashin i8 descriptor and a query vector.
#[must_use]
pub fn approx_dot(code: &[u8; 64], rot: &[f32; 64]) -> f32 {
    let mut dot = 0.0_f32;
    let mut norm_sq = 0_i32;
    for (&raw, &q) in code.iter().zip(rot) {
        let v = raw as i8;
        let vi = i32::from(v);
        norm_sq += vi * vi;
        dot += f32::from(v) * q;
    }
    if norm_sq <= 0 {
        return 0.0;
    }
    (dot / (norm_sq as f32).sqrt()).clamp(-1.0, 1.0)
}

/// Match active 3D landmarks against a query frame without spatial radius restriction.
#[must_use]
pub fn match_tracks_kashin(
    tracks: &ActiveTrackSet,
    query: &PreparedKashinQuery,
    cap: usize,
) -> (Vec<[f64; 3]>, Vec<[f64; 2]>) {
    let n_r = tracks.len().min(cap);
    let q_valid = query.valid_indices();
    let n_q = q_valid.len().min(cap);
    if n_r == 0 || n_q == 0 {
        return (Vec::new(), Vec::new());
    }
    let q_signs = query.sign_bits();
    let q_rots = query.rotated();
    let q_px = query.pixels();
    let mut fwd = vec![(usize::MAX, -2.0_f32, -2.0_f32); n_r];
    let mut rev = vec![(usize::MAX, -2.0_f32); n_q];

    for (r, fwd_slot) in fwd.iter_mut().enumerate() {
        let rs = tracks.sign_bits[r];
        let rc = &tracks.kashin_i8[r];
        for q in 0..n_q {
            if (rs ^ q_signs[q]).count_ones() > 24 {
                continue;
            }
            let dot = approx_dot(rc, &q_rots[q]);
            if dot > fwd_slot.1 {
                *fwd_slot = (q, dot, fwd_slot.1);
            } else if dot > fwd_slot.2 {
                fwd_slot.2 = dot;
            }
            if dot > rev[q].1 {
                rev[q] = (r, dot);
            }
        }
    }
    let mut world = Vec::new();
    let mut px = Vec::new();
    for (r, &(q, b1, b2)) in fwd.iter().enumerate() {
        if q < n_q
            && rev[q].0 == r
            && b1 >= 0.62
            && (b2 <= -1.5 || (1.0 - b1).max(0.0) <= 0.85 * (1.0 - b2).max(0.0))
        {
            world.push(tracks.world_points_enu[r]);
            px.push(q_px[q_valid[q]]);
        }
    }
    (world, px)
}

/// Match virtual corridor landmarks against a query frame.
#[must_use]
pub fn match_virtual_kashin(
    virt: &VirtualCorridorView,
    query: &PreparedKashinQuery,
    cap: usize,
) -> (Vec<[f64; 3]>, Vec<[f64; 2]>) {
    let probe = ActiveTrackSet {
        world_points_enu: virt.world_points_enu.iter().take(cap).copied().collect(),
        sign_bits: virt.sign_bits.iter().take(cap).copied().collect(),
        kashin_i8: virt.kashin_i8.iter().take(cap).copied().collect(),
    };
    match_tracks_kashin(&probe, query, cap)
}

#[cfg(test)]
mod tests;
