//! Reference kinematics, dynamics profiles, triggers, and adaptive search radius.

use nalgebra::Matrix3;

/// Clamps interval time step to the approved operational window [1e-4, 60.0] seconds.
pub fn clamp_dt_s(dt_s: f64) -> f64 {
    if !dt_s.is_finite() || dt_s < 1e-4 {
        1e-4
    } else if dt_s > 60.0 {
        60.0
    } else {
        dt_s
    }
}

/// Continuous-time 6-DoF kinematic state with covariance representation.
#[derive(Debug, Clone)]
pub struct KinematicState {
    /// Position in local tangent plane (East-North-Up) in meters.
    pub eye_enu: [f64; 3],
    /// Translational velocity in meters per second.
    pub vel_enu_mps: [f64; 3],
    /// Camera attitude rotation matrix from world (ENU) to camera frame.
    pub rot_world_to_cam: Matrix3<f64>,
    /// Angular velocity in camera frame in radians per second.
    pub omega_cam_rad_s: [f64; 3],
    /// Position covariance matrix in meters squared.
    pub pos_cov_enu_m2: Matrix3<f64>,
    /// Attitude uncertainty standard deviation in radians.
    pub att_sigma_rad: f64,
}

impl Default for KinematicState {
    fn default() -> Self {
        Self::new([0.0, 0.0, 100.0], [0.0, 0.0, 0.0])
    }
}

impl KinematicState {
    /// Constructs a new kinematic state with initial position and velocity.
    pub fn new(eye_enu: [f64; 3], vel_enu_mps: [f64; 3]) -> Self {
        Self {
            eye_enu,
            vel_enu_mps,
            rot_world_to_cam: Matrix3::identity(),
            omega_cam_rad_s: [0.0, 0.0, 0.0],
            pos_cov_enu_m2: Matrix3::from_diagonal_element(1.0),
            att_sigma_rad: 0.01,
        }
    }

    /// Predicts position after time step dt_s using continuous kinematics.
    pub fn predict_position(&self, dt_s: f64) -> [f64; 3] {
        let dt = clamp_dt_s(dt_s);
        [
            self.eye_enu[0] + self.vel_enu_mps[0] * dt,
            self.eye_enu[1] + self.vel_enu_mps[1] * dt,
            self.eye_enu[2] + self.vel_enu_mps[2] * dt,
        ]
    }

    /// Propagates position and attitude covariance over interval dt_s.
    pub fn propagate_covariance(&mut self, dt_s: f64, a_rms_mps2: f64, omega_rms_rad_s: f64) {
        let dt = clamp_dt_s(dt_s);
        let q_pos = 0.5 * a_rms_mps2 * dt * dt;
        let pos_noise = Matrix3::from_diagonal_element(q_pos * q_pos);
        self.pos_cov_enu_m2 += pos_noise;
        self.att_sigma_rad += omega_rms_rad_s * dt;
    }

    /// Projects 3D position and attitude uncertainty to 2D pixel uncertainty.
    pub fn project_pixel_uncertainty(&self, fx: f64, fy: f64, median_depth_m: f64) -> f64 {
        let depth = if median_depth_m > 1.0 {
            median_depth_m
        } else {
            1.0
        };
        let focal = 0.5 * (fx.abs() + fy.abs());
        let pos_sigma = (self.pos_cov_enu_m2[(0, 0)] + self.pos_cov_enu_m2[(1, 1)]).sqrt();
        let pixel_from_pos = (pos_sigma / depth) * focal;
        let pixel_from_att = self.att_sigma_rad * focal;
        (pixel_from_pos * pixel_from_pos + pixel_from_att * pixel_from_att).sqrt()
    }

    /// 3D Mahalanobis innovation test against joint covariance.
    pub fn mahalanobis_innovation_gate(
        &self,
        cand_eye: [f64; 3],
        geom_pos_cov: Matrix3<f64>,
        chi2_gate: f64,
    ) -> bool {
        let diff = nalgebra::Vector3::new(
            cand_eye[0] - self.eye_enu[0],
            cand_eye[1] - self.eye_enu[1],
            cand_eye[2] - self.eye_enu[2],
        );
        let joint_cov = self.pos_cov_enu_m2 + geom_pos_cov;
        let Some(inv_cov) = joint_cov.try_inverse() else {
            return false;
        };
        let d2 = diff.dot(&(inv_cov * diff));
        d2.is_finite() && d2 <= chi2_gate
    }
}

/// Vehicle flight envelope profile defining velocity and acceleration limits.
#[derive(Debug, Clone, PartialEq)]
pub struct VehicleDynamicsProfile {
    /// Profile identifier preset name.
    pub name: String,
    /// Maximum horizontal speed in meters per second.
    pub max_speed_mps: f64,
    /// Maximum vertical speed in meters per second.
    pub max_vertical_speed_mps: f64,
    /// Root-mean-square acceleration in meters per second squared.
    pub rms_accel_mps2: f64,
    /// Maximum angular rate in radians per second.
    pub max_angular_rate_rad_s: f64,
}

impl VehicleDynamicsProfile {
    /// Rotary survey drone preset: low speed, low acceleration, moderate turn rate.
    pub fn rotary_survey() -> Self {
        Self {
            name: "rotary-survey".to_owned(),
            max_speed_mps: 25.0,
            max_vertical_speed_mps: 8.0,
            rms_accel_mps2: 5.0,
            max_angular_rate_rad_s: 1.5,
        }
    }

    /// Fixed wing aircraft preset: cruising speeds, bounded climb rates.
    pub fn fixed_wing() -> Self {
        Self {
            name: "fixed-wing".to_owned(),
            max_speed_mps: 80.0,
            max_vertical_speed_mps: 15.0,
            rms_accel_mps2: 12.0,
            max_angular_rate_rad_s: 2.5,
        }
    }

    /// High performance / jet platform preset up to 800 m/s, 100 m/s², 6.0 rad/s.
    pub fn high_performance() -> Self {
        Self {
            name: "high-performance".to_owned(),
            max_speed_mps: 800.0,
            max_vertical_speed_mps: 200.0,
            rms_accel_mps2: 100.0,
            max_angular_rate_rad_s: 6.0,
        }
    }

    /// Computes maximum expected displacement over interval dt_s.
    pub fn max_expected_displacement(&self, dt_s: f64) -> f64 {
        let dt = clamp_dt_s(dt_s);
        self.max_speed_mps * dt + 0.5 * self.rms_accel_mps2 * dt * dt
    }

    /// Clamps 3D velocity vector to profile limits.
    pub fn clamp_velocity(&self, vel: [f64; 3]) -> [f64; 3] {
        let h_speed = (vel[0] * vel[0] + vel[1] * vel[1]).sqrt();
        let scale = if h_speed > self.max_speed_mps && h_speed > 1e-6 {
            self.max_speed_mps / h_speed
        } else {
            1.0
        };
        let vz = vel[2].clamp(-self.max_vertical_speed_mps, self.max_vertical_speed_mps);
        [vel[0] * scale, vel[1] * scale, vz]
    }
}

/// Trigger configuration parameters for time and event schedules.
#[derive(Debug, Clone)]
pub struct TriggerConfig {
    /// Mask evaluation interval in seconds.
    pub mask_interval_s: f64,
    /// Anchor solver interval in seconds.
    pub anchor_interval_s: f64,
    /// Position uncertainty threshold triggering anchor acquisition in meters.
    pub uncertainty_threshold_m: f64,
    /// Map cell boundary stride size in meters.
    pub map_cell_stride_m: f64,
}

impl Default for TriggerConfig {
    fn default() -> Self {
        Self {
            mask_interval_s: 1.0,
            anchor_interval_s: 3.0,
            uncertainty_threshold_m: 50.0,
            map_cell_stride_m: 500.0,
        }
    }
}

/// State tracking last trigger timestamps and positions.
#[derive(Debug, Clone, Default)]
pub struct TriggerState {
    /// Timestamp of last mask trigger in seconds.
    pub last_mask_time_s: Option<f64>,
    /// Timestamp of last anchor trigger in seconds.
    pub last_anchor_time_s: Option<f64>,
    /// Last map cell coordinates [cell_x, cell_y].
    pub last_map_cell: Option<[i64; 2]>,
}

/// Trigger evaluation outcome flags.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TriggerOutcomes {
    /// True when mask evaluation must run.
    pub trigger_mask: bool,
    /// True when anchor acquisition must run.
    pub trigger_anchor: bool,
    /// True when map cell boundary was crossed.
    pub cell_boundary_crossed: bool,
}

impl TriggerState {
    /// Evaluates triggers against current time, position, and uncertainty.
    pub fn evaluate(
        &mut self,
        time_s: f64,
        pos_enu: [f64; 3],
        pos_sigma_m: f64,
        config: &TriggerConfig,
    ) -> TriggerOutcomes {
        let trigger_mask = match self.last_mask_time_s {
            None => true,
            Some(t) => (time_s - t) >= config.mask_interval_s,
        };
        if trigger_mask {
            self.last_mask_time_s = Some(time_s);
        }

        let time_anchor = match self.last_anchor_time_s {
            None => true,
            Some(t) => (time_s - t) >= config.anchor_interval_s,
        };
        let unc_anchor = pos_sigma_m >= config.uncertainty_threshold_m;

        let cell_x = (pos_enu[0] / config.map_cell_stride_m).floor() as i64;
        let cell_y = (pos_enu[1] / config.map_cell_stride_m).floor() as i64;
        let cell_boundary_crossed = match self.last_map_cell {
            None => false,
            Some(last) => last != [cell_x, cell_y],
        };
        self.last_map_cell = Some([cell_x, cell_y]);

        let trigger_anchor = time_anchor || unc_anchor || cell_boundary_crossed;
        if trigger_anchor {
            self.last_anchor_time_s = Some(time_s);
        }

        TriggerOutcomes {
            trigger_mask,
            trigger_anchor,
            cell_boundary_crossed,
        }
    }
}

/// Search mode selected by dynamic pixel uncertainty.
#[derive(Debug, Clone, PartialEq)]
pub enum SearchMode {
    /// Local radius search with pixel radius clamped to [24.0, 96.0].
    RadiusMatching(f64),
    /// Global feature matching across frustum via LighterGlue.
    GlobalLighterGlue,
}

/// Selects search mode from projected pixel uncertainty sigma_px.
pub fn determine_search_mode(sigma_px: f64) -> SearchMode {
    let gate_3sigma = 3.0 * sigma_px;
    if gate_3sigma <= 64.0 {
        let radius = gate_3sigma.clamp(24.0, 96.0);
        SearchMode::RadiusMatching(radius)
    } else {
        SearchMode::GlobalLighterGlue
    }
}
