//! Per-frame timing, lookahead tile scheduling, feature extraction, and cold-prior setup.

use super::FlightVisualEngine;
use crate::{
    anchor::{AnchorPrior, LOW_AGL_FOCAL_MULTIPLIERS, OBLIQUE_FOCAL_MULTIPLIERS},
    error::FlightVisualError,
    masking::AmortizedMasker,
    surface::Surface,
    tile_builder::synthesize_corridor_tiles,
    types::{
        AttitudePrior, CalibrationMode, CameraIntrinsics, DeadReckonStep, ExecutionPolicy,
        LiveFrameInput,
    },
};
use image::GrayImage;
use nalgebra::Matrix3;
use navigate_locate::corridor::{ColdDeadReckonPrior, CorridorTracker, compute_dt_s};
use navigate_match::SparseFeatures;
use navigate_visual::CameraModel;
use std::time::Instant;

impl FlightVisualEngine {
    pub(super) fn step_lookahead_tiles(
        &mut self,
        anchor_prior: AnchorPrior,
        virtual_now_ms: u64,
        collect_diag: bool,
    ) -> Result<(Vec<f64>, usize), FlightVisualError> {
        let mut tile_build_ms = Vec::new();
        let mut sync_built = 0_usize;
        if !self.config.enable_live_tile_synth {
            return Ok((tile_build_ms, sync_built));
        }
        if let Some(tb) = self.tile_builder.as_mut() {
            tb.poll_with_policy(
                &self.resources.store,
                self.config.policy,
                virtual_now_ms,
                &mut tile_build_ms,
            )?;
            if let Some(la) =
                estimate_lookahead_enu(&self.resources.surface, &self.state, anchor_prior)
            {
                tb.request_neighborhood(
                    &self.resources.surface,
                    &self.resources.store,
                    la,
                    virtual_now_ms,
                )?;
            }
        } else if !self.config.background_workers
            && let Some(la) =
                estimate_lookahead_enu(&self.resources.surface, &self.state, anchor_prior)
        {
            let ck = self.resources.store.tile_key_at(la[0], la[1]);
            if !self.resources.store.has_tile(ck) {
                let t_tb = Instant::now();
                let entries = synthesize_corridor_tiles(
                    &self.resources.surface,
                    Some(&self.sync_gpu_renderer),
                    &self.resources.xfeat,
                    &self.resources.projector,
                    ck,
                    self.tile_params,
                )?;
                if !entries.is_empty() {
                    for entry in entries {
                        self.resources.store.insert_tile(entry, true)?;
                        self.sync_tiles_built = self.sync_tiles_built.saturating_add(1);
                        sync_built = sync_built.saturating_add(1);
                    }
                    if collect_diag {
                        tile_build_ms.push(t_tb.elapsed().as_secs_f64() * 1000.0);
                    }
                }
            }
        }
        Ok((tile_build_ms, sync_built))
    }

    pub(super) fn effective_intrinsics(&self, mut intr: CameraIntrinsics) -> CameraIntrinsics {
        if matches!(intr.calibration_confidence, CalibrationMode::Uncalibrated)
            && let Some(locked_px) = self.focal_estimator.locked_focal_px
        {
            intr.fx = locked_px;
            intr.fy = locked_px;
        }
        intr
    }

    pub(super) fn ensure_session_configured(
        &mut self,
        render_size: [u32; 2],
        k_render: &Matrix3<f64>,
        cell_m: f64,
    ) -> Result<(), FlightVisualError> {
        if !self.session_configured {
            let cam_model = CameraModel {
                width: render_size[0].max(32),
                height: render_size[1].max(32),
                fx: k_render[(0, 0)].max(10.0),
                fy: k_render[(1, 1)].max(10.0),
                cx: k_render[(0, 2)],
                cy: k_render[(1, 2)],
            };
            self.state
                .configure_session_for_body(
                    cam_model,
                    cell_m,
                    self.resources.local_frame.reference_ellipsoid(),
                    self.config.attitude_budget,
                )
                .map_err(FlightVisualError::Invalid)?;
            self.session_configured = true;
        }
        Ok(())
    }

    pub(super) fn extract_query_features(
        &self,
        qry_gray: &GrayImage,
        mask_small: &GrayImage,
        valid_fraction: f64,
        anchor_busy: bool,
    ) -> Result<(SparseFeatures, f64, bool), FlightVisualError> {
        let kp_limit = self.config.search_opts.query_keypoints.clamp(16, 4096);
        let live_flight = matches!(self.config.policy, ExecutionPolicy::LiveFlight);
        let cold_paced_vo = live_flight
            && anchor_busy
            && self.state.active_tracks.is_empty()
            && self.state.prev_prepared.is_some();
        let is_coreml = self.resources.xfeat.runtime_name() == "coreml";
        let next_seq = self.state.frame_seq.wrapping_add(1);
        let cold_stride = if is_coreml { 12 } else { 6 };
        let skip_cold_xfeat = cold_paced_vo && !next_seq.is_multiple_of(cold_stride);
        let warm_paced_track = live_flight && !self.state.active_tracks.is_empty();
        let max_confirm = if live_flight { 1 } else { 2 };
        let provisional_pending = self.active_status.is_some_and(|st| {
            st.confidence == crate::types::AnchorConfidence::Provisional
                && st.confirm_attempts < max_confirm
        });
        let subwindow_warm = !is_coreml && valid_fraction < 0.70;
        let warm_kp_cap = if anchor_busy {
            if subwindow_warm {
                512
            } else if is_coreml {
                256
            } else {
                384
            }
        } else if is_coreml {
            256
        } else if subwindow_warm || provisional_pending || self.state.active_tracks.len() < 160 {
            640
        } else {
            512
        };
        let qry_kp_cap = if valid_fraction <= 0.0 || skip_cold_xfeat {
            0
        } else if cold_paced_vo {
            kp_limit.min(256)
        } else if warm_paced_track {
            let base = kp_limit.min(warm_kp_cap);
            if is_coreml && valid_fraction < 0.70 {
                let cells = (qry_gray.width() / 8).saturating_mul(qry_gray.height() / 8);
                let sub = ((f64::from(cells) * valid_fraction * 0.55).round() as usize)
                    .clamp(256, kp_limit);
                base.min(sub)
            } else {
                base
            }
        } else if !self.state.active_tracks.is_empty() && anchor_busy {
            kp_limit.min(768)
        } else if valid_fraction < 0.70 {
            let cells = (qry_gray.width() / 8).saturating_mul(qry_gray.height() / 8);
            let subwindow_cap =
                ((f64::from(cells) * valid_fraction * 0.55).round() as usize).clamp(1280, kp_limit);
            kp_limit.min(subwindow_cap)
        } else {
            kp_limit
        };
        if qry_kp_cap == 0 {
            return Ok((SparseFeatures::empty(), 0.0, skip_cold_xfeat));
        }
        let t_x = Instant::now();
        let feat = self
            .resources
            .xfeat
            .extract_masked(qry_gray, Some(mask_small), qry_kp_cap)?;
        Ok((feat, t_x.elapsed().as_secs_f64() * 1000.0, false))
    }

    pub(super) fn advance_timing_and_continuity(
        &mut self,
        frame: &LiveFrameInput<'_>,
        anchor_prior: AnchorPrior,
        frame_label: &str,
    ) -> Result<(f64, bool), FlightVisualError> {
        let cur_time_s = (frame.stamp.acquired_at.as_nanos() as f64) * 1e-9;
        let prev_time_s = self
            .state
            .last_monotonic_ns
            .map(|p| (p.as_nanos() as f64) * 1e-9);
        let mut discontinuous = false;
        let dt_s = match prev_time_s {
            Some(prev) if (cur_time_s - prev) > 5.0 => {
                discontinuous = true;
                (cur_time_s - prev).max(5.001)
            }
            _ => compute_dt_s(Some(cur_time_s), prev_time_s, 30.0),
        };
        let cur_prior_enu = anchor_prior.lat_lon().map(|(lat, lon)| {
            self.resources
                .surface
                .local(lat, lon, anchor_prior.alt_msl_m.unwrap_or(0.0))
        });
        if let (Some(cp), Some(pp)) = (cur_prior_enu, self.state.prev_prior_enu) {
            let jump_m = (cp[0] - pp[0]).hypot(cp[1] - pp[1]);
            if jump_m > 3.0 * self.config.dynamics_profile.max_displacement(dt_s) {
                discontinuous = true;
            }
        }
        self.state.prev_prior_enu = cur_prior_enu;
        self.state.prev_time_s = Some(cur_time_s);
        if discontinuous {
            self.state.reset_discontinuous_track();
            self.active_status = None;
            self.recent_backup = None;
            if matches!(
                frame.intrinsics.calibration_confidence,
                CalibrationMode::Uncalibrated
            ) {
                self.focal_estimator.locked_focal_px = None;
            }
        } else if prev_time_s.is_some() {
            self.state.time_since_anchor_s += dt_s;
        }
        self.state
            .observe_frame_monotonic(frame.stamp.acquired_at, "", frame_label)
            .map_err(FlightVisualError::NonMonotonicTimestamp)?;
        Ok((dt_s, discontinuous))
    }

    pub(super) fn sync_uncalibrated_regime(
        &mut self,
        intrinsics: CameraIntrinsics,
        nominal_agl_m: f64,
        attitude_override: Option<(f64, f64)>,
        k_render: &Matrix3<f64>,
    ) {
        if matches!(
            intrinsics.calibration_confidence,
            CalibrationMode::Uncalibrated
        ) && self.focal_estimator.locked_focal_px.is_none()
        {
            let low_agl_nadir = nominal_agl_m <= 250.0
                && attitude_override.is_none_or(|(d, _)| d >= 70.0)
                && self.masker.horizon_attitude(k_render).is_none();
            self.focal_estimator.sweep_multipliers = if low_agl_nadir {
                LOW_AGL_FOCAL_MULTIPLIERS
            } else {
                OBLIQUE_FOCAL_MULTIPLIERS
            };
        }
    }

    pub(super) fn init_cold_prior_if_needed(
        &mut self,
        frame: &LiveFrameInput<'_>,
        anchor_prior: AnchorPrior,
        nominal_agl_m: f64,
    ) {
        if self.state.prev_eye.is_none()
            && self.state.cold_prior.is_none()
            && let Some((lat, lon)) = anchor_prior.lat_lon()
        {
            let dem = self.resources.surface.sample_dem(lat, lon).unwrap_or(0.0);
            let alt = anchor_prior
                .alt_msl_m
                .unwrap_or(dem + nominal_agl_m)
                .max(dem + 15.0);
            let c_enu = self.resources.surface.local(lat, lon, alt);
            let sigma_0 = frame
                .filter_prior
                .map_or(35.0, |p| p.horizontal_sigma_m.max(10.0));
            let max_rad = (sigma_0 * 6.0).clamp(180.0, 1_500.0);
            let yaw_rad = anchor_prior.finite_heading_deg().map(f64::to_radians);
            self.state
                .set_cold_prior(ColdDeadReckonPrior::new(c_enu, sigma_0, max_rad, yaw_rad));
        }
    }
}

pub(super) fn build_anchor_prior(frame: &LiveFrameInput<'_>) -> AnchorPrior {
    let (lat_deg, lon_deg, prior_alt) = frame.filter_prior.map_or((None, None, None), |p| {
        let lat = p.center_geodetic.latitude_rad.to_degrees();
        let lon = p.center_geodetic.longitude_rad.to_degrees();
        let alt = p
            .center_geodetic
            .altitude_m
            .is_finite()
            .then_some(p.center_geodetic.altitude_m);
        (
            lat.is_finite().then_some(lat),
            lon.is_finite().then_some(lon),
            alt,
        )
    });
    let alt_msl_m = frame
        .baro_alt_msl_m
        .map(|(a, _)| a)
        .filter(|a| a.is_finite())
        .or(prior_alt);
    let heading_deg = frame
        .imu_attitude
        .and_then(|a| a.yaw_rad)
        .filter(|y| y.is_finite())
        .map(|y| y.to_degrees().rem_euclid(360.0));
    AnchorPrior {
        lat_deg,
        lon_deg,
        alt_msl_m,
        heading_deg,
    }
}

pub(super) fn attitude_override_deg(imu: Option<AttitudePrior>) -> Option<(f64, f64)> {
    let att = imu?;
    if !att.pitch_rad.is_finite() || !att.roll_rad.is_finite() || att.tilt_sigma_rad > 1.2 {
        return None;
    }
    Some((
        att.pitch_rad.to_degrees().clamp(-45.0, 89.5),
        att.roll_rad.to_degrees().clamp(-80.0, 80.0),
    ))
}

pub(super) fn estimate_lookahead_enu(
    surface: &Surface,
    state: &CorridorTracker,
    prior: AnchorPrior,
) -> Option<[f64; 2]> {
    if let Some(eye) = state.prev_eye {
        let mut fwd = [state.prev_vel[0] * 2.0, state.prev_vel[1] * 2.0];
        if let Some(rot) = state.prev_rot {
            let look = (eye[2].abs() * 1.4).clamp(200.0, 2_000.0);
            fwd[0] += rot[(2, 0)] * look;
            fwd[1] += rot[(2, 1)] * look;
        }
        return Some([eye[0] + fwd[0], eye[1] + fwd[1]]);
    }
    if let Some(cp) = state.cold_prior {
        return Some([
            cp.center_enu[0] + cp.cam_offset_m[0],
            cp.center_enu[1] + cp.cam_offset_m[1],
        ]);
    }
    let (lat, lon) = prior.lat_lon()?;
    let enu = surface.local(lat, lon, 0.0);
    if let Some(hdg_deg) = prior.finite_heading_deg() {
        let dem = surface.sample_dem(lat, lon).unwrap_or(0.0);
        let agl = prior
            .alt_msl_m
            .map_or(480.0, |alt| (alt - dem).clamp(25.0, 4_000.0));
        let stand_off = (agl * 1.4).clamp(150.0, 2_000.0);
        let rad = hdg_deg.to_radians();
        return Some([
            enu[0] + stand_off * rad.sin(),
            enu[1] + stand_off * rad.cos(),
        ]);
    }
    Some([enu[0], enu[1]])
}

pub(super) fn prior_odometry_hint(
    surface: &Surface,
    state: &CorridorTracker,
    prior: AnchorPrior,
    attitude_override: Option<(f64, f64)>,
    masker: &AmortizedMasker,
    k_render: &Matrix3<f64>,
) -> (Option<Matrix3<f64>>, f64) {
    let solved_agl = state.prev_eye.map(|eye| {
        let geo = surface.geodetic(eye);
        let dem = surface.sample_dem(geo[0], geo[1]).unwrap_or(0.0);
        (geo[2] - dem).clamp(20.0, 6_000.0)
    });
    let Some((lat, lon)) = prior.lat_lon() else {
        return (None, solved_agl.unwrap_or(480.0));
    };
    let dem = surface.sample_dem(lat, lon).unwrap_or(0.0);
    let agl = solved_agl.unwrap_or_else(|| {
        prior
            .alt_msl_m
            .map_or(480.0, |alt| (alt - dem).clamp(25.0, 6_000.0))
    });
    let (down_deg, roll_deg) = attitude_override
        .or_else(|| masker.horizon_attitude(k_render))
        .unwrap_or((55.0, 0.0));
    let rot = prior
        .finite_heading_deg()
        .map(|hdg| surface.camera_rotation(lat, lon, hdg, down_deg, roll_deg));
    (rot, agl)
}

pub(super) fn extract_dead_reckon_step(
    state: &CorridorTracker,
    k_render: &Matrix3<f64>,
    agl_est_m: f64,
) -> Option<DeadReckonStep> {
    let step = state.pose_steps.back()?;
    if state.last_flow_vectors < 6 && !step.has_world_pose {
        return None;
    }
    let yaw_delta_rad = step.rot_step[(1, 0)].atan2(step.rot_step[(0, 0)]);
    let delta_enu_m = if step.eye_delta != [0.0; 3] {
        step.eye_delta
    } else if let Some([du, dv]) = step.derotated_flow_px {
        let fx = k_render[(0, 0)].max(1.0);
        let fy = k_render[(1, 1)].max(1.0);
        [-agl_est_m * du / fx, agl_est_m * dv / fy, 0.0]
    } else {
        [0.0; 3]
    };
    Some(DeadReckonStep {
        delta_enu_m,
        yaw_delta_rad,
        agl_est_m,
    })
}
