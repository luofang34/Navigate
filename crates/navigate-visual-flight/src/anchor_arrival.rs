//! Delayed-measurement (`OOSM`) anchor propagation, track protection, and backup recovery.

use crate::{
    anchor::{
        FocalSweepEstimator, SeedPose, nadir_screen::rot_to_row_major,
        should_override_provisional_lock,
    },
    anchor_worker::AnchorOutcome,
    diagnostics::{AnchorArrivalRecord, AttemptRecord},
    tracks::{ARRIVAL_FRUSTUM_MARGIN_PX, merge_consistent_fg_tracks},
    types::{AnchorConfidence, FusionEligibility, TrackingMode},
};
use nalgebra::{Matrix3, Vector3};
use navigate_locate::corridor::{
    CorridorTracker, KinematicState, SessionAnchorReport, VehicleDynamicsProfile,
    plausibility::compute_fisher_geometry_covariance,
};
use navigate_match::ActiveTrackSet;
use navigate_visual::{LocalFrame, PoseCandidate, fusion_bridge::VisualFix, session::MapCell};
use std::collections::BTreeMap;

/// Active anchor confidence and prior center tracked by [`crate::FlightVisualEngine`].
#[derive(Clone, Copy, Debug)]
pub(crate) struct ActiveAnchorStatus {
    pub confidence: AnchorConfidence,
    pub holdout_median_px: f64,
    pub prior_center_enu: Option<[f64; 3]>,
    pub confirm_attempts: u8,
}

/// Retained backup of the most recent propagated map anchor (`<= 1.5 s`) for fast-roll recovery.
#[derive(Clone)]
pub(crate) struct RecentAnchorBackup {
    pub arrival_frame_seq: usize,
    pub arrival_ns: u64,
    pub pose: SeedPose,
    pub seeded_tracks: ActiveTrackSet,
    pub agl_m: f64,
    pub used_count: u8,
}

/// Admission metadata for one solved frame or anchor.
#[derive(Clone, Debug)]
pub(crate) struct FrameAdmission {
    pub tracking_mode: TrackingMode,
    pub fusion_eligibility: FusionEligibility,
    pub independent_ekf_fix: bool,
    pub map_cell: Option<[i64; 2]>,
    pub earlier_anchor_id: Option<String>,
    pub visual_fix: Option<VisualFix>,
    pub capture_lag_ns: u64,
}

impl FrameAdmission {
    pub(crate) const fn warm_track() -> Self {
        Self {
            tracking_mode: TrackingMode::WarmTrack,
            fusion_eligibility: FusionEligibility::CorrelatedTrack,
            independent_ekf_fix: false,
            map_cell: None,
            earlier_anchor_id: None,
            visual_fix: None,
            capture_lag_ns: 0,
        }
    }

    pub(crate) const fn acquiring_anchor() -> Self {
        Self {
            tracking_mode: TrackingMode::AcquiringAnchor,
            fusion_eligibility: FusionEligibility::AcquiringAnchor,
            independent_ekf_fix: false,
            map_cell: None,
            earlier_anchor_id: None,
            visual_fix: None,
            capture_lag_ns: 0,
        }
    }

    pub(crate) const fn unresolved() -> Self {
        Self {
            tracking_mode: TrackingMode::Unresolved,
            fusion_eligibility: FusionEligibility::Rejected,
            independent_ekf_fix: false,
            map_cell: None,
            earlier_anchor_id: None,
            visual_fix: None,
            capture_lag_ns: 0,
        }
    }

    pub(crate) fn from_session_report(report: SessionAnchorReport) -> Self {
        let fusion_eligibility = parse_eligibility(report.fusion_eligibility);
        Self {
            tracking_mode: TrackingMode::CorridorAnchor,
            fusion_eligibility,
            independent_ekf_fix: report.independent_ekf_fix,
            map_cell: report.map_cell,
            earlier_anchor_id: report.earlier_anchor_id,
            visual_fix: report.visual_fix,
            capture_lag_ns: report.capture_lag_ns,
        }
    }
}

pub(crate) fn parse_eligibility(label: &str) -> FusionEligibility {
    match label {
        "independent" | "relocated" => FusionEligibility::Independent,
        "shared_map_cell" | "untracked" => FusionEligibility::SharedMapCell,
        "correlated_track" => FusionEligibility::CorrelatedTrack,
        "acquiring_anchor" => FusionEligibility::AcquiringAnchor,
        _ => FusionEligibility::Rejected,
    }
}

pub(crate) fn classify_anchor_cell_fallback(
    visited: &mut BTreeMap<MapCell, String>,
    independent_fixes: &mut usize,
    catalog_sha256: &str,
    eye: [f64; 3],
    cell_m: f64,
    frame_label: &str,
    capture_lag_ns: u64,
) -> FrameAdmission {
    let cell = MapCell {
        map_manifest_sha256: catalog_sha256.to_owned(),
        east: (eye[0] / cell_m).floor() as i64,
        north: (eye[1] / cell_m).floor() as i64,
    };
    let coords = Some([cell.east, cell.north]);
    let (independent_ekf_fix, fusion_eligibility, earlier_anchor_id) =
        if let Some(earlier) = visited.get(&cell).cloned() {
            (false, FusionEligibility::SharedMapCell, Some(earlier))
        } else {
            visited.insert(cell, frame_label.to_owned());
            *independent_fixes = independent_fixes.saturating_add(1);
            (true, FusionEligibility::Independent, None)
        };
    FrameAdmission {
        tracking_mode: TrackingMode::CorridorAnchor,
        fusion_eligibility,
        independent_ekf_fix,
        map_cell: coords,
        earlier_anchor_id,
        visual_fix: None,
        capture_lag_ns,
    }
}

/// Staged propagated anchor ready for immediate use or warm-track fallback on the arrival frame.
pub(crate) struct ArrivedAnchor {
    pub seed_idx: usize,
    pub step_idx: usize,
    pub pose: SeedPose,
    pub candidate: PoseCandidate,
    pub seeded_tracks: ActiveTrackSet,
    pub admission: FrameAdmission,
    pub lg_layers: usize,
    pub polish_rays: usize,
    pub attempts: Vec<AttemptRecord>,
    pub fg_healthy_on_arrival: bool,
}

/// Mutable state borrowed when applying a completed [`AnchorOutcome`].
pub(crate) struct AnchorApplyCtx<'a> {
    pub state: &'a mut CorridorTracker,
    pub active_status: &'a mut Option<ActiveAnchorStatus>,
    pub recent_backup: &'a mut Option<RecentAnchorBackup>,
    pub independent_fixes: &'a mut usize,
    pub focal_estimator: &'a mut FocalSweepEstimator,
    pub catalog_sha256: &'a str,
    pub local_frame: LocalFrame,
    pub cell_m: f64,
    pub dynamics_profile: VehicleDynamicsProfile,
}

impl<'a> AnchorApplyCtx<'a> {
    /// Apply `outcome` at `(arrived_frame_seq, arrival_ns)` and return `(Option<ArrivedAnchor>, AnchorArrivalRecord)`.
    pub(crate) fn apply(
        &mut self,
        outcome: AnchorOutcome,
        arrived_at: Option<(usize, u64, &str)>,
    ) -> (Option<ArrivedAnchor>, AnchorArrivalRecord) {
        if let Some(locked) = outcome.locked_focal_px {
            self.focal_estimator.locked_focal_px = Some(locked);
        }
        let capture_ns = outcome.capture_stamp.acquired_at.as_nanos();
        let arrival_ns = arrived_at.map_or(capture_ns, |(_, ns, _)| ns.max(capture_ns));
        let capture_lag_ns = arrival_ns.saturating_sub(capture_ns);
        let propagated_frames =
            arrived_at.map_or(0, |(seq, _, _)| seq.saturating_sub(outcome.sent_frame_seq));
        let arrived_frame_seq = arrived_at.map(|(s, _, _)| s);
        let arrived_frame_label = arrived_at.map(|(_, _, l)| l.to_owned());

        let Some((mut tracks, (seed_idx, step_idx, ref cand))) = outcome.seeded.zip(outcome.best)
        else {
            if let Some(st) = self.active_status.as_mut()
                && st.confidence == AnchorConfidence::Provisional
            {
                st.confirm_attempts = st.confirm_attempts.saturating_add(1);
                self.state.frames_since_anchor = 0;
                self.state.time_since_anchor_s = 0.0;
            }
            let rec = AnchorArrivalRecord {
                sent_frame_seq: outcome.sent_frame_seq,
                sent_frame_label: outcome.frame_label,
                arrived_frame_seq,
                arrived_frame_label,
                propagated_frames,
                capture_lag_ns,
                accepted: false,
                confidence: None,
                independent_ekf_fix: false,
                fusion_eligibility: FusionEligibility::Rejected,
                map_cell: None,
                earlier_anchor_id: None,
                active_tracks: self.state.active_tracks.len(),
                merged_into_healthy_foreground: false,
                anchor_ms: outcome.elapsed_ms,
                attempts: outcome.attempts,
            };
            return (None, rec);
        };

        let conf = outcome.confidence.unwrap_or(AnchorConfidence::Confirmed);
        let (prop_eye, prop_rot) = if arrived_at.is_some() {
            self.state.propagate_pose_from_calibrated(
                outcome.sent_frame_seq,
                cand.eye,
                cand.rotation,
                &outcome.k_render,
                outcome.agl_m,
                &self.dynamics_profile,
            )
        } else {
            (cand.eye, cand.rotation)
        };
        let (mut q_vo_enu_m2, _path_m, turn_rad) = if arrived_at.is_some() {
            self.state.compute_vo_propagation_covariance(
                outcome.sent_frame_seq,
                cand.rotation,
                &outcome.k_render,
                outcome.agl_m,
                &self.dynamics_profile,
            )
        } else {
            (Matrix3::zeros(), 0.0, 0.0)
        };

        let fisher_pos = compute_fisher_geometry_covariance(cand)
            .fixed_view::<3, 3>(0, 0)
            .into_owned();
        if conf == AnchorConfidence::Provisional {
            let hx = (fisher_pos[(0, 0)] * 3.0).max(48.0);
            let hy = (fisher_pos[(1, 1)] * 3.0).max(48.0);
            q_vo_enu_m2 += Matrix3::from_diagonal(&Vector3::new(hx, hy, 4.0));
        }

        let override_provisional = self.active_status.is_some_and(|st| {
            st.confidence == AnchorConfidence::Provisional
                && should_override_provisional_lock(st.holdout_median_px, cand, conf)
        });
        let fg_healthy = arrived_at.is_some()
            && !override_provisional
            && !self.state.active_tracks.is_empty()
            && self.state.prev_eye.is_some()
            && self.state.time_since_accepted_pose_s <= 0.25
            && propagated_frames > 4;

        let intrinsics = [
            outcome.k_render[(0, 0)],
            outcome.k_render[(1, 1)],
            outcome.k_render[(0, 2)],
            outcome.k_render[(1, 2)],
        ];
        let margin_px = (ARRIVAL_FRUSTUM_MARGIN_PX + 4.0 * (propagated_frames as f64)).min(320.0);
        let mut fallback_tracks = tracks.clone();
        if arrived_at.is_some() && !fallback_tracks.is_empty() {
            fallback_tracks.retain_in_frustum(
                prop_eye,
                &rot_to_row_major(&prop_rot),
                intrinsics,
                outcome.render_size,
                margin_px,
            );
        }

        let mut merged_into_fg = false;
        if fg_healthy {
            let fg_eye = self.state.prev_eye.unwrap_or(prop_eye);
            let fg_rot = self.state.prev_rot.unwrap_or(prop_rot);
            let allow_merge = outcome.was_polished || cand.score.holdout_median_px <= 4.0;
            if allow_merge && !tracks.is_empty() {
                merged_into_fg = merge_consistent_fg_tracks(
                    &mut self.state.active_tracks,
                    &tracks,
                    (prop_eye, &prop_rot),
                    (fg_eye, &fg_rot),
                    &outcome.k_render,
                    outcome.render_size,
                );
            }
            if conf == AnchorConfidence::Confirmed {
                *self.active_status = Some(ActiveAnchorStatus {
                    confidence: AnchorConfidence::Confirmed,
                    holdout_median_px: cand.score.holdout_median_px,
                    prior_center_enu: outcome.query_prior_enu,
                    confirm_attempts: 0,
                });
            } else if let Some(st) = self.active_status.as_mut()
                && st.confidence == AnchorConfidence::Provisional
            {
                st.confirm_attempts = st.confirm_attempts.saturating_add(1);
            }
        } else {
            tracks = fallback_tracks.clone();
            self.state.active_tracks = tracks;
            if arrived_at.is_some() {
                clamp_tracker_velocity_on_arrival(self.state, &self.dynamics_profile);
                self.state.prev_eye = Some(prop_eye);
                self.state.prev_rot = Some(prop_rot);
                self.state.vo_rot = Some(prop_rot);
                self.state.cold_prior = None;
                self.state.time_since_accepted_pose_s = 0.0;
                let arrival_cov = fisher_pos + q_vo_enu_m2 + Matrix3::identity() * 4.0;
                let arrival_att_sigma = (0.01 + 0.15 * turn_rad).min(0.35);
                if let Some(kin) = self.state.kinematic.as_mut() {
                    kin.eye_enu = prop_eye;
                    kin.rot_world_to_cam = prop_rot;
                    kin.vel_enu_mps = self.state.prev_vel;
                    kin.pos_cov_enu_m2 = arrival_cov;
                    kin.att_sigma_rad = arrival_att_sigma;
                } else {
                    let mut kin = KinematicState::new(prop_eye, prop_rot);
                    kin.vel_enu_mps = self.state.prev_vel;
                    kin.pos_cov_enu_m2 = arrival_cov;
                    kin.att_sigma_rad = arrival_att_sigma;
                    self.state.kinematic = Some(kin);
                }
                if let (Some(session), Some(kin)) = (
                    self.state.corridor_session.as_mut(),
                    self.state.kinematic.clone(),
                ) {
                    session.dead_reckoner.kinematic = kin;
                }
            }
            let next_attempts = match (self.active_status.as_ref(), conf) {
                (Some(prev), AnchorConfidence::Provisional)
                    if prev.confidence == AnchorConfidence::Provisional =>
                {
                    prev.confirm_attempts.saturating_add(1)
                }
                _ => 0,
            };
            *self.active_status = Some(ActiveAnchorStatus {
                confidence: conf,
                holdout_median_px: cand.score.holdout_median_px,
                prior_center_enu: outcome.query_prior_enu,
                confirm_attempts: next_attempts,
            });
        }
        self.state.frames_since_anchor = 0;
        self.state.time_since_anchor_s = 0.0;

        let saved_fg_kin = fg_healthy.then(|| self.state.kinematic.clone()).flatten();
        let mut prop_cand = cand.clone();
        prop_cand.eye = prop_eye;
        prop_cand.rotation = prop_rot;

        let adm = if let Ok(report) = self.state.admit_propagated_anchor(
            &prop_cand,
            self.local_frame,
            (
                self.catalog_sha256,
                self.cell_m,
                &outcome.frame_label,
                "corridor-background-anchor",
            ),
            (capture_ns, arrival_ns),
            &q_vo_enu_m2,
        ) {
            if report.independent_ekf_fix {
                *self.independent_fixes = self.independent_fixes.saturating_add(1);
            }
            FrameAdmission::from_session_report(report)
        } else {
            classify_anchor_cell_fallback(
                &mut self.state.visited_cells,
                self.independent_fixes,
                self.catalog_sha256,
                prop_cand.eye,
                self.cell_m,
                &outcome.frame_label,
                capture_lag_ns,
            )
        };
        if let (Some(session), Some(kin)) = (self.state.corridor_session.as_mut(), saved_fg_kin) {
            session.dead_reckoner.kinematic = kin;
        }

        if let Some((arr_seq, arr_ns, _)) = arrived_at {
            *self.recent_backup = Some(RecentAnchorBackup {
                arrival_frame_seq: arr_seq,
                arrival_ns: arr_ns,
                pose: (prop_eye, prop_rot),
                seeded_tracks: fallback_tracks.clone(),
                agl_m: outcome.agl_m,
                used_count: 0,
            });
        }

        let rec = AnchorArrivalRecord {
            sent_frame_seq: outcome.sent_frame_seq,
            sent_frame_label: outcome.frame_label,
            arrived_frame_seq,
            arrived_frame_label,
            propagated_frames,
            capture_lag_ns,
            accepted: true,
            confidence: Some(conf),
            independent_ekf_fix: adm.independent_ekf_fix,
            fusion_eligibility: adm.fusion_eligibility,
            map_cell: adm.map_cell,
            earlier_anchor_id: adm.earlier_anchor_id.clone(),
            active_tracks: self.state.active_tracks.len(),
            merged_into_healthy_foreground: merged_into_fg,
            anchor_ms: outcome.elapsed_ms,
            attempts: outcome.attempts.clone(),
        };

        let staged = arrived_at.map(|_| ArrivedAnchor {
            seed_idx,
            step_idx,
            pose: (prop_eye, prop_rot),
            candidate: prop_cand,
            seeded_tracks: fallback_tracks,
            admission: adm,
            lg_layers: outcome.lg_layers,
            polish_rays: outcome.polish_rays,
            attempts: outcome.attempts,
            fg_healthy_on_arrival: fg_healthy,
        });
        (staged, rec)
    }
}

fn clamp_tracker_velocity_on_arrival(
    state: &mut CorridorTracker,
    dynamics_profile: &VehicleDynamicsProfile,
) {
    if state.prev_eye.is_none() || state.time_since_accepted_pose_s > 2.0 {
        state.prev_vel = [0.0, 0.0, 0.0];
    } else {
        let [vx, vy, vz] = state.prev_vel;
        let horiz = vx.hypot(vy);
        let max_h = dynamics_profile.max_speed_mps.max(0.0);
        let scale_h = if horiz > max_h && horiz > 0.0 {
            max_h / horiz
        } else {
            1.0
        };
        let max_v = dynamics_profile.max_vertical_speed_mps.max(0.0);
        state.prev_vel = [vx * scale_h, vy * scale_h, vz.clamp(-max_v, max_v)];
    }
}
