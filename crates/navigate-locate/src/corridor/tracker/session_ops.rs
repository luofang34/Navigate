//! Session lifecycle and navigation filter bridge methods on [`CorridorTracker`].

use super::CorridorTracker;
use crate::corridor::session::{
    FilterIntegritySnapshot, SessionAnchorReport, VisualCorridorSession,
};
use nalgebra::Matrix3;
use navigate_contract::MonotonicNanos;
use navigate_geodesy::BodyEllipsoid;
use navigate_visual::{
    CameraModel, LocalFrame, fusion_bridge::VisualAttitudeBudget, holdout_pnp::PoseCandidate,
    session::MapCell,
};

impl CorridorTracker {
    /// Create a tracker configured with explicit `intrinsics` and return an error on failure.
    ///
    /// # Errors
    /// Returns an error string when [`VisualCorridorSession::new`] fails.
    pub fn try_new(intrinsics: CameraModel) -> Result<Self, String> {
        let session = VisualCorridorSession::new(intrinsics)?;
        let mut tracker = Self::new();
        tracker.corridor_session = Some(session);
        Ok(tracker)
    }

    /// Reconfigure session camera intrinsics and map-cell size before frame processing.
    ///
    /// # Errors
    /// Returns an error string when session configuration fails.
    pub fn configure_session(
        &mut self,
        intrinsics: CameraModel,
        cell_m: f64,
    ) -> Result<(), String> {
        self.configure_session_for_body(intrinsics, cell_m, BodyEllipsoid::WGS84, None)
    }

    /// Reconfigure session camera intrinsics, map-cell size, body ellipsoid, and attitude budget.
    ///
    /// # Errors
    /// Returns an error string when session configuration fails.
    pub fn configure_session_for_body(
        &mut self,
        intrinsics: CameraModel,
        cell_m: f64,
        ellipsoid: BodyEllipsoid,
        attitude_budget: Option<VisualAttitudeBudget>,
    ) -> Result<(), String> {
        if let Some(session) = self.corridor_session.as_mut() {
            session.configure_corridor_for_body(intrinsics, cell_m, ellipsoid, attitude_budget)?;
        } else {
            let mut session = VisualCorridorSession::new(intrinsics)?;
            session.configure_corridor_for_body(intrinsics, cell_m, ellipsoid, attitude_budget)?;
            self.corridor_session = Some(session);
        }
        self.visited_cells.clear();
        self.last_monotonic_ns = None;
        Ok(())
    }

    /// Register one processed camera frame at caller-supplied monotonic `stamp_ns`.
    ///
    /// # Errors
    /// Returns an error string when no session is configured, `stamp_ns` is zero,
    /// or `stamp_ns` is not strictly increasing.
    pub fn observe_frame_monotonic(
        &mut self,
        stamp_ns: MonotonicNanos,
        raw_sha256: &str,
        case_id: &str,
    ) -> Result<u64, String> {
        let next_ns = stamp_ns.as_nanos();
        if next_ns == 0 {
            return Err("monotonic frame timestamp must be non-zero".into());
        }
        if let Some(prev) = self.last_monotonic_ns
            && stamp_ns <= prev
        {
            return Err(format!(
                "non-monotonic frame timestamp: received {} <= previous {}",
                next_ns,
                prev.as_nanos()
            ));
        }
        let session = self
            .corridor_session
            .as_mut()
            .ok_or_else(|| "corridor session not configured with camera intrinsics".to_owned())?;
        let seq = session.next_seq;
        let sha = session.unique_frame_digest(raw_sha256, case_id, seq);
        session.register_frame_with_digest(next_ns, &sha)?;
        self.last_monotonic_ns = Some(stamp_ns);
        Ok(next_ns)
    }

    /// Record a warm-tracked relative motion step in the visual session.
    pub fn record_warm_track(&mut self, cand: &PoseCandidate, catalog_sha256: &str) {
        if let Some(session) = self.corridor_session.as_mut() {
            session.record_warm_track_step(cand.eye, &cand.rotation, catalog_sha256);
        }
    }

    /// Submit a zero-lag accepted map anchor on the current frame.
    ///
    /// # Errors
    /// Returns an error string when no session is configured or anchor submission fails.
    pub fn admit_anchor(
        &mut self,
        cand: &PoseCandidate,
        local_frame: LocalFrame,
        catalog_sha256: &str,
        cell_m: f64,
        case_id: &str,
        backend: &str,
    ) -> Result<SessionAnchorReport, String> {
        let arrival_ns = self
            .last_monotonic_ns
            .map(MonotonicNanos::as_nanos)
            .ok_or_else(|| "cannot admit anchor before observing a monotonic frame".to_owned())?;
        self.admit_propagated_anchor(
            cand,
            local_frame,
            (catalog_sha256, cell_m, case_id, backend),
            (arrival_ns, arrival_ns),
            &Matrix3::zeros(),
        )
    }

    /// Submit a delayed background anchor captured at `capture_ns` and VO-propagated to `arrival_ns`.
    ///
    /// # Errors
    /// Returns an error string when no session is configured or anchor submission fails.
    pub fn admit_propagated_anchor(
        &mut self,
        cand: &PoseCandidate,
        local_frame: LocalFrame,
        (catalog_sha256, cell_m, case_id, backend): (&str, f64, &str, &str),
        (capture_ns, arrival_ns): (u64, u64),
        q_vo_enu_m2: &Matrix3<f64>,
    ) -> Result<SessionAnchorReport, String> {
        let cell = MapCell {
            map_manifest_sha256: catalog_sha256.to_owned(),
            east: (cand.eye[0] / cell_m).floor() as i64,
            north: (cand.eye[1] / cell_m).floor() as i64,
        };
        self.visited_cells
            .entry(cell)
            .or_insert_with(|| case_id.to_owned());

        let session = self
            .corridor_session
            .as_mut()
            .ok_or_else(|| "corridor session unavailable".to_owned())?;
        session.record_propagated_anchor_candidate(
            cand,
            local_frame,
            (catalog_sha256, cell_m, case_id, backend),
            (capture_ns, arrival_ns),
            q_vo_enu_m2,
        )
    }

    /// Tick the navigation filter and return the current integrity and revision snapshot.
    pub fn filter_snapshot(&mut self, local_frame: Option<LocalFrame>) -> FilterIntegritySnapshot {
        let Some(t_ns) = self.last_monotonic_ns.map(MonotonicNanos::as_nanos) else {
            return FilterIntegritySnapshot::default();
        };
        let Some(session) = self.corridor_session.as_mut() else {
            return FilterIntegritySnapshot::default();
        };
        session.refresh_filter_snapshot(t_ns, local_frame)
    }

    /// Clear inter-frame pose, landmark tracks, and kinematic state across a flight or time jump.
    pub fn reset_discontinuous_track(&mut self) {
        self.active_tracks = navigate_match::ActiveTrackSet::empty();
        self.prev_eye = None;
        self.prev_rot = None;
        self.prev_vel = [0.0; 3];
        self.prev_rot_step = Matrix3::identity();
        self.prev_prepared = None;
        self.vo_rot = None;
        self.cold_prior = None;
        self.pose_steps.clear();
        self.frames_since_anchor = 0;
        self.time_since_anchor_s = 0.0;
        self.time_since_accepted_pose_s = f64::INFINITY;
        self.kinematic = None;
        if let Some(session) = self.corridor_session.as_mut() {
            session.advance_epoch();
            session.dead_reckoner.clear_filter_solution();
        }
    }

    /// Snap tracker pose on an anchor-arrival frame without inferring velocity from the residual.
    pub fn snap_pose_preserving_velocity(
        &mut self,
        cand: &PoseCandidate,
        prepared: navigate_match::PreparedKashinQuery,
    ) {
        self.prev_eye = Some(cand.eye);
        self.prev_rot = Some(cand.rotation);
        self.vo_rot = Some(cand.rotation);
        self.cold_prior = None;
        self.prev_prepared = Some(prepared);
        self.time_since_accepted_pose_s = 0.0;
        let fisher_pos = crate::corridor::plausibility::compute_fisher_geometry_covariance(cand)
            .fixed_view::<3, 3>(0, 0)
            .into_owned();
        let tightened_cov = fisher_pos + Matrix3::identity() * 2.25;
        if let Some(kin) = self.kinematic.as_mut() {
            kin.eye_enu = cand.eye;
            kin.rot_world_to_cam = cand.rotation;
            kin.pos_cov_enu_m2 = tightened_cov;
            kin.att_sigma_rad = 0.01;
        } else {
            let mut kin = crate::corridor::kinematics::KinematicState::new(cand.eye, cand.rotation);
            kin.pos_cov_enu_m2 = tightened_cov;
            self.kinematic = Some(kin);
        }
        if let (Some(session), Some(kin)) = (self.corridor_session.as_mut(), self.kinematic.clone())
        {
            session.dead_reckoner.kinematic = kin;
        }
    }
}
