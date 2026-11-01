//! Session lifecycle and navigation filter bridge methods on [`CorridorTracker`].

use super::CorridorTracker;
use crate::corridor::{
    kinematics::{MAX_DELTA_T_S, MIN_DELTA_T_S},
    session::{FilterIntegritySnapshot, SessionAnchorReport, VisualCorridorSession},
};
use navigate_visual::{CameraModel, LocalFrame, holdout_pnp::PoseCandidate};
use navigate_visual_session::MapCell;

impl CorridorTracker {
    /// Create a tracker and return an error if session initialization fails.
    ///
    /// # Errors
    /// Returns an error string when [`VisualCorridorSession::new`] fails.
    pub fn try_new() -> Result<Self, String> {
        let session = VisualCorridorSession::new()?;
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
        if self.corridor_session.is_none() {
            self.corridor_session = Some(VisualCorridorSession::new()?);
        }
        if let Some(session) = self.corridor_session.as_mut() {
            session.configure_corridor(intrinsics, cell_m)?;
        }
        self.visited_cells.clear();
        self.monotonic_time_ns = 1_000_000_000;
        Ok(())
    }

    /// Register one processed camera frame in [`VisualSession`](navigate_visual_session::VisualSession).
    ///
    /// # Errors
    /// Returns an error string when frame registration fails.
    pub fn observe_frame(
        &mut self,
        time_s: Option<f64>,
        dt_s: f64,
        raw_sha256: &str,
        case_id: &str,
    ) -> Result<u64, String> {
        let dt_ns = (dt_s.clamp(MIN_DELTA_T_S, MAX_DELTA_T_S) * 1_000_000_000.0).round() as u64;
        let candidate_ns = time_s
            .filter(|t| t.is_finite() && *t >= 0.0)
            .map(|t| 1_000_000_000_u64.saturating_add((t * 1_000_000_000.0).round() as u64));
        let next_ns = match candidate_ns {
            Some(c) if c > self.monotonic_time_ns => c,
            _ => self.monotonic_time_ns.saturating_add(dt_ns.max(1_000)),
        };
        self.monotonic_time_ns = next_ns;

        if self.corridor_session.is_none() {
            self.corridor_session = Some(VisualCorridorSession::new()?);
        }
        if let Some(session) = self.corridor_session.as_mut() {
            let seq = session.next_seq;
            let sha = session.unique_frame_digest(raw_sha256, case_id, seq);
            session.register_frame_with_digest(next_ns, &sha)?;
        }
        Ok(next_ns)
    }

    /// Record a warm-tracked relative motion step in the visual session.
    pub fn record_warm_track(&mut self, cand: &PoseCandidate, catalog_sha256: &str) {
        if let Some(session) = self.corridor_session.as_mut() {
            session.record_warm_track_step(cand.eye, &cand.rotation, catalog_sha256);
        }
    }

    /// Submit an accepted map anchor through `VisualSession`, `VisualFixSource`, and `NavigationFilter`.
    ///
    /// # Errors
    /// Returns an error string when anchor submission fails.
    pub fn admit_anchor(
        &mut self,
        cand: &PoseCandidate,
        local_frame: LocalFrame,
        catalog_sha256: &str,
        cell_m: f64,
        case_id: &str,
        backend: &str,
    ) -> Result<SessionAnchorReport, String> {
        if self.corridor_session.is_none() {
            self.corridor_session = Some(VisualCorridorSession::new()?);
        }
        let cell = MapCell {
            map_manifest_sha256: catalog_sha256.to_owned(),
            east: (cand.eye[0] / cell_m).floor() as i64,
            north: (cand.eye[1] / cell_m).floor() as i64,
        };
        self.visited_cells
            .entry(cell)
            .or_insert_with(|| case_id.to_owned());

        let Some(session) = self.corridor_session.as_mut() else {
            return Err("corridor session unavailable".into());
        };
        session.record_anchor_candidate(cand, local_frame, catalog_sha256, cell_m, case_id, backend)
    }

    /// Tick the navigation filter and return the current integrity and revision snapshot.
    pub fn filter_snapshot(&mut self, local_frame: Option<LocalFrame>) -> FilterIntegritySnapshot {
        let t_ns = self.monotonic_time_ns.max(1);
        let Some(session) = self.corridor_session.as_mut() else {
            return FilterIntegritySnapshot::default();
        };
        session.refresh_filter_snapshot(t_ns, local_frame)
    }
}
