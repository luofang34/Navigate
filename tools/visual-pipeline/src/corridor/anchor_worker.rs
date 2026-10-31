//! Background map-anchor acquisition for `corridor-track --background-refresh`.
//!
//! A periodic re-anchor runs on a named worker thread while the warm tracker
//! keeps solving frames against the current [`ActiveTrackSet`]. The anchor
//! result belongs to the frame that was sent. On arrival it replaces the active
//! tracks and counts as an EKF fix for that frame, but it does not change the
//! motion state, because the pose is older than the current frame.

use super::{
    CorridorTrackOptions,
    admission::{classify_anchor_cell, seed_tracks},
    anchor::acquire_corridor_anchor,
    background::BackgroundWorker,
    latency::StageLatencies,
    motion::TrackState,
};
use crate::{PipelineError, pose::PoseCandidate, surface::Surface, testset::PhotoCase};
use nalgebra::Matrix3;
use navigate_match::{
    ActiveTrackSet, CorridorLandmarkBank, HadamardProjector, LighterGlue, PreparedKashinQuery,
    SparseFeatures, XFeat,
};
use serde_json::{Value, json};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

/// Time that a finished run waits for an in-flight anchor job to stop.
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);

/// Read-only models and maps that the anchor worker thread owns.
pub(super) struct AnchorContext {
    pub(super) surface: Arc<Surface>,
    pub(super) bank: Arc<CorridorLandmarkBank>,
    pub(super) projector: HadamardProjector,
    pub(super) xfeat: XFeat,
    pub(super) lighterglue: Option<LighterGlue>,
    pub(super) opts: CorridorTrackOptions,
}

/// One query frame for map-anchor acquisition.
pub(super) struct AnchorJob {
    pub(super) case: PhotoCase,
    pub(super) qry_feat: SparseFeatures,
    pub(super) prepared: PreparedKashinQuery,
    pub(super) k: Matrix3<f64>,
    pub(super) k_render: Matrix3<f64>,
    pub(super) render_size: [u32; 2],
    pub(super) pred_pose: Option<([f64; 3], Matrix3<f64>)>,
    pub(super) live_attitude: Option<(f64, f64)>,
}

/// Result of one [`AnchorJob`].
pub(super) struct AnchorOutcome {
    case_id: String,
    best: Option<PoseCandidate>,
    seeded: Option<ActiveTrackSet>,
    attempts: usize,
    elapsed_ms: f64,
}

impl AnchorContext {
    fn run(&self, job: &AnchorJob) -> Result<AnchorOutcome, PipelineError> {
        let t0 = Instant::now();
        let mut attempts = Vec::new();
        let acq = acquire_corridor_anchor(
            (&*self.surface, &*self.bank, &self.projector),
            (&self.xfeat, self.lighterglue.as_ref()),
            (&job.case, &self.opts),
            (&job.qry_feat, &job.prepared),
            (&job.k, &job.k_render, job.render_size),
            (job.pred_pose, job.live_attitude),
            &mut attempts,
        )?;
        let best = acq.best.map(|(_, _, cand)| cand);
        let seeded = best
            .as_ref()
            .filter(|c| c.score.geometry_accepted)
            .map(|cand| {
                seed_tracks(
                    &self.surface,
                    &job.prepared,
                    &job.qry_feat,
                    cand,
                    &acq.best_view,
                )
            });
        Ok(AnchorOutcome {
            case_id: job.case.id.clone(),
            best,
            seeded,
            attempts: attempts.len(),
            elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

/// Anchor worker thread with at most one job in flight.
pub(super) struct BackgroundAnchor {
    worker: BackgroundWorker<AnchorJob, Result<AnchorOutcome, PipelineError>>,
    in_flight: bool,
}

impl BackgroundAnchor {
    pub(super) fn spawn(context: AnchorContext) -> Result<Self, PipelineError> {
        let worker =
            BackgroundWorker::spawn("corridor-map-anchor", SHUTDOWN_TIMEOUT, move |job| {
                context.run(&job)
            })?;
        Ok(Self {
            worker,
            in_flight: false,
        })
    }

    /// Send `job` unless a job already runs.
    pub(super) fn submit_if_idle(
        &mut self,
        job: impl FnOnce() -> AnchorJob,
    ) -> Result<(), PipelineError> {
        if !self.in_flight {
            self.worker.submit(job())?;
            self.in_flight = true;
        }
        Ok(())
    }

    /// Return the finished job without waiting.
    pub(super) fn poll(&mut self) -> Result<Option<AnchorOutcome>, PipelineError> {
        if !self.in_flight {
            return Ok(None);
        }
        let outcome = self.worker.try_recv()?.transpose()?;
        self.in_flight = outcome.is_none();
        Ok(outcome)
    }

    /// Wait for the job that runs, if any.
    pub(super) fn wait(&mut self) -> Result<Option<AnchorOutcome>, PipelineError> {
        if !self.in_flight {
            return Ok(None);
        }
        self.in_flight = false;
        self.worker.recv_blocking()?.map(Some)
    }
}

/// Bookkeeping that an arriving anchor result updates.
pub(super) struct AnchorArrival<'a> {
    pub(super) state: &'a mut TrackState,
    pub(super) independent_fixes: &'a mut usize,
    pub(super) latencies: &'a mut StageLatencies,
    pub(super) results: &'a mut Vec<Value>,
    pub(super) catalog_sha256: &'a str,
    pub(super) cell_m: f64,
}

impl AnchorArrival<'_> {
    /// Apply `outcome` on frame `arrived_at` (`None` after the last frame).
    pub(super) fn apply(&mut self, outcome: AnchorOutcome, arrived_at: Option<&str>) {
        self.latencies.anchor_ms.push(outcome.elapsed_ms);
        let mut record = json!({
            "anchor_result_for": outcome.case_id,
            "arrived_at": arrived_at,
            "accepted": outcome.seeded.is_some(),
            "attempts": outcome.attempts,
            "anchor_ms": outcome.elapsed_ms,
        });
        if let (Some(tracks), Some(cand)) = (outcome.seeded, outcome.best.as_ref()) {
            self.state.active_tracks = tracks;
            self.state.frames_since_anchor = 0;
            let adm = classify_anchor_cell(
                &mut self.state.visited_cells,
                self.independent_fixes,
                self.catalog_sha256,
                cand.eye,
                self.cell_m,
                &outcome.case_id,
            );
            if let Some(map) = record.as_object_mut() {
                map.insert("independent_ekf_fix".into(), adm.independent_ekf_fix.into());
                map.insert("fusion_eligibility".into(), adm.fusion_eligibility.into());
                map.insert("map_cell".into(), json!(adm.map_cell));
                map.insert(
                    "active_tracks".into(),
                    self.state.active_tracks.len().into(),
                );
            }
        }
        self.results.push(record);
    }
}

#[cfg(test)]
mod tests;
