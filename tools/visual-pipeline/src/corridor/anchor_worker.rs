//! Background map-anchor acquisition and odometry-propagated arrival for `corridor-track`.

use super::{
    CorridorTrackOptions,
    admission::{classify_anchor_cell, seed_tracks},
    anchor::acquire_corridor_anchor,
    background::BackgroundWorker,
    latency::StageLatencies,
    motion::{TrackState, propagate_pose_from},
};
use crate::{
    PipelineError,
    pose::PoseCandidate,
    realtime::helpers::rot_to_row_major,
    surface::gpu::GpuQueueRole,
    surface::{Surface, SurfaceGpuRenderer},
    testset::PhotoCase,
    xfeat_runtime::LoadedXFeat,
};
use nalgebra::Matrix3;
use navigate_match::{
    ActiveTrackSet, CorridorTileStore, HadamardProjector, LighterGlue, PreparedKashinQuery,
    SparseFeatures,
};
use serde_json::{Value, json};
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
const ARRIVAL_FRUSTUM_MARGIN_PX: f64 = 64.0;

/// Read-only models and maps that the anchor worker thread owns.
pub(super) struct AnchorContext {
    pub(super) surface: Arc<Surface>,
    pub(super) store: Arc<CorridorTileStore>,
    pub(super) projector: HadamardProjector,
    pub(super) xfeat: LoadedXFeat,
    pub(super) lighterglue: Option<LighterGlue>,
    pub(super) opts: CorridorTrackOptions,
}

/// One query frame captured for background map-anchor acquisition.
pub(super) struct AnchorJob {
    pub(super) sent_frame_seq: usize,
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
    pub(super) sent_frame_seq: usize,
    pub(super) case_id: String,
    pub(super) k_render: Matrix3<f64>,
    pub(super) render_size: [u32; 2],
    pub(super) best: Option<PoseCandidate>,
    pub(super) seeded: Option<ActiveTrackSet>,
    pub(super) attempts: usize,
    pub(super) elapsed_ms: f64,
}

impl AnchorContext {
    fn run(
        &self,
        gpu_renderer: &Mutex<Option<SurfaceGpuRenderer>>,
        job: &AnchorJob,
    ) -> Result<AnchorOutcome, PipelineError> {
        let t0 = Instant::now();
        let mut attempts = Vec::new();
        let acq = acquire_corridor_anchor(
            (&*self.surface, &*self.store, &self.projector),
            Some(gpu_renderer),
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
            sent_frame_seq: job.sent_frame_seq,
            case_id: job.case.id.clone(),
            k_render: job.k_render,
            render_size: job.render_size,
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
        let init_gpu = context
            .opts
            .polish_anchors
            .then(|| {
                let k = Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
                SurfaceGpuRenderer::try_new_with_role(
                    &context.surface,
                    &k,
                    [800, 600],
                    GpuQueueRole::BackgroundAnchor,
                )
            })
            .flatten();
        let gpu_renderer: Mutex<Option<SurfaceGpuRenderer>> = Mutex::new(init_gpu);
        let worker =
            BackgroundWorker::spawn("corridor-map-anchor", SHUTDOWN_TIMEOUT, move |job| {
                context.run(&gpu_renderer, &job)
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
    ) -> Result<bool, PipelineError> {
        if !self.in_flight {
            self.worker.submit(job())?;
            self.in_flight = true;
            return Ok(true);
        }
        Ok(false)
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

    /// Drain an in-flight job after the frame loop completes.
    pub(super) fn wait_for_shutdown(&mut self) -> Result<Option<AnchorOutcome>, PipelineError> {
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

impl<'a> AnchorArrival<'a> {
    /// Apply `outcome` on frame `arrived_at` (`None` after the last frame) and return the propagated pose.
    pub(super) fn apply(
        &mut self,
        outcome: AnchorOutcome,
        arrived_at: Option<&str>,
    ) -> Option<([f64; 3], Matrix3<f64>)> {
        self.latencies.anchor_ms.push(outcome.elapsed_ms);
        let propagated_frames = if arrived_at.is_some() {
            self.state.frame_seq.saturating_sub(outcome.sent_frame_seq)
        } else {
            0
        };
        let mut record = json!({
            "anchor_result_for": outcome.case_id,
            "arrived_at": arrived_at,
            "propagated_frames": propagated_frames,
            "accepted": outcome.seeded.is_some(),
            "attempts": outcome.attempts,
            "anchor_ms": outcome.elapsed_ms,
        });
        let mut propagated_pose = None;
        if let (Some(mut tracks), Some(cand)) = (outcome.seeded, outcome.best.as_ref()) {
            let (prop_eye, prop_rot) = if arrived_at.is_some() {
                propagate_pose_from(self.state, outcome.sent_frame_seq, cand.eye, cand.rotation)
            } else {
                (cand.eye, cand.rotation)
            };
            if arrived_at.is_some() && !tracks.is_empty() {
                let intrinsics = [
                    outcome.k_render[(0, 0)],
                    outcome.k_render[(1, 1)],
                    outcome.k_render[(0, 2)],
                    outcome.k_render[(1, 2)],
                ];
                tracks.retain_in_frustum(
                    prop_eye,
                    &rot_to_row_major(&prop_rot),
                    intrinsics,
                    outcome.render_size,
                    ARRIVAL_FRUSTUM_MARGIN_PX,
                );
            }
            self.state.active_tracks = tracks;
            self.state.frames_since_anchor = 0;
            if arrived_at.is_some() {
                if let Some(prev) = self.state.prev_eye {
                    let dt = self.state.time_since_anchor_s.clamp(1e-4, 60.0);
                    self.state.prev_vel = [
                        (prop_eye[0] - prev[0]) / dt,
                        (prop_eye[1] - prev[1]) / dt,
                        (prop_eye[2] - prev[2]) / dt,
                    ];
                } else {
                    self.state.prev_vel = [0.0, 0.0, 0.0];
                }
                self.state.prev_eye = Some(prop_eye);
                self.state.prev_rot = Some(prop_rot);
                self.state.vo_rot = Some(prop_rot);
                if let Some(kin) = self.state.kinematic.as_mut() {
                    kin.eye_enu = prop_eye;
                    kin.rot_world_to_cam = prop_rot;
                    kin.vel_enu_mps = self.state.prev_vel;
                }
            }
            propagated_pose = Some((prop_eye, prop_rot));
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
        propagated_pose
    }

    /// Apply an anchor outcome to the track state and record telemetry.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn apply_outcome(
        state: &'a mut TrackState,
        independent_fixes: &'a mut usize,
        latencies: &'a mut StageLatencies,
        results: &'a mut Vec<Value>,
        catalog_sha256: &'a str,
        cell_m: f64,
        outcome: AnchorOutcome,
        arrived_at: Option<&str>,
    ) -> Option<([f64; 3], Matrix3<f64>)> {
        Self {
            state,
            independent_fixes,
            latencies,
            results,
            catalog_sha256,
            cell_m,
        }
        .apply(outcome, arrived_at)
    }
}

pub use navigate_locate::corridor::AnchorWorkerState;

#[cfg(test)]
mod tests;
