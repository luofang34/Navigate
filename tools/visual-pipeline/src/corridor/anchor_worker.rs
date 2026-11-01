//! Background map-anchor acquisition and odometry-propagated arrival for `corridor-track`.

use super::{
    CorridorTrackOptions,
    admission::{FusionAdmission, classify_anchor_cell, seed_tracks},
    anchor::{
        AnchorQueryAux, acquire_corridor_anchor, is_plausible_anchor_candidate,
        multitier::SearchTierMatcher,
    },
    background::BackgroundWorker,
    kinematics::VehicleDynamicsProfile,
    latency::StageLatencies,
    motion::{TrackState, propagate_pose_from},
};
use crate::{
    PipelineError,
    pose::PoseCandidate,
    realtime::helpers::rot_to_row_major,
    stages::focal::FocalLengthEstimator,
    surface::gpu::GpuQueueRole,
    surface::{Surface, SurfaceGpuRenderer},
    testset::PhotoCase,
    xfeat_runtime::LoadedXFeat,
};
use image::GrayImage;
use nalgebra::Matrix3;
use navigate_match::{
    ActiveTrackSet, CorridorTileStore, HadamardProjector, LighterGlue, PreparedKashinQuery,
    SparseFeatures,
};
use navigate_visual::LocalFrame;
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
    pub(super) search_tier_matcher: Option<Arc<SearchTierMatcher>>,
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
    pub(super) mask_small: Option<GrayImage>,
    pub(super) focal_locked: bool,
}

/// Result of one [`AnchorJob`].
pub(super) struct AnchorOutcome {
    pub(super) sent_frame_seq: usize,
    pub(super) case_id: String,
    pub(super) k_render: Matrix3<f64>,
    pub(super) render_size: [u32; 2],
    pub(super) best: Option<PoseCandidate>,
    pub(super) seeded: Option<ActiveTrackSet>,
    pub(super) locked_focal_px: Option<f64>,
    pub(super) attempts: Vec<Value>,
    pub(super) elapsed_ms: f64,
}

/// Propagated candidate pose, admission telemetry, and attempt log from an arrived map anchor.
pub(super) struct ArrivedAnchor {
    pub(super) pose: ([f64; 3], Matrix3<f64>),
    pub(super) candidate: PoseCandidate,
    pub(super) admission: FusionAdmission,
    pub(super) attempts: Vec<Value>,
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
            AnchorQueryAux {
                pred_pose: job.pred_pose,
                live_attitude: job.live_attitude,
                mask_small: job.mask_small.as_ref(),
                search_tier_matcher: self.search_tier_matcher.as_deref(),
                focal_locked: job.focal_locked,
            },
            &mut attempts,
        )?;
        let best = acq.best.map(|(_, _, cand)| cand);
        let k_render_eff = best.as_ref().map_or(job.k_render, |c| c.k);
        let seeded = best
            .as_ref()
            .filter(|c| {
                c.score.geometry_accepted && is_plausible_anchor_candidate(c, &self.surface)
            })
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
            k_render: k_render_eff,
            render_size: job.render_size,
            best,
            seeded,
            locked_focal_px: acq.locked_focal_px,
            attempts,
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
    pub(super) local_frame: Option<LocalFrame>,
    pub(super) focal_estimator: Option<&'a mut FocalLengthEstimator>,
    pub(super) dynamics_profile: VehicleDynamicsProfile,
}

impl<'a> AnchorArrival<'a> {
    /// Apply `outcome` on frame `arrived_at` (`None` after the last frame) and return the arrived anchor.
    pub(super) fn apply(
        &mut self,
        outcome: AnchorOutcome,
        arrived_at: Option<&str>,
    ) -> Option<ArrivedAnchor> {
        self.latencies.anchor_ms.push(outcome.elapsed_ms);
        if let (Some(locked), Some(est)) = (outcome.locked_focal_px, self.focal_estimator.as_mut())
        {
            est.locked_focal_px = Some(locked);
        }
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
            "attempt_count": outcome.attempts.len(),
            "attempts": &outcome.attempts,
            "anchor_ms": outcome.elapsed_ms,
        });
        let mut arrived_anchor = None;
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
            if arrived_at.is_some() {
                if self.state.prev_eye.is_none() || self.state.time_since_accepted_pose_s > 2.0 {
                    self.state.prev_vel = [0.0, 0.0, 0.0];
                } else {
                    let [vx, vy, vz] = self.state.prev_vel;
                    let horiz = vx.hypot(vy);
                    let max_h = self.dynamics_profile.max_speed_mps.max(0.0);
                    let scale_h = if horiz > max_h && horiz > 0.0 {
                        max_h / horiz
                    } else {
                        1.0
                    };
                    let max_v = self.dynamics_profile.max_vertical_speed_mps.max(0.0);
                    self.state.prev_vel = [vx * scale_h, vy * scale_h, vz.clamp(-max_v, max_v)];
                }
                self.state.prev_eye = Some(prop_eye);
                self.state.prev_rot = Some(prop_rot);
                self.state.vo_rot = Some(prop_rot);
                self.state.time_since_accepted_pose_s = 0.0;
                if let Some(kin) = self.state.kinematic.as_mut() {
                    kin.eye_enu = prop_eye;
                    kin.rot_world_to_cam = prop_rot;
                    kin.vel_enu_mps = self.state.prev_vel;
                } else {
                    let mut kin =
                        navigate_locate::corridor::KinematicState::new(prop_eye, prop_rot);
                    kin.vel_enu_mps = self.state.prev_vel;
                    self.state.kinematic = Some(kin);
                }
                if let (Some(session), Some(kin)) = (
                    self.state.corridor_session.as_mut(),
                    self.state.kinematic.clone(),
                ) {
                    session.dead_reckoner.kinematic = kin;
                }
            }
            self.state.frames_since_anchor = 0;
            self.state.time_since_anchor_s = 0.0;

            let mut prop_cand = cand.clone();
            prop_cand.eye = prop_eye;
            prop_cand.rotation = prop_rot;
            let frame_opt = self
                .local_frame
                .or_else(|| LocalFrame::wgs84_enu(40.0, -75.0).ok());
            let adm = if let Some(lf) = frame_opt
                && let Ok(report) = self.state.admit_anchor(
                    &prop_cand,
                    lf,
                    self.catalog_sha256,
                    self.cell_m,
                    &outcome.case_id,
                    "corridor-background-anchor",
                ) {
                if report.independent_ekf_fix {
                    *self.independent_fixes += 1;
                }
                FusionAdmission::from_session_report(report)
            } else {
                classify_anchor_cell(
                    &mut self.state.visited_cells,
                    self.independent_fixes,
                    self.catalog_sha256,
                    cand.eye,
                    self.cell_m,
                    &outcome.case_id,
                )
            };
            if let Some(map) = record.as_object_mut() {
                map.insert("independent_ekf_fix".into(), adm.independent_ekf_fix.into());
                map.insert("fusion_eligibility".into(), adm.fusion_eligibility.into());
                map.insert("map_cell".into(), json!(adm.map_cell));
                map.insert(
                    "active_tracks".into(),
                    self.state.active_tracks.len().into(),
                );
            }
            arrived_anchor = Some(ArrivedAnchor {
                pose: (prop_eye, prop_rot),
                candidate: prop_cand,
                admission: adm,
                attempts: outcome.attempts,
            });
        }
        self.results.push(record);
        arrived_anchor
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
        local_frame: Option<LocalFrame>,
        focal_estimator: &'a mut FocalLengthEstimator,
        dynamics_profile: VehicleDynamicsProfile,
        outcome: AnchorOutcome,
        arrived_at: Option<&str>,
    ) -> Option<ArrivedAnchor> {
        Self {
            state,
            independent_fixes,
            latencies,
            results,
            catalog_sha256,
            cell_m,
            local_frame,
            focal_estimator: Some(focal_estimator),
            dynamics_profile,
        }
        .apply(outcome, arrived_at)
    }
}

pub use navigate_locate::corridor::AnchorWorkerState;

#[cfg(test)]
mod tests;
