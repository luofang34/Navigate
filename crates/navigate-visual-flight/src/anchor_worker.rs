//! Bounded `corridor-map-anchor` worker with virtual-clock deterministic replay support.

use crate::{
    anchor::{
        AnchorQuery, AnchorQueryAux, AnchorSearchOpts, SeedPose, acquire_corridor_anchor,
        is_plausible_anchor_candidate,
    },
    diagnostics::AttemptRecord,
    error::FlightVisualError,
    runtime::{LoadedMatcher, LoadedXFeat},
    surface::{GpuQueueRole, SharedSurfaceGpu, Surface, SurfaceGpuRenderer},
    tracks::seed_tracks,
    types::{AnchorConfidence, ExecutionPolicy},
    worker::SupersedeWorker,
};
use image::{GrayImage, RgbImage};
use nalgebra::Matrix3;
use navigate_contract::ObservationStamp;
use navigate_locate::corridor::{ColdDeadReckonPrior, multitier::SearchTierMatcher};
use navigate_match::{
    ActiveTrackSet, CorridorTileStore, HadamardProjector, KashinScratch, PreparedKashinQuery,
    SparseFeatures,
};
use navigate_visual::PoseCandidate;
use std::{
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);

/// Read-only models, terrain, and tile store owned by the background anchor worker.
pub(crate) struct AnchorContext {
    pub surface: Arc<Surface>,
    pub store: Arc<CorridorTileStore>,
    pub projector: HadamardProjector,
    pub xfeat: LoadedXFeat,
    pub matcher: Option<LoadedMatcher>,
    pub search_tier_matcher: Option<Arc<SearchTierMatcher>>,
    pub search_opts: AnchorSearchOpts,
}

/// `Arc`-shared frame payload submitted to [`BackgroundAnchor`] without deep vector clones.
pub(crate) struct AnchorJob {
    pub sent_frame_seq: usize,
    pub capture_stamp: ObservationStamp,
    pub frame_label: String,
    pub query: AnchorQuery,
    pub qry_feat: Arc<SparseFeatures>,
    pub prepared: Arc<PreparedKashinQuery>,
    pub k_full: Matrix3<f64>,
    pub k_render: Matrix3<f64>,
    pub render_size: [u32; 2],
    pub pred_pose: Option<SeedPose>,
    pub live_attitude: Option<(f64, f64)>,
    pub cold_prior: Option<ColdDeadReckonPrior>,
    pub confirmatory_center_enu: Option<[f64; 3]>,
    pub small_rgb: Option<Arc<RgbImage>>,
    pub mask_small: Option<Arc<GrayImage>>,
    pub luma_p50: u8,
    pub collect_attempts: bool,
}

/// Completed output of one [`AnchorJob`].
pub(crate) struct AnchorOutcome {
    pub sent_frame_seq: usize,
    pub capture_stamp: ObservationStamp,
    pub frame_label: String,
    pub query_prior_enu: Option<[f64; 3]>,
    pub k_render: Matrix3<f64>,
    pub render_size: [u32; 2],
    pub agl_m: f64,
    pub best: Option<(usize, usize, PoseCandidate)>,
    pub seeded: Option<ActiveTrackSet>,
    pub locked_focal_px: Option<f64>,
    pub confidence: Option<AnchorConfidence>,
    pub was_polished: bool,
    pub lg_layers: usize,
    pub polish_rays: usize,
    pub attempts: Vec<AttemptRecord>,
    pub elapsed_ms: f64,
}

impl AnchorContext {
    fn run(
        &self,
        scratch: &mut KashinScratch,
        gpu_renderer: &Mutex<Option<SurfaceGpuRenderer>>,
        job: &AnchorJob,
    ) -> Result<AnchorOutcome, FlightVisualError> {
        let t0 = Instant::now();
        let mut attempts_vec = Vec::new();
        let attempts_opt = job.collect_attempts.then_some(&mut attempts_vec);
        let acq = acquire_corridor_anchor(
            (&*self.surface, &*self.store, &self.projector),
            scratch,
            Some(gpu_renderer),
            (
                &self.xfeat,
                self.matcher.as_ref().map(LoadedMatcher::as_matcher),
            ),
            (&job.query, &self.search_opts),
            (&job.qry_feat, &job.prepared),
            (&job.k_full, &job.k_render, job.render_size),
            AnchorQueryAux {
                pred_pose: job.pred_pose,
                live_attitude: job.live_attitude,
                cold_prior: job.cold_prior.as_ref(),
                confirmatory_center_enu: job.confirmatory_center_enu,
                small_rgb: job.small_rgb.as_deref(),
                mask_small: job.mask_small.as_deref(),
                luma_p50: Some(job.luma_p50),
                search_tier_matcher: self.search_tier_matcher.as_deref(),
            },
            attempts_opt,
        )?;
        let k_render_eff = acq.best.as_ref().map_or(job.k_render, |(_, _, c)| c.k);
        let agl_m = acq.best.as_ref().map_or(120.0, |(_, _, c)| {
            let geo = self.surface.geodetic(c.eye);
            let dem = self.surface.sample_dem(geo[0], geo[1]).unwrap_or(0.0);
            (geo[2] - dem).clamp(15.0, 8_000.0)
        });
        let seeded = acq
            .best
            .as_ref()
            .filter(|(_, _, c)| {
                c.score.geometry_accepted && is_plausible_anchor_candidate(c, &self.surface)
            })
            .map(|(_, _, cand)| {
                seed_tracks(
                    &self.surface,
                    &job.prepared,
                    &job.qry_feat,
                    cand,
                    &acq.best_view,
                    acq.was_polished,
                )
            });
        let query_prior_enu = job.query.prior.lat_lon().map(|(lat, lon)| {
            let dem = self.surface.sample_dem(lat, lon).unwrap_or(0.0);
            let alt = job.query.prior.alt_msl_m.unwrap_or(dem + agl_m);
            self.surface.local(lat, lon, alt)
        });
        Ok(AnchorOutcome {
            sent_frame_seq: job.sent_frame_seq,
            capture_stamp: job.capture_stamp,
            frame_label: job.frame_label.clone(),
            query_prior_enu,
            k_render: k_render_eff,
            render_size: job.render_size,
            agl_m,
            best: acq.best,
            seeded,
            locked_focal_px: acq.locked_focal_px,
            confidence: acq.confidence,
            was_polished: acq.was_polished,
            lg_layers: acq.lg_layers,
            polish_rays: acq.polish_rays,
            attempts: attempts_vec,
            elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

/// Bounded single-flight anchor worker supporting `LiveFlight` and `DeterministicReplay`.
pub(crate) struct BackgroundAnchor {
    worker: SupersedeWorker<AnchorJob, Result<AnchorOutcome, FlightVisualError>>,
    submit_virtual_ms: Option<u64>,
}

impl BackgroundAnchor {
    pub(crate) fn spawn(
        context: AnchorContext,
        shared_gpu: Option<&Arc<SharedSurfaceGpu>>,
        init_k: &Matrix3<f64>,
        init_render_size: [u32; 2],
        init_pose: Option<SeedPose>,
    ) -> Result<Self, FlightVisualError> {
        let init_gpu = if context.search_opts.polish_anchors {
            shared_gpu
                .and_then(|shared| {
                    SurfaceGpuRenderer::from_shared(
                        shared,
                        init_k,
                        init_render_size,
                        GpuQueueRole::BackgroundAnchor,
                    )
                })
                .or_else(|| {
                    SurfaceGpuRenderer::try_new_with_role(
                        &context.surface,
                        init_k,
                        init_render_size,
                        GpuQueueRole::BackgroundAnchor,
                    )
                })
                .map(|mut g| {
                    if let Some((eye, ref rot)) = init_pose {
                        drop(g.render_view(&context.surface, eye, rot, init_k, init_render_size));
                    }
                    g
                })
        } else {
            None
        };
        let gpu_renderer = Mutex::new(init_gpu);
        let mut scratch = KashinScratch::with_capacity(512);
        let worker = SupersedeWorker::spawn("corridor-map-anchor", SHUTDOWN_TIMEOUT, move |job| {
            context.run(&mut scratch, &gpu_renderer, &job)
        })?;
        Ok(Self {
            worker,
            submit_virtual_ms: None,
        })
    }

    pub(crate) const fn is_in_flight(&self) -> bool {
        self.submit_virtual_ms.is_some()
    }

    pub(crate) fn submit_if_idle(
        &mut self,
        virtual_now_ms: u64,
        build_job: impl FnOnce() -> AnchorJob,
    ) -> Result<bool, FlightVisualError> {
        if self.submit_virtual_ms.is_some() {
            return Ok(false);
        }
        let sent = self.worker.submit_if_idle(build_job)?;
        if sent {
            self.submit_virtual_ms = Some(virtual_now_ms);
        }
        Ok(sent)
    }

    pub(crate) fn poll_with_policy(
        &mut self,
        policy: ExecutionPolicy,
        virtual_now_ms: u64,
    ) -> Result<Option<AnchorOutcome>, FlightVisualError> {
        let Some(submit_ms) = self.submit_virtual_ms else {
            return Ok(None);
        };
        match policy {
            ExecutionPolicy::LiveFlight => {
                let Some(res) = self.worker.try_recv()? else {
                    return Ok(None);
                };
                self.submit_virtual_ms = None;
                match res {
                    Ok(out) => Ok(Some(out)),
                    Err(err) => {
                        self.worker.mark_faulted();
                        Err(err)
                    }
                }
            }
            ExecutionPolicy::DeterministicReplay {
                anchor_budget_ms, ..
            } => {
                if virtual_now_ms < submit_ms.saturating_add(anchor_budget_ms) {
                    return Ok(None);
                }
                self.submit_virtual_ms = None;
                match self.worker.recv_blocking()? {
                    Ok(out) => Ok(Some(out)),
                    Err(err) => {
                        self.worker.mark_faulted();
                        Err(err)
                    }
                }
            }
        }
    }

    pub(crate) fn drain_shutdown(&mut self) -> Result<Option<AnchorOutcome>, FlightVisualError> {
        if self.submit_virtual_ms.is_none() {
            return Ok(None);
        }
        self.submit_virtual_ms = None;
        match self.worker.recv_blocking()? {
            Ok(out) => Ok(Some(out)),
            Err(err) => {
                self.worker.mark_faulted();
                Err(err)
            }
        }
    }

    pub(crate) fn is_alive(&self) -> bool {
        self.worker.is_alive()
    }

    pub(crate) fn is_faulted(&self) -> bool {
        self.worker.is_faulted()
    }

    pub(crate) fn superseded_count(&self) -> u32 {
        self.worker.superseded_count()
    }

    pub(crate) fn dropped_count(&self) -> u32 {
        self.worker.dropped_count()
    }
}
