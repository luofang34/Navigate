//! Frame pose solving, warm track gating, track top-up, and anchor submission.

use crate::{
    PipelineError,
    corridor::{
        CorridorTrackOptions,
        admission::{FusionAdmission, classify_anchor_cell, seed_tracks, top_up_tracks},
        anchor::acquire_corridor_anchor,
        anchor_worker::{AnchorJob, BackgroundAnchor},
        kinematics::{project_track_strategy, should_force_anchor},
        latency::StageLatencies,
        masking::AmortizedMasker,
        motion::{TrackState, predict_interframe_pose, try_warm_track},
    },
    pose::PoseCandidate,
    surface::{Surface, SurfaceGpuRenderer},
    testset::PhotoCase,
    xfeat_runtime::LoadedXFeat,
};
use nalgebra::Matrix3;
use navigate_match::{
    ActiveTrackSet, CorridorTileStore, HadamardProjector, LighterGlue, PreparedKashinQuery,
    SparseFeatures,
};
use serde_json::Value;
use std::{sync::Mutex, time::Instant};

/// Context parameters for solving a single frame pose.
pub(super) struct FrameSolveParams<'a> {
    pub surface: &'a Surface,
    pub store: &'a CorridorTileStore,
    pub projector: &'a HadamardProjector,
    pub sync_gpu_renderer: &'a Mutex<Option<SurfaceGpuRenderer>>,
    pub xfeat: &'a LoadedXFeat,
    pub lighterglue: Option<&'a LighterGlue>,
    pub opts: &'a CorridorTrackOptions,
    pub case: &'a PhotoCase,
    pub qry_feat: &'a SparseFeatures,
    pub prepared: &'a PreparedKashinQuery,
    pub k: Matrix3<f64>,
    pub k_render: Matrix3<f64>,
    pub render_size: [u32; 2],
    pub masker: &'a AmortizedMasker,
    pub nominal_agl_m: f64,
    pub dt_s: f64,
    pub cell_m: f64,
    pub min_trk_inliers: usize,
    pub arrived_pose: Option<([f64; 3], Matrix3<f64>)>,
}

/// Outcome of solving a single frame pose.
pub(super) struct FrameSolveOutcome {
    pub best: Option<(usize, usize, PoseCandidate)>,
    pub admission: FusionAdmission,
    pub lg_layers: usize,
    pub polish_rays: usize,
    pub solve_ms: f64,
    pub attempts_log: Vec<Value>,
}

/// Solves camera pose for the current frame using warm tracking or anchor acquisition.
pub(super) fn solve_frame_pose(
    params: FrameSolveParams<'_>,
    state: &mut TrackState,
    background_anchor: &mut Option<BackgroundAnchor>,
    independent_fixes: &mut usize,
    latencies: &mut StageLatencies,
    warm_tracked_frames: &mut usize,
    acquiring_frames: &mut usize,
) -> Result<FrameSolveOutcome, PipelineError> {
    let t_solve = Instant::now();
    let mut attempts_log = Vec::new();
    let mut lg_layers = 0_usize;
    let mut polish_rays = 0_usize;

    let pred_pose = params.arrived_pose.or_else(|| {
        predict_interframe_pose(
            state,
            params.prepared,
            &params.k_render,
            params.dt_s,
            &params.opts.dynamics_profile,
        )
    });

    let mut force_anchor = should_force_anchor(
        params.opts.anchor_interval,
        state.frames_since_anchor,
        params.opts.anchor_interval_s,
        state.time_since_anchor_s,
        params.opts.anchor_uncertainty_m,
        state.kinematic.as_ref(),
        params.opts.anchor_on_cell_boundary,
        pred_pose.as_ref(),
        &state.visited_cells,
        params.cell_m,
    );

    if force_anchor
        && !state.active_tracks.is_empty()
        && let Some(worker) = background_anchor.as_mut()
    {
        let seq = state.frame_seq;
        let _ = worker.submit_if_idle(|| AnchorJob {
            sent_frame_seq: seq,
            case: params.case.clone(),
            qry_feat: params.qry_feat.clone(),
            prepared: params.prepared.clone(),
            k: params.k,
            k_render: params.k_render,
            render_size: params.render_size,
            pred_pose,
            live_attitude: params.masker.horizon_attitude(&params.k_render),
        })?;
        force_anchor = false;
    }

    let strategy = project_track_strategy(
        state.kinematic.as_ref(),
        pred_pose,
        &state.active_tracks,
        &params.k_render,
        params.nominal_agl_m,
    );

    let mut solved_warm = None;
    if !state.active_tracks.is_empty()
        && !force_anchor
        && let Some(pred) = pred_pose
    {
        solved_warm = try_warm_track(
            &state.active_tracks,
            params.prepared,
            pred,
            &params.k_render,
            params.render_size,
            params.min_trk_inliers,
            &mut attempts_log,
            strategy,
        )?;
        if let Some(cand) = solved_warm.as_ref() {
            let passes_gate = if let Some(ref kin) = state.kinematic {
                let geom_cov = nalgebra::Matrix3::identity()
                    * (cand.score.train_rms_px.max(0.5).powi(2) * 2.0);
                kin.mahalanobis_innovation_gate(cand.eye, geom_cov, 11.345)
            } else {
                true
            };
            if !passes_gate {
                solved_warm = None;
            }
        }
    }

    let (best, admission) = if let Some(cand) = solved_warm {
        *warm_tracked_frames += 1;
        state.frames_since_anchor += 1;
        if cand.score.inliers < 260 {
            top_up_tracks(
                &mut state.active_tracks,
                params.surface,
                params.prepared,
                params.qry_feat,
                &cand,
                params.render_size,
            );
        }
        (Some((0, 0, cand)), FusionAdmission::warm_track())
    } else if let Some(worker) = background_anchor.as_mut() {
        if params.masker.has_usable_key(params.render_size) || !params.masker.has_segformer() {
            let seq = state.frame_seq;
            let _ = worker.submit_if_idle(|| AnchorJob {
                sent_frame_seq: seq,
                case: params.case.clone(),
                qry_feat: params.qry_feat.clone(),
                prepared: params.prepared.clone(),
                k: params.k,
                k_render: params.k_render,
                render_size: params.render_size,
                pred_pose,
                live_attitude: params.masker.horizon_attitude(&params.k_render),
            })?;
        }
        *acquiring_frames += 1;
        (None, FusionAdmission::acquiring_anchor())
    } else {
        let t_anchor = Instant::now();
        let acq = acquire_corridor_anchor(
            (params.surface, params.store, params.projector),
            Some(params.sync_gpu_renderer),
            (params.xfeat, params.lighterglue),
            (params.case, params.opts),
            (params.qry_feat, params.prepared),
            (&params.k, &params.k_render, params.render_size),
            (pred_pose, params.masker.horizon_attitude(&params.k_render)),
            &mut attempts_log,
        )?;
        latencies
            .anchor_ms
            .push(t_anchor.elapsed().as_secs_f64() * 1000.0);
        lg_layers += acq.lg_layers;
        polish_rays += acq.polish_rays;
        if let Some((_, _, ref cand)) = acq.best
            && cand.score.geometry_accepted
        {
            state.frames_since_anchor = 0;
            state.time_since_anchor_s = 0.0;
            state.active_tracks = seed_tracks(
                params.surface,
                params.prepared,
                params.qry_feat,
                cand,
                &acq.best_view,
            );
            let adm = classify_anchor_cell(
                &mut state.visited_cells,
                independent_fixes,
                &params.surface.catalog_sha256,
                cand.eye,
                params.cell_m,
                &params.case.id,
            );
            (acq.best, adm)
        } else {
            state.active_tracks = ActiveTrackSet::empty();
            (acq.best, FusionAdmission::unresolved())
        }
    };

    let solve_ms = t_solve.elapsed().as_secs_f64() * 1000.0;
    Ok(FrameSolveOutcome {
        best,
        admission,
        lg_layers,
        polish_rays,
        solve_ms,
        attempts_log,
    })
}
