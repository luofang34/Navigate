//! Frame pose solving, warm track gating, track top-up, and session anchor submission.

use crate::{
    PipelineError,
    corridor::{
        CorridorTrackOptions,
        admission::{
            FusionAdmission, TOP_UP_FRUSTUM_MARGIN_PX, classify_anchor_cell, seed_tracks,
            top_up_tracks_with_bank,
        },
        anchor::{
            AnchorQueryAux, acquire_corridor_anchor, is_plausible_anchor_candidate,
            multitier::SearchTierMatcher, plausibility::compute_fisher_geometry_covariance,
        },
        anchor_worker::{AnchorJob, ArrivedAnchor, BackgroundAnchor},
        kinematics::{CHI2_GATE_3DOF_99, project_track_strategy, should_force_anchor},
        latency::StageLatencies,
        masking::AmortizedMasker,
        motion::{TrackState, predict_interframe_pose, try_warm_track},
    },
    pose::PoseCandidate,
    realtime::helpers::rot_to_row_major,
    stages::{focal::FocalLengthEstimator, scaled_size},
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
use serde_json::Value;
use std::{sync::Mutex, time::Instant};

/// Compute the live tracking render size for `30 Hz` inference.
#[must_use]
pub(super) fn tracking_render_size(case_size: [u32; 2]) -> [u32; 2] {
    scaled_size(case_size, 800)
}

/// Context parameters for solving a single frame pose.
pub(super) struct FrameSolveParams<'a> {
    pub surface: &'a Surface,
    pub store: &'a CorridorTileStore,
    pub projector: &'a HadamardProjector,
    pub sync_gpu_renderer: &'a Mutex<Option<SurfaceGpuRenderer>>,
    pub xfeat: &'a LoadedXFeat,
    pub lighterglue: Option<&'a LighterGlue>,
    pub search_tier_matcher: Option<&'a SearchTierMatcher>,
    pub opts: &'a CorridorTrackOptions,
    pub case: &'a PhotoCase,
    pub qry_feat: &'a SparseFeatures,
    pub prepared: &'a PreparedKashinQuery,
    pub mask_small: &'a GrayImage,
    pub k: Matrix3<f64>,
    pub k_render: Matrix3<f64>,
    pub render_size: [u32; 2],
    pub masker: &'a AmortizedMasker,
    pub local_frame: Option<LocalFrame>,
    pub nominal_agl_m: f64,
    pub dt_s: f64,
    pub cell_m: f64,
    pub min_trk_inliers: usize,
    pub arrived_anchor: Option<ArrivedAnchor>,
}

/// Outcome of solving a single frame pose.
pub(super) struct FrameSolveOutcome {
    pub best: Option<(usize, usize, PoseCandidate)>,
    pub admission: FusionAdmission,
    pub lg_layers: usize,
    pub polish_rays: usize,
    pub solve_ms: f64,
    pub attempts_log: Vec<Value>,
    pub used_arrived_anchor: bool,
}

/// Solve camera pose for the current frame using warm tracking or anchor acquisition.
#[allow(clippy::too_many_arguments)]
pub(super) fn solve_frame_pose(
    params: FrameSolveParams<'_>,
    state: &mut TrackState,
    focal_estimator: &mut FocalLengthEstimator,
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
    let focal_locked = focal_estimator.locked_focal_px.is_some();

    let pred_pose = params.arrived_anchor.as_ref().map(|a| a.pose).or_else(|| {
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

    if force_anchor && !state.active_tracks.is_empty() && background_anchor.is_some() {
        if state.frames_since_anchor >= 8
            && let Some(worker) = background_anchor.as_mut()
        {
            let seq = state.frame_seq;
            let mask_clone = params.mask_small.clone();
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
                mask_small: Some(mask_clone),
                focal_locked,
            })?;
        }
        force_anchor = false;
    }

    let predicted_kin = state.kinematic.as_ref().map(|kin| {
        let mut kp = kin.clone();
        if let Some((pe, pr)) = pred_pose {
            kp.eye_enu = pe;
            kp.rot_world_to_cam = pr;
        }
        let step_disp = params.opts.dynamics_profile.max_displacement(params.dt_s);
        let vel_norm_sq = kp.vel_enu_mps[0] * kp.vel_enu_mps[0]
            + kp.vel_enu_mps[1] * kp.vel_enu_mps[1]
            + kp.vel_enu_mps[2] * kp.vel_enu_mps[2];
        let step_var = if vel_norm_sq < 1e-6 {
            (step_disp * 0.65).powi(2).max(4.0)
        } else {
            let accel_disp =
                0.5 * params.opts.dynamics_profile.rms_accel_mps2 * params.dt_s * params.dt_s;
            accel_disp.max(step_disp * 0.30).powi(2).max(2.25)
        };
        kp.pos_cov_enu_m2 += Matrix3::identity() * step_var;
        kp.att_sigma_rad += params.opts.dynamics_profile.max_angular_rate_rad_s * 0.1 * params.dt_s;
        kp
    });

    let strategy = project_track_strategy(
        predicted_kin.as_ref(),
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
            let plausible = is_plausible_anchor_candidate(cand, params.surface);
            let passes_gate = if let Some(ref kin) = predicted_kin {
                let cov6 = compute_fisher_geometry_covariance(cand);
                let geom_cov = cov6.fixed_view::<3, 3>(0, 0).into_owned();
                kin.mahalanobis_innovation_gate(cand.eye, geom_cov, CHI2_GATE_3DOF_99)
            } else {
                true
            };
            if !plausible || !passes_gate {
                solved_warm = None;
            }
        }
    }

    let mut used_arrived_anchor = false;
    let (best, admission) = if let Some(cand) = solved_warm {
        *warm_tracked_frames += 1;
        state.frames_since_anchor += 1;
        state.record_warm_track(&cand, &params.surface.catalog_sha256);
        let visible_cap = params.qry_feat.len();
        let target_tracks = 1280.min(visible_cap);
        let check_frustum = state.frames_since_anchor.is_multiple_of(6) || cand.score.inliers < 180;
        if check_frustum {
            let intrinsics = [
                cand.k[(0, 0)],
                cand.k[(1, 1)],
                cand.k[(0, 2)],
                cand.k[(1, 2)],
            ];
            state.active_tracks.retain_in_frustum(
                cand.eye,
                &rot_to_row_major(&cand.rotation),
                intrinsics,
                params.render_size,
                TOP_UP_FRUSTUM_MARGIN_PX,
            );
        }
        let need_urgent_top_up = state.active_tracks.len() < 128 || cand.score.inliers < 40;
        let need_periodic_top_up = state.frames_since_anchor.is_multiple_of(6)
            && state.active_tracks.len() < target_tracks;
        if need_urgent_top_up || need_periodic_top_up {
            top_up_tracks_with_bank(
                &mut state.active_tracks,
                params.surface,
                params.store,
                params.projector,
                params.prepared,
                params.qry_feat,
                &cand,
                params.render_size,
            );
        }
        (Some((0, 0, cand)), FusionAdmission::warm_track())
    } else if let Some(arr) = params.arrived_anchor {
        used_arrived_anchor = true;
        attempts_log.extend(arr.attempts);
        (Some((0, 0, arr.candidate)), arr.admission)
    } else if let Some(worker) = background_anchor.as_mut() {
        if params.masker.has_usable_key(params.render_size) || !params.masker.has_segformer() {
            let seq = state.frame_seq;
            let mask_clone = params.mask_small.clone();
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
                mask_small: Some(mask_clone),
                focal_locked,
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
            AnchorQueryAux {
                pred_pose,
                live_attitude: params.masker.horizon_attitude(&params.k_render),
                mask_small: Some(params.mask_small),
                search_tier_matcher: params.search_tier_matcher,
                focal_locked,
            },
            &mut attempts_log,
        )?;
        latencies
            .anchor_ms
            .push(t_anchor.elapsed().as_secs_f64() * 1000.0);
        lg_layers += acq.lg_layers;
        polish_rays += acq.polish_rays;
        if let Some(locked_px) = acq.locked_focal_px {
            focal_estimator.locked_focal_px = Some(locked_px);
        }
        if let Some((_, _, ref cand)) = acq.best
            && cand.score.geometry_accepted
            && is_plausible_anchor_candidate(cand, params.surface)
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
            let adm = if let Some(lf) = params.local_frame
                && let Ok(report) = state.admit_anchor(
                    cand,
                    lf,
                    &params.surface.catalog_sha256,
                    params.cell_m,
                    &params.case.id,
                    "corridor-map-anchor",
                ) {
                if report.independent_ekf_fix {
                    *independent_fixes += 1;
                }
                FusionAdmission::from_session_report(report)
            } else {
                classify_anchor_cell(
                    &mut state.visited_cells,
                    independent_fixes,
                    &params.surface.catalog_sha256,
                    cand.eye,
                    params.cell_m,
                    &params.case.id,
                )
            };
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
        used_arrived_anchor,
    })
}
