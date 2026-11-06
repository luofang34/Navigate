//! Per-frame pose solving: warm 3D tracking, fast-roll fallback, and map-anchor dispatch.

use super::{
    FlightVisualEngine,
    warm_track::{
        attempt_warm_solve, predict_step_kinematics, refresh_warm_frustum_and_top_up,
        resolve_anchor_query_prior, try_recover_from_recent_backup,
    },
};
use crate::{
    anchor::{
        AnchorPrior, AnchorQuery, AnchorQueryAux, acquire_corridor_anchor,
        is_plausible_anchor_candidate,
    },
    anchor_arrival::{
        ActiveAnchorStatus, ArrivedAnchor, FrameAdmission, classify_anchor_cell_fallback,
    },
    anchor_worker::AnchorJob,
    diagnostics::AttemptRecord,
    error::FlightVisualError,
    runtime::LoadedMatcher,
    tracks::{seed_tracks, top_up_tracks_with_bank},
    types::{AnchorConfidence, CalibrationMode, LiveFrameInput},
};
use image::{GrayImage, RgbImage};
use nalgebra::Matrix3;
use navigate_locate::corridor::{project_track_strategy, should_force_anchor};
use navigate_match::{ActiveTrackSet, KashinScratch, PreparedKashinQuery, SparseFeatures};
use navigate_visual::{PoseCandidate, session::MapCell};
use std::{sync::Arc, time::Instant};

pub(super) struct FrameSolveInput<'a> {
    pub frame: &'a LiveFrameInput<'a>,
    pub frame_label: &'a str,
    pub virtual_now_ms: u64,
    pub anchor_prior: AnchorPrior,
    pub attitude_override: Option<(f64, f64)>,
    pub qry_feat: SparseFeatures,
    pub prepared: &'a PreparedKashinQuery,
    pub small_rgb: Option<RgbImage>,
    pub mask_small: GrayImage,
    pub luma_p50: u8,
    pub k_full: Matrix3<f64>,
    pub k_render: Matrix3<f64>,
    pub render_size: [u32; 2],
    pub nominal_agl_m: f64,
    pub dt_s: f64,
    pub cell_m: f64,
    pub arrived_anchor: Option<ArrivedAnchor>,
    pub collect_diag: bool,
}

pub(super) struct FrameSolveResult {
    pub best: Option<(usize, usize, PoseCandidate)>,
    pub admission: FrameAdmission,
    pub lg_layers: usize,
    pub polish_rays: usize,
    pub solve_ms: f64,
    pub attempts: Vec<AttemptRecord>,
    pub used_arrived_anchor: bool,
    pub qry_feat_len: usize,
}

pub(super) fn solve_frame_pose(
    eng: &mut FlightVisualEngine,
    inp: FrameSolveInput<'_>,
) -> Result<FrameSolveResult, FlightVisualError> {
    let t_solve = Instant::now();
    let mut attempts = Vec::new();
    let mut lg_layers = 0_usize;
    let mut polish_rays = 0_usize;
    let qry_feat_len = inp.qry_feat.len();
    let calibrated_focal = !matches!(
        inp.frame.intrinsics.calibration_confidence,
        CalibrationMode::Uncalibrated
    );
    let focal_locked = eng.focal_estimator.locked_focal_px.is_some();

    let pred_pose = inp
        .arrived_anchor
        .as_ref()
        .filter(|a| !a.fg_healthy_on_arrival)
        .map(|a| a.pose)
        .or_else(|| {
            eng.state.predict_interframe_pose(
                inp.prepared,
                &inp.k_render,
                inp.dt_s,
                &eng.config.dynamics_profile,
            )
        });

    let max_confirm = if matches!(eng.config.policy, crate::types::ExecutionPolicy::LiveFlight) {
        1
    } else {
        2
    };
    let provisional_confirm = eng.active_status.is_some_and(|st| {
        st.confidence == AnchorConfidence::Provisional && st.confirm_attempts < max_confirm
    }) && eng.state.time_since_anchor_s >= 0.75
        && eng.state.frames_since_anchor >= 8;
    let confirmatory_center_enu = provisional_confirm
        .then(|| eng.active_status.and_then(|st| st.prior_center_enu))
        .flatten();
    let anchor_pred_pose = (!provisional_confirm).then_some(pred_pose).flatten();

    let mut force_anchor = provisional_confirm
        || (eng.config.anchor_on_cell_boundary
            && !eng.state.active_tracks.is_empty()
            && eng.state.time_since_anchor_s >= 3.5
            && (inp.luma_p50 < 75 || eng.state.active_tracks.len() < 500))
        || should_force_anchor(
            eng.config.anchor_interval_frames,
            eng.state.frames_since_anchor,
            eng.config.anchor_interval_s,
            eng.state.time_since_anchor_s,
            eng.config.search_opts.anchor_uncertainty_m,
            eng.state.kinematic.as_ref(),
            eng.config.anchor_on_cell_boundary,
            pred_pose.as_ref(),
            &eng.state.visited_cells,
            inp.cell_m,
        );

    let forced_pred_cell =
        (force_anchor && !provisional_confirm && eng.config.anchor_on_cell_boundary)
            .then_some(pred_pose)
            .flatten()
            .map(|(pe, _)| MapCell {
                map_manifest_sha256: eng.resources.surface.catalog_sha256.clone(),
                east: (pe[0] / inp.cell_m).floor() as i64,
                north: (pe[1] / inp.cell_m).floor() as i64,
            });

    let effective_query_prior = resolve_anchor_query_prior(
        &eng.resources.surface,
        inp.anchor_prior,
        eng.state.cold_prior.as_ref(),
        inp.nominal_agl_m,
        anchor_pred_pose,
    );
    let query = AnchorQuery {
        prior: effective_query_prior,
        calibrated_focal,
        focal_locked,
        attitude_override: inp.attitude_override,
    };

    let qry_feat_arc = Arc::new(inp.qry_feat);
    let mask_arc = Arc::new(inp.mask_small);
    let rgb_arc = inp.small_rgb.map(Arc::new);

    if force_anchor && !eng.state.active_tracks.is_empty() && eng.background_anchor.is_some() {
        if eng.state.frames_since_anchor >= 8
            && let Some(worker) = eng.background_anchor.as_mut()
        {
            let seq = eng.state.frame_seq;
            let live_att = eng.masker.horizon_attitude(&inp.k_render);
            let cold_prior = eng.state.cold_prior;
            let prep_arc = Arc::new(inp.prepared.clone());
            let qf = Arc::clone(&qry_feat_arc);
            let mk = Arc::clone(&mask_arc);
            let rgb = rgb_arc.clone();
            let stamp = inp.frame.stamp;
            let label = inp.frame_label.to_owned();
            let submitted = worker.submit_if_idle(inp.virtual_now_ms, move || AnchorJob {
                sent_frame_seq: seq,
                capture_stamp: stamp,
                frame_label: label,
                query,
                qry_feat: qf,
                prepared: prep_arc,
                k_full: inp.k_full,
                k_render: inp.k_render,
                render_size: inp.render_size,
                pred_pose: anchor_pred_pose,
                live_attitude: live_att,
                cold_prior,
                confirmatory_center_enu,
                small_rgb: rgb,
                mask_small: Some(mk),
                luma_p50: inp.luma_p50,
                collect_attempts: inp.collect_diag,
            })?;
            if submitted && let Some(ref cell) = forced_pred_cell {
                eng.state
                    .visited_cells
                    .entry(cell.clone())
                    .or_insert_with(|| inp.frame_label.to_owned());
            }
        }
        force_anchor = false;
    }

    let predicted_kin = predict_step_kinematics(
        eng.state.kinematic.as_ref(),
        pred_pose,
        inp.dt_s,
        &eng.config.dynamics_profile,
    );
    let strategy = project_track_strategy(
        predicted_kin.as_ref(),
        pred_pose,
        &eng.state.active_tracks,
        &inp.k_render,
        inp.nominal_agl_m,
    );
    let min_trk = eng.config.min_track_inliers.max(24);

    let mut solved_warm = None;
    if !eng.state.active_tracks.is_empty()
        && !force_anchor
        && let Some(pred) = pred_pose
    {
        solved_warm = attempt_warm_solve(
            &eng.resources.surface,
            &eng.state.active_tracks,
            inp.prepared,
            pred,
            (&inp.k_render, inp.render_size, min_trk),
            predicted_kin.as_ref(),
            strategy,
            inp.collect_diag.then_some(&mut attempts),
        )?;
    }

    let mut used_arrived_anchor = false;
    let (best, admission) = if let Some(cand) = solved_warm {
        eng.state.frames_since_anchor = eng.state.frames_since_anchor.saturating_add(1);
        eng.state
            .record_warm_track(&cand, &eng.resources.surface.catalog_sha256);
        refresh_warm_frustum_and_top_up(eng, &cand, inp.prepared, &qry_feat_arc, inp.render_size);
        (Some((0, 0, cand)), FrameAdmission::warm_track())
    } else if let Some(arr) = inp.arrived_anchor {
        used_arrived_anchor = true;
        lg_layers += arr.lg_layers;
        polish_rays += arr.polish_rays;
        if arr.fg_healthy_on_arrival {
            eng.state.active_tracks = arr.seeded_tracks;
            eng.state.prev_eye = Some(arr.candidate.eye);
            eng.state.prev_rot = Some(arr.candidate.rotation);
            eng.state.vo_rot = Some(arr.candidate.rotation);
            eng.state.time_since_accepted_pose_s = 0.0;
        }
        top_up_tracks_with_bank(
            &mut eng.state.active_tracks,
            &eng.resources.surface,
            &eng.resources.store,
            inp.prepared,
            &qry_feat_arc,
            &arr.candidate,
            inp.render_size,
        );
        if inp.collect_diag {
            attempts.extend(arr.attempts);
        }
        (
            Some((arr.seed_idx, arr.step_idx, arr.candidate)),
            arr.admission,
        )
    } else if let Some(cand) = try_recover_from_recent_backup(
        eng,
        inp.frame.stamp.acquired_at.as_nanos(),
        inp.prepared,
        &qry_feat_arc,
        (&inp.k_render, inp.render_size, min_trk),
        inp.collect_diag.then_some(&mut attempts),
    )? {
        eng.state.frames_since_anchor = eng.state.frames_since_anchor.saturating_add(1);
        eng.state
            .record_warm_track(&cand, &eng.resources.surface.catalog_sha256);
        (Some((0, 0, cand)), FrameAdmission::warm_track())
    } else if let Some(worker) = eng.background_anchor.as_mut() {
        let should_submit_miss =
            eng.state.active_tracks.is_empty() || eng.state.time_since_accepted_pose_s >= 0.10;
        if !qry_feat_arc.is_empty() && should_submit_miss {
            let seq = eng.state.frame_seq;
            let live_att = eng.masker.horizon_attitude(&inp.k_render);
            let cold_prior = eng.state.cold_prior;
            let prep_arc = Arc::new(inp.prepared.clone());
            let qf = Arc::clone(&qry_feat_arc);
            let mk = Arc::clone(&mask_arc);
            let rgb = rgb_arc;
            let stamp = inp.frame.stamp;
            let label = inp.frame_label.to_owned();
            let _ = worker.submit_if_idle(inp.virtual_now_ms, move || AnchorJob {
                sent_frame_seq: seq,
                capture_stamp: stamp,
                frame_label: label,
                query,
                qry_feat: qf,
                prepared: prep_arc,
                k_full: inp.k_full,
                k_render: inp.k_render,
                render_size: inp.render_size,
                pred_pose,
                live_attitude: live_att,
                cold_prior,
                confirmatory_center_enu: None,
                small_rgb: rgb,
                mask_small: Some(mk),
                luma_p50: inp.luma_p50,
                collect_attempts: inp.collect_diag,
            })?;
        }
        (None, FrameAdmission::acquiring_anchor())
    } else {
        let mut scratch = KashinScratch::with_capacity(512);
        let acq = acquire_corridor_anchor(
            (
                &eng.resources.surface,
                &eng.resources.store,
                &eng.resources.projector,
            ),
            &mut scratch,
            Some(&eng.sync_gpu_renderer),
            (
                &eng.resources.xfeat,
                eng.resources
                    .matcher
                    .as_ref()
                    .map(LoadedMatcher::as_matcher),
            ),
            (&query, &eng.config.search_opts),
            (&qry_feat_arc, inp.prepared),
            (&inp.k_full, &inp.k_render, inp.render_size),
            AnchorQueryAux {
                pred_pose,
                live_attitude: eng.masker.horizon_attitude(&inp.k_render),
                cold_prior: eng.state.cold_prior.as_ref(),
                confirmatory_center_enu,
                small_rgb: rgb_arc.as_deref(),
                mask_small: Some(&mask_arc),
                luma_p50: Some(inp.luma_p50),
                search_tier_matcher: eng.resources.search_tier_matcher.as_deref(),
            },
            inp.collect_diag.then_some(&mut attempts),
        )?;
        lg_layers += acq.lg_layers;
        polish_rays += acq.polish_rays;
        if let Some(locked_px) = acq.locked_focal_px {
            eng.focal_estimator.locked_focal_px = Some(locked_px);
        }
        if let Some((_, _, ref cand)) = acq.best
            && cand.score.geometry_accepted
            && is_plausible_anchor_candidate(cand, &eng.resources.surface)
        {
            eng.state.frames_since_anchor = 0;
            eng.state.time_since_anchor_s = 0.0;
            eng.state.active_tracks = seed_tracks(
                &eng.resources.surface,
                inp.prepared,
                &qry_feat_arc,
                cand,
                &acq.best_view,
                acq.was_polished,
            );
            let conf = acq.confidence.unwrap_or(AnchorConfidence::Confirmed);
            eng.active_status = Some(ActiveAnchorStatus {
                confidence: conf,
                holdout_median_px: cand.score.holdout_median_px,
                prior_center_enu: effective_query_prior.lat_lon().map(|(lat, lon)| {
                    eng.resources.surface.local(
                        lat,
                        lon,
                        effective_query_prior.alt_msl_m.unwrap_or(0.0),
                    )
                }),
                confirm_attempts: 0,
            });
            let adm = if let Ok(report) = eng.state.admit_anchor(
                cand,
                eng.resources.local_frame,
                &eng.resources.surface.catalog_sha256,
                inp.cell_m,
                inp.frame_label,
                "corridor-map-anchor",
            ) {
                if report.independent_ekf_fix {
                    eng.independent_fixes = eng.independent_fixes.saturating_add(1);
                }
                FrameAdmission::from_session_report(report)
            } else {
                classify_anchor_cell_fallback(
                    &mut eng.state.visited_cells,
                    &mut eng.independent_fixes,
                    &eng.resources.surface.catalog_sha256,
                    cand.eye,
                    inp.cell_m,
                    inp.frame_label,
                    0,
                )
            };
            if let Some(cell) = forced_pred_cell {
                eng.state
                    .visited_cells
                    .entry(cell)
                    .or_insert_with(|| inp.frame_label.to_owned());
            }
            (acq.best, adm)
        } else {
            eng.state.active_tracks = ActiveTrackSet::empty();
            (acq.best, FrameAdmission::unresolved())
        }
    };

    Ok(FrameSolveResult {
        best,
        admission,
        lg_layers,
        polish_rays,
        solve_ms: t_solve.elapsed().as_secs_f64() * 1000.0,
        attempts,
        used_arrived_anchor,
        qry_feat_len,
    })
}
