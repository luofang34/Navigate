//! Two-tier `25-30 Hz` temporal corridor tracker, session fusion, and windowed tile streaming.

use super::{
    CorridorTrackOptions,
    admission::{
        apply_case_overrides, estimate_lookahead_enu, prior_odometry_hint, write_corridor_report,
    },
    anchor::multitier::load_search_tier_matcher,
    anchor_worker::{AnchorArrival, BackgroundAnchor},
    latency::StageLatencies,
    masking::{AmortizedMasker, MaskFrameTelemetry},
    motion::{TrackState, advance_unsolved_prepared, record_odometry_step, update_motion_state},
};

mod lookahead;
mod reporting;
mod solve;
use crate::{
    PipelineError,
    realtime::helpers::{check_encoder_contract, load_lighterglue_model, raw_file_sha256},
    stages::{
        estimate_intrinsics_with_estimator, focal::FocalLengthEstimator, load_case_mask,
        load_case_rgb, load_profile_cases, scale_k,
    },
    surface::Surface,
    util::write_json,
    xfeat_runtime::LoadedXFeat,
};
use image::imageops::{FilterType, colorops::grayscale};
use lookahead::{
    advance_case_timing_and_continuity, ensure_lookahead_tiles, init_tracker_session,
    pace_if_realtime, poll_anchor_before_frame, spawn_corridor_workers,
};
use navigate_match::{
    CorridorTileStore, DEFAULT_WINDOW_TILES, HadamardProjector, PreparedKashinQuery, SparseFeatures,
};
use navigate_visual::LocalFrame;
use reporting::{RunSummaryParams, build_run_summary, record_frame_telemetry};
use serde_json::{Value, json};
use solve::{FrameSolveParams, solve_frame_pose, tracking_render_size};
use std::{fs, sync::Arc, time::Instant};

/// Localize and track sequential frames against a [`CorridorTileStore`] (`corridor-track`).
///
/// # Errors
///
/// Returns [`PipelineError`] when bank verification, masking, matching, or report writing fails.
pub fn run_corridor_track(opts: &CorridorTrackOptions) -> Result<Value, PipelineError> {
    let surface = Arc::new(Surface::from_catalog(&opts.catalog_path)?);
    let local_frame = LocalFrame::wgs84_enu(surface.origin_lat, surface.origin_lon).ok();
    let catalog_digest = raw_file_sha256(&opts.catalog_path)?;
    let xfeat = LoadedXFeat::load(&opts.xfeat_weights_path, &opts.xfeat_runtime)?;
    let lighterglue = opts
        .lighterglue_weights_path
        .as_deref()
        .map(|p| load_lighterglue_model(p, xfeat.device()))
        .transpose()?;
    let store = Arc::new(CorridorTileStore::open(
        &opts.corridor_bank_path,
        DEFAULT_WINDOW_TILES,
    )?);
    if store.catalog_sha256 != catalog_digest {
        return Err(PipelineError::Invalid(
            "corridor bank catalog_sha256 does not match --catalog".into(),
        ));
    }
    check_encoder_contract("corridor bank", &store.encoder_contract, &xfeat)?;

    let search_tier_matcher = load_search_tier_matcher(
        opts.dinov2_weights_path.as_deref(),
        opts.place_index_path.as_deref(),
        Arc::clone(&store),
        xfeat.device(),
    )?;

    let seg_device = if opts.background_refresh {
        burn::tensor::Device::flex()
    } else {
        xfeat.device().clone()
    };
    let mask_interval = opts
        .mask_interval_s
        .map_or(opts.mask_interval, |s| ((s * 30.0).round() as usize).max(1));
    let mut masker = AmortizedMasker::new(
        opts.seg_model_dir.as_deref(),
        &opts.mask_policy,
        6,
        mask_interval.max(1),
        &seg_device,
    )?;
    if opts.background_refresh {
        masker = masker.with_background_refresh()?;
    }
    fs::create_dir_all(&opts.output_dir).map_err(|source| PipelineError::Io {
        path: opts.output_dir.clone(),
        source,
    })?;
    write_json(
        &opts.output_dir.join("run.json"),
        &json!({ "catalog_sha256": surface.catalog_sha256 }),
    )?;

    let projector = HadamardProjector::new(store.seed);
    let cases = load_profile_cases(&opts.profile_path)?;
    let workers = spawn_corridor_workers(
        &surface,
        &store,
        projector,
        &xfeat,
        lighterglue.as_ref(),
        search_tier_matcher.as_ref(),
        &cases,
        opts,
    )?;
    let sync_gpu_renderer = workers.sync_gpu_renderer;
    let tile_params = workers.tile_params;
    let mut background_anchor = workers.background_anchor;
    let mut tile_builder = workers.tile_builder;
    let shader_warmup_ms = workers.shader_warmup_ms;

    let mut latencies = StageLatencies::default();
    let mut tile_build_ms = Vec::new();
    let (mut sync_tiles_built, mut anchor_results) = (0_usize, Vec::new());
    let (kp_limit, cell_m) = (
        opts.query_keypoints.clamp(16, 4096),
        opts.map_cell_m.clamp(50.0, 20_000.0),
    );
    let min_trk_inliers = opts.min_track_inliers.max(24);
    let mut state = TrackState::new();
    let mut focal_estimator = FocalLengthEstimator::default();
    init_tracker_session(&mut state, &cases, opts, &focal_estimator, cell_m)?;

    let (mut evaluated, mut accepted, mut acquiring_frames) = (0_usize, 0_usize, 0_usize);
    let (mut warm_tracked_frames, mut independent_fixes, mut seg_refreshes) =
        (0_usize, 0_usize, 0_usize);
    let mut telemetry = Vec::new();
    let mut pace_origin: Option<(f64, Instant)> = None;
    let mut vo_accum_dt_s = 0.0_f64;

    for raw_case in &cases {
        if let Some(allow) = &opts.only_cases
            && !allow.contains(&raw_case.id)
        {
            continue;
        }
        evaluated += 1;
        pace_if_realtime(opts.pace_realtime, raw_case, &mut pace_origin);
        let polled_outcome = poll_anchor_before_frame(
            &mut background_anchor,
            &mut focal_estimator,
            opts.pace_realtime,
            evaluated,
            raw_case,
            &state,
        )?;
        let case_t0 = Instant::now();
        let case = apply_case_overrides(raw_case, opts);

        if let Some(lookahead) = estimate_lookahead_enu(&surface, &state, &case) {
            ensure_lookahead_tiles(
                &surface,
                &store,
                &mut tile_builder,
                &sync_gpu_renderer,
                &xfeat,
                &projector,
                tile_params,
                lookahead,
                &mut tile_build_ms,
                &mut sync_tiles_built,
            )?;
        }

        let query_rgb = load_case_rgb(&opts.dataset_dir, &case.image)?;
        let disk_mask = (!case.valid_mask.is_empty())
            .then(|| load_case_mask(&opts.dataset_dir, &case.valid_mask, case.size))
            .transpose()?;

        let k = estimate_intrinsics_with_estimator(&case, &focal_estimator);
        let render_size = tracking_render_size(case.size);
        let sx = f64::from(render_size[0]) / f64::from(case.size[0].max(1));
        let sy = f64::from(render_size[1]) / f64::from(case.size[1].max(1));
        let k_render = scale_k(&k, sx, sy);

        let qry_small = image::imageops::resize(
            &query_rgb,
            render_size[0],
            render_size[1],
            FilterType::Triangle,
        );
        let fallback_small = disk_mask.as_ref().map(|m| {
            image::imageops::resize(m, render_size[0], render_size[1], FilterType::Nearest)
        });
        let est_rot_for_mask = state
            .prev_rot
            .or(state.vo_rot)
            .map(|r| state.prev_rot_step * r);
        if evaluated == 1 && state.active_tracks.is_empty() && masker.has_segformer() {
            match masker.fast_initial_pass(&qry_small, est_rot_for_mask.as_ref()) {
                Ok(Some(ms)) => {
                    seg_refreshes += 1;
                    latencies.seg_refresh_ms.push(ms);
                }
                Ok(None) => {}
                Err(e) => tracing::warn!("frame 1 fast Segformer pass failed: {e}"),
            }
            if opts.pace_realtime
                && let Some((_, ref mut wall0)) = pace_origin
            {
                *wall0 = Instant::now();
            }
        }
        let anchor_busy = polled_outcome.is_none()
            && background_anchor
                .as_ref()
                .is_some_and(BackgroundAnchor::is_in_flight);
        let allow_seg =
            !(state.active_tracks.is_empty() && masker.has_usable_key(render_size) && anchor_busy);
        let qry_gray = grayscale(&qry_small);
        let (mask_small, mask_tel): (_, MaskFrameTelemetry) = masker
            .mask_frame_with_luma_controlled(
                &qry_small,
                &qry_gray,
                fallback_small.as_ref(),
                est_rot_for_mask.as_ref(),
                &k_render,
                allow_seg,
            )?;
        seg_refreshes += usize::from(mask_tel.segformer_refreshed);
        latencies.seg_refresh_ms.extend(mask_tel.segformer_ms);
        let cold_paced_vo = opts.pace_realtime
            && anchor_busy
            && state.active_tracks.is_empty()
            && state.prev_prepared.is_some();
        let skip_cold_xfeat = cold_paced_vo && !evaluated.is_multiple_of(6);
        let qry_kp_cap = if mask_tel.valid_fraction <= 0.0 || skip_cold_xfeat {
            0
        } else if cold_paced_vo {
            kp_limit.min(256)
        } else {
            kp_limit
        };
        let (qry_feat, query_xfeat_ms) = if qry_kp_cap > 0 {
            let t_xfeat = Instant::now();
            let feat = xfeat.extract_masked(&qry_gray, Some(&mask_small), qry_kp_cap)?;
            (feat, t_xfeat.elapsed().as_secs_f64() * 1000.0)
        } else {
            (SparseFeatures::empty(), 0.0)
        };
        let prepared = PreparedKashinQuery::from_features(&projector, &qry_feat)?;

        let (obs_time_s, dt_s, discontinuous) = advance_case_timing_and_continuity(
            &surface,
            &mut state,
            &mut focal_estimator,
            &case,
            &opts.dynamics_profile,
        );
        if discontinuous {
            vo_accum_dt_s = 0.0;
        }
        state
            .observe_frame(obs_time_s, dt_s, &case.image_sha256, &case.id)
            .map_err(PipelineError::Invalid)?;

        let (prior_rot_hint, mut nominal_agl_m) =
            prior_odometry_hint(&surface, &state, &case, &masker, &k_render);
        if skip_cold_xfeat {
            vo_accum_dt_s += dt_s.max(0.0);
            state.record_skipped_odometry_step(dt_s);
        } else {
            let vo_dt_s = dt_s + std::mem::take(&mut vo_accum_dt_s);
            record_odometry_step(
                &mut state,
                &prepared,
                &k_render,
                prior_rot_hint,
                nominal_agl_m,
                vo_dt_s,
                &opts.dynamics_profile,
            );
        }

        let arrived_anchor = polled_outcome.and_then(|outcome| {
            let target_id = (!discontinuous).then_some(case.id.as_str());
            let arr = AnchorArrival::apply_outcome(
                &mut state,
                &mut independent_fixes,
                &mut latencies,
                &mut anchor_results,
                &surface.catalog_sha256,
                (cell_m, dt_s),
                local_frame,
                &mut focal_estimator,
                opts.dynamics_profile,
                outcome,
                target_id,
            );
            if discontinuous {
                state.reset_discontinuous_track();
            }
            arr
        });

        let had_arrived_anchor = arrived_anchor.is_some();
        let k_after_arrival = estimate_intrinsics_with_estimator(&case, &focal_estimator);
        let k_render_after_arrival = scale_k(&k_after_arrival, sx, sy);
        if had_arrived_anchor {
            (_, nominal_agl_m) =
                prior_odometry_hint(&surface, &state, &case, &masker, &k_render_after_arrival);
        }

        let solve_params = FrameSolveParams {
            surface: &surface,
            store: &store,
            projector: &projector,
            sync_gpu_renderer: &sync_gpu_renderer,
            xfeat: &xfeat,
            lighterglue: lighterglue.as_ref(),
            search_tier_matcher: search_tier_matcher.as_deref(),
            opts,
            case: &case,
            qry_feat: &qry_feat,
            prepared: &prepared,
            mask_small: &mask_small,
            k: k_after_arrival,
            k_render: k_render_after_arrival,
            render_size,
            masker: &masker,
            local_frame,
            nominal_agl_m,
            dt_s,
            cell_m,
            min_trk_inliers,
            arrived_anchor,
        };
        let outcome = solve_frame_pose(
            solve_params,
            &mut state,
            &mut focal_estimator,
            &mut background_anchor,
            &mut independent_fixes,
            &mut latencies,
            &mut warm_tracked_frames,
            &mut acquiring_frames,
        )?;
        let (best, admission, lg_layers, polish_rays, solve_ms, attempts_log) = (
            outcome.best,
            outcome.admission,
            outcome.lg_layers,
            outcome.polish_rays,
            outcome.solve_ms,
            outcome.attempts_log,
        );

        if let Some((_, _, ref cand)) = best
            && cand.score.geometry_accepted
        {
            masker.note_solved_rotation(cand.rotation);
            if outcome.used_arrived_anchor {
                state.prev_prepared = Some(prepared);
            } else if had_arrived_anchor {
                state.snap_pose_preserving_velocity(cand, prepared);
            } else {
                update_motion_state(&mut state, cand, prepared, dt_s, &opts.dynamics_profile);
            }
        } else if !skip_cold_xfeat {
            advance_unsolved_prepared(&mut state, prepared, dt_s, &opts.dynamics_profile);
        }
        let filter_snap = state.filter_snapshot(local_frame);

        let total_case_ms = case_t0.elapsed().as_secs_f64() * 1000.0;
        if admission.tracking_mode == "warm_track" {
            latencies
                .warm_track_ms
                .push(mask_tel.mask_ms + query_xfeat_ms + solve_ms);
        }
        let is_accepted = best
            .as_ref()
            .is_some_and(|(_, _, c)| c.score.geometry_accepted);
        record_frame_telemetry(
            &mut telemetry,
            &case.id,
            is_accepted,
            xfeat.runtime_name(),
            &admission,
            &filter_snap,
            state.active_tracks.len(),
            store.cached_tile_count(),
            store.resident_window_bytes(),
            qry_feat.len(),
            lg_layers,
            polish_rays,
            &mask_tel,
            query_xfeat_ms,
            solve_ms,
            total_case_ms,
        );

        write_corridor_report(
            &opts.output_dir,
            &surface,
            &case.id,
            &admission,
            &filter_snap,
            attempts_log,
            best,
            &mut accepted,
        )?;
    }

    if let Some(mut worker) = background_anchor.take() {
        if let Some(outcome) = worker.wait_for_shutdown()? {
            let default_dt = 1.0_f64 / 30.0;
            let _ = AnchorArrival::apply_outcome(
                &mut state,
                &mut independent_fixes,
                &mut latencies,
                &mut anchor_results,
                &surface.catalog_sha256,
                (cell_m, default_dt),
                local_frame,
                &mut focal_estimator,
                opts.dynamics_profile,
                outcome,
                None,
            );
        }
        drop(worker);
    }
    let final_filter_snap = state.filter_snapshot(local_frame);
    let mut tiles_built_live = sync_tiles_built;
    if let Some(mut tb) = tile_builder.take() {
        tb.poll(&mut tile_build_ms)?;
        tiles_built_live += tb.built_tiles();
        drop(tb);
    }
    drop(masker);

    let summary = build_run_summary(RunSummaryParams {
        evaluated,
        accepted,
        acquiring_frames,
        warm_tracked_frames,
        independent_fixes,
        seg_refreshes,
        runtime_name: xfeat.runtime_name(),
        indexed_tile_count: store.indexed_tile_count(),
        cached_tile_count: store.cached_tile_count(),
        resident_window_bytes: store.resident_window_bytes(),
        tiles_built_live,
        tile_build_ms: &tile_build_ms,
        shader_warmup_ms,
        output_dir: &opts.output_dir.display().to_string(),
        telemetry,
        background_refresh: opts.background_refresh,
        anchor_results,
        latencies: &latencies,
        filter_snap: &final_filter_snap,
        locked_focal_px: focal_estimator.locked_focal_px,
    });
    Ok(summary)
}
