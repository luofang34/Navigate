//! Two-tier `25-30 Hz` temporal corridor tracker, windowed tile streaming, and [`MapCell`] EKF gate.

use super::{
    CorridorTrackOptions,
    admission::{
        apply_case_overrides, estimate_lookahead_enu, prior_odometry_hint, write_corridor_report,
    },
    anchor_worker::{AnchorArrival, AnchorContext, BackgroundAnchor},
    kinematics::compute_dt_s,
    latency::StageLatencies,
    masking::{AmortizedMasker, MaskFrameTelemetry},
    motion::{TrackState, advance_unsolved_prepared, record_odometry_step, update_motion_state},
    tile_builder::{BackgroundTileBuilder, TileBuilderContext, TileSynthParams},
};

mod lookahead;
mod reporting;
mod solve;
use crate::{
    PipelineError,
    realtime::helpers::{check_encoder_contract, load_lighterglue_model, raw_file_sha256},
    stages::{
        estimate_intrinsics, load_case_mask, load_case_rgb, load_profile_cases, scale_k,
        scaled_size,
    },
    surface::gpu::GpuQueueRole,
    surface::{Surface, SurfaceGpuRenderer},
    util::write_json,
    xfeat_runtime::LoadedXFeat,
};
use image::imageops::{FilterType, colorops::grayscale};
use lookahead::ensure_lookahead_tiles;
use navigate_match::{
    CorridorTileStore, DEFAULT_WINDOW_TILES, HadamardProjector, PreparedKashinQuery, SparseFeatures,
};
use reporting::{RunSummaryParams, build_run_summary, record_frame_telemetry};
use serde_json::{Value, json};
use solve::{FrameSolveParams, solve_frame_pose};
use std::{
    fs,
    sync::{Arc, Mutex},
    time::Instant,
};

/// Localize and track sequential frames against a [`CorridorTileStore`] (`corridor-track`).
///
/// # Errors
///
/// Returns [`PipelineError`] when bank verification, masking, matching, or report writing fails.
pub fn run_corridor_track(opts: &CorridorTrackOptions) -> Result<Value, PipelineError> {
    let surface = Arc::new(Surface::from_catalog(&opts.catalog_path)?);
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
    let init_k = nalgebra::Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0);
    let t_warmup = Instant::now();
    let _ = xfeat.warmup(lighterglue.as_ref())?;
    let sync_gpu = (!opts.background_refresh && opts.polish_anchors)
        .then(|| {
            SurfaceGpuRenderer::try_new_with_role(
                &surface,
                &init_k,
                [800, 600],
                GpuQueueRole::ForegroundTracking,
            )
        })
        .flatten();
    let sync_gpu_renderer: Mutex<Option<SurfaceGpuRenderer>> = Mutex::new(sync_gpu);
    let tile_params = TileSynthParams {
        tile_m: f64::from(store.tile_size_m.max(10.0)),
        agl_m: 480.0,
        max_per_tile: 4096,
        stride: 4,
        include_obliques: !cfg!(debug_assertions),
    };
    let need_live_tiles = cases.iter().any(|c| {
        estimate_lookahead_enu(&surface, &TrackState::new(), &apply_case_overrides(c, opts))
            .is_some_and(|la| !store.has_tile(store.tile_key_at(la[0], la[1])))
    });
    let mut background_anchor = opts
        .background_refresh
        .then(|| {
            BackgroundAnchor::spawn(AnchorContext {
                surface: Arc::clone(&surface),
                store: Arc::clone(&store),
                projector,
                xfeat: xfeat.clone(),
                lighterglue: lighterglue.clone(),
                opts: opts.clone(),
            })
        })
        .transpose()?;
    let mut tile_builder = opts
        .background_refresh
        .then(|| {
            BackgroundTileBuilder::spawn(
                TileBuilderContext {
                    surface: Arc::clone(&surface),
                    store: Arc::clone(&store),
                    projector,
                    xfeat: xfeat.clone(),
                    params: tile_params,
                },
                need_live_tiles,
            )
        })
        .transpose()?;
    let shader_warmup_ms = t_warmup.elapsed().as_secs_f64() * 1000.0;

    let mut latencies = StageLatencies::default();
    let mut tile_build_ms = Vec::new();
    let (mut sync_tiles_built, mut anchor_results) = (0_usize, Vec::new());
    let (kp_limit, cell_m) = (
        opts.query_keypoints.clamp(16, 4096),
        opts.map_cell_m.clamp(50.0, 20_000.0),
    );
    let min_trk_inliers = opts.min_track_inliers.max(24);
    let mut state = TrackState::new();
    let (mut evaluated, mut accepted, mut acquiring_frames) = (0_usize, 0_usize, 0_usize);
    let (mut warm_tracked_frames, mut independent_fixes, mut seg_refreshes) =
        (0_usize, 0_usize, 0_usize);
    let mut telemetry = Vec::new();

    for raw_case in &cases {
        if let Some(allow) = &opts.only_cases
            && !allow.contains(&raw_case.id)
        {
            continue;
        }
        evaluated += 1;
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

        let k = estimate_intrinsics(&case);
        let render_size = scaled_size(case.size, 800);
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
        if evaluated == 1
            && state.active_tracks.is_empty()
            && masker.has_segformer()
            && let Err(e) = masker.fast_initial_pass(&qry_small, est_rot_for_mask.as_ref())
        {
            tracing::warn!("frame 1 fast Segformer pass failed: {e}");
        }
        let qry_gray = grayscale(&qry_small);
        let (mask_small, mask_tel): (_, MaskFrameTelemetry) = masker.mask_frame_with_luma(
            &qry_small,
            &qry_gray,
            fallback_small.as_ref(),
            est_rot_for_mask.as_ref(),
            &k_render,
        )?;
        seg_refreshes += usize::from(mask_tel.segformer_refreshed);
        latencies.seg_refresh_ms.extend(mask_tel.segformer_ms);
        let qry_kp_cap = if mask_tel.valid_fraction > 0.0 {
            kp_limit
        } else {
            0
        };
        let (qry_feat, query_xfeat_ms) = if qry_kp_cap > 0 {
            let t_xfeat = Instant::now();
            let feat = xfeat.extract_masked(&qry_gray, Some(&mask_small), qry_kp_cap)?;
            (feat, t_xfeat.elapsed().as_secs_f64() * 1000.0)
        } else {
            (SparseFeatures::empty(), 0.0)
        };
        let prepared = PreparedKashinQuery::from_features(&projector, &qry_feat)?;

        let dt_s = compute_dt_s(case.time_s, state.prev_time_s, 30.0);
        state.prev_time_s = case.time_s;
        state.time_since_anchor_s += dt_s;

        let (prior_rot_hint, nominal_agl_m) =
            prior_odometry_hint(&surface, &case, &masker, &k_render);
        record_odometry_step(
            &mut state,
            &prepared,
            &k_render,
            prior_rot_hint,
            nominal_agl_m,
            dt_s,
            &opts.dynamics_profile,
        );

        let mut arrived_pose = None;
        if let Some(worker) = background_anchor.as_mut()
            && let Some(outcome) = worker.poll()?
        {
            arrived_pose = AnchorArrival::apply_outcome(
                &mut state,
                &mut independent_fixes,
                &mut latencies,
                &mut anchor_results,
                &surface.catalog_sha256,
                cell_m,
                outcome,
                Some(&case.id),
            );
            if arrived_pose.is_some() {
                state.time_since_anchor_s = 0.0;
            }
        }

        let solve_params = FrameSolveParams {
            surface: &surface,
            store: &store,
            projector: &projector,
            sync_gpu_renderer: &sync_gpu_renderer,
            xfeat: &xfeat,
            lighterglue: lighterglue.as_ref(),
            opts,
            case: &case,
            qry_feat: &qry_feat,
            prepared: &prepared,
            k,
            k_render,
            render_size,
            masker: &masker,
            nominal_agl_m,
            dt_s,
            cell_m,
            min_trk_inliers,
            arrived_pose,
        };
        let outcome = solve_frame_pose(
            solve_params,
            &mut state,
            &mut background_anchor,
            &mut independent_fixes,
            &mut latencies,
            &mut warm_tracked_frames,
            &mut acquiring_frames,
        )?;
        let best = outcome.best;
        let admission = outcome.admission;
        let lg_layers = outcome.lg_layers;
        let polish_rays = outcome.polish_rays;
        let solve_ms = outcome.solve_ms;
        let attempts_log = outcome.attempts_log;

        if let Some((_, _, ref cand)) = best
            && cand.score.geometry_accepted
        {
            masker.note_solved_rotation(cand.rotation);
            update_motion_state(&mut state, cand, prepared, dt_s, &opts.dynamics_profile);
        } else {
            advance_unsolved_prepared(&mut state, prepared, dt_s, &opts.dynamics_profile);
        }

        let total_case_ms = case_t0.elapsed().as_secs_f64() * 1000.0;
        if admission.tracking_mode == "warm_track" {
            latencies.warm_track_ms.push(total_case_ms);
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
            attempts_log,
            best,
            &mut accepted,
        )?;
    }

    if let Some(mut worker) = background_anchor.take() {
        if let Some(outcome) = worker.wait_for_shutdown()? {
            let _ = AnchorArrival::apply_outcome(
                &mut state,
                &mut independent_fixes,
                &mut latencies,
                &mut anchor_results,
                &surface.catalog_sha256,
                cell_m,
                outcome,
                None,
            );
        }
        drop(worker);
    }
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
    });
    Ok(summary)
}
