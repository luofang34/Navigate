//! Two-tier `25-30 Hz` temporal corridor tracker, windowed tile streaming, and [`MapCell`] EKF gate.

use super::{
    CorridorTrackOptions,
    admission::{
        FusionAdmission, apply_case_overrides, classify_anchor_cell, estimate_lookahead_enu,
        prior_odometry_hint, seed_tracks, top_up_tracks, write_corridor_report,
    },
    anchor::acquire_corridor_anchor,
    anchor_worker::{AnchorArrival, AnchorContext, AnchorJob, BackgroundAnchor},
    latency::{StageLatencies, percentile},
    masking::{AmortizedMasker, MaskFrameTelemetry},
    motion::{
        TrackState, advance_unsolved_prepared, predict_interframe_pose, record_odometry_step,
        try_warm_track, update_motion_state,
    },
    tile_builder::{
        BackgroundTileBuilder, TileBuilderContext, TileSynthParams, synthesize_corridor_tiles,
    },
};
use crate::{
    PipelineError,
    realtime::helpers::{check_encoder_contract, load_lighterglue_model, raw_file_sha256},
    stages::{
        estimate_intrinsics, load_case_mask, load_case_rgb, load_profile_cases, scale_k,
        scaled_size,
    },
    surface::{Surface, SurfaceGpuRenderer},
    util::write_json,
    xfeat_runtime::LoadedXFeat,
};
use image::imageops::{FilterType, colorops::grayscale};
use navigate_match::{
    ActiveTrackSet, CorridorTileStore, DEFAULT_WINDOW_TILES, HadamardProjector, PreparedKashinQuery,
};
use serde_json::{Value, json};
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
    let mut masker = AmortizedMasker::new(
        opts.seg_model_dir.as_deref(),
        &opts.mask_policy,
        6,
        opts.mask_interval.max(1),
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
        .then(|| SurfaceGpuRenderer::try_new(&surface, &init_k, [800, 600]))
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
    let mut sync_tiles_built = 0_usize;
    let mut anchor_results = Vec::new();
    let kp_limit = opts.query_keypoints.clamp(16, 4096);
    let cell_m = opts.map_cell_m.clamp(50.0, 20_000.0);
    let min_trk_inliers = opts.min_track_inliers.max(24);
    let mut state = TrackState::new();
    let mut evaluated = 0_usize;
    let mut accepted = 0_usize;
    let mut acquiring_frames = 0_usize;
    let mut warm_tracked_frames = 0_usize;
    let mut independent_fixes = 0_usize;
    let mut seg_refreshes = 0_usize;
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
            if let Some(tb) = tile_builder.as_mut() {
                tb.poll(&mut tile_build_ms)?;
                tb.request_neighborhood(&surface, &store, lookahead)?;
            } else {
                let ck = store.tile_key_at(lookahead[0], lookahead[1]);
                if !store.has_tile(ck) {
                    let t_tb = Instant::now();
                    let entries = synthesize_corridor_tiles(
                        &surface,
                        Some(&sync_gpu_renderer),
                        &xfeat,
                        &projector,
                        ck,
                        tile_params,
                    )?;
                    if !entries.is_empty() {
                        for entry in entries {
                            store.insert_tile(entry, true)?;
                            sync_tiles_built += 1;
                        }
                        tile_build_ms.push(t_tb.elapsed().as_secs_f64() * 1000.0);
                    }
                }
            }
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
        let qry_kp_floor = 256_usize.min(kp_limit);
        let qry_kp_cap = (((kp_limit as f64) * mask_tel.valid_fraction.clamp(0.25, 1.0)).round()
            as usize)
            .clamp(qry_kp_floor, kp_limit);

        let t_xfeat = Instant::now();
        let qry_feat = xfeat.extract_masked(&qry_gray, Some(&mask_small), qry_kp_cap)?;
        let query_xfeat_ms = t_xfeat.elapsed().as_secs_f64() * 1000.0;
        let prepared = PreparedKashinQuery::from_features(&projector, &qry_feat)?;

        let (prior_rot_hint, nominal_agl_m) =
            prior_odometry_hint(&surface, &case, &masker, &k_render);
        record_odometry_step(
            &mut state,
            &prepared,
            &k_render,
            prior_rot_hint,
            nominal_agl_m,
        );

        let mut arrived_pose = None;
        if let Some(worker) = background_anchor.as_mut()
            && let Some(outcome) = worker.poll()?
        {
            arrived_pose = AnchorArrival {
                state: &mut state,
                independent_fixes: &mut independent_fixes,
                latencies: &mut latencies,
                results: &mut anchor_results,
                catalog_sha256: &surface.catalog_sha256,
                cell_m,
            }
            .apply(outcome, Some(&case.id));
        }

        let t_solve = Instant::now();
        let mut attempts_log = Vec::new();
        let mut lg_layers = 0_usize;
        let mut polish_rays = 0_usize;
        let pred_pose =
            arrived_pose.or_else(|| predict_interframe_pose(&state, &prepared, &k_render));
        let mut force_anchor =
            opts.anchor_interval > 0 && state.frames_since_anchor >= opts.anchor_interval;
        if force_anchor
            && !state.active_tracks.is_empty()
            && let Some(worker) = background_anchor.as_mut()
        {
            let seq = state.frame_seq;
            let _ = worker.submit_if_idle(|| AnchorJob {
                sent_frame_seq: seq,
                case: case.clone(),
                qry_feat: qry_feat.clone(),
                prepared: prepared.clone(),
                k,
                k_render,
                render_size,
                pred_pose,
                live_attitude: masker.horizon_attitude(&k_render),
            })?;
            force_anchor = false;
        }

        let mut solved_warm = None;
        if !state.active_tracks.is_empty()
            && !force_anchor
            && let Some(pred) = pred_pose
        {
            solved_warm = try_warm_track(
                &state.active_tracks,
                &prepared,
                pred,
                &k_render,
                render_size,
                min_trk_inliers,
                &mut attempts_log,
            )?;
        }

        let (best, admission) = if let Some(cand) = solved_warm {
            warm_tracked_frames += 1;
            state.frames_since_anchor += 1;
            if cand.score.inliers < 260 {
                top_up_tracks(
                    &mut state.active_tracks,
                    &surface,
                    &prepared,
                    &qry_feat,
                    &cand,
                    render_size,
                );
            }
            (Some((0, 0, cand)), FusionAdmission::warm_track())
        } else if let Some(worker) = background_anchor.as_mut() {
            let seq = state.frame_seq;
            let _ = worker.submit_if_idle(|| AnchorJob {
                sent_frame_seq: seq,
                case: case.clone(),
                qry_feat: qry_feat.clone(),
                prepared: prepared.clone(),
                k,
                k_render,
                render_size,
                pred_pose,
                live_attitude: masker.horizon_attitude(&k_render),
            })?;
            acquiring_frames += 1;
            (None, FusionAdmission::acquiring_anchor())
        } else {
            let t_anchor = Instant::now();
            let acq = acquire_corridor_anchor(
                (&surface, &store, &projector),
                Some(&sync_gpu_renderer),
                (&xfeat, lighterglue.as_ref()),
                (&case, opts),
                (&qry_feat, &prepared),
                (&k, &k_render, render_size),
                (pred_pose, masker.horizon_attitude(&k_render)),
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
                state.active_tracks =
                    seed_tracks(&surface, &prepared, &qry_feat, cand, &acq.best_view);
                let adm = classify_anchor_cell(
                    &mut state.visited_cells,
                    &mut independent_fixes,
                    &surface.catalog_sha256,
                    cand.eye,
                    cell_m,
                    &case.id,
                );
                (acq.best, adm)
            } else {
                state.active_tracks = ActiveTrackSet::empty();
                (acq.best, FusionAdmission::unresolved())
            }
        };
        let solve_ms = t_solve.elapsed().as_secs_f64() * 1000.0;

        if let Some((_, _, ref cand)) = best
            && cand.score.geometry_accepted
        {
            masker.note_solved_rotation(cand.rotation);
            update_motion_state(&mut state, cand, prepared);
        } else {
            advance_unsolved_prepared(&mut state, prepared);
        }

        let total_case_ms = case_t0.elapsed().as_secs_f64() * 1000.0;
        if admission.tracking_mode == "warm_track" {
            latencies.warm_track_ms.push(total_case_ms);
        }
        let is_accepted = best
            .as_ref()
            .is_some_and(|(_, _, c)| c.score.geometry_accepted);
        telemetry.push(json!({
            "id": case.id,
            "accepted": is_accepted,
            "xfeat_runtime": xfeat.runtime_name(),
            "tracking_mode": admission.tracking_mode,
            "evidence_kind": admission.evidence_kind,
            "independent_ekf_fix": admission.independent_ekf_fix,
            "fusion_eligibility": admission.fusion_eligibility,
            "map_cell": admission.map_cell,
            "active_tracks": state.active_tracks.len(),
            "cached_tiles": store.cached_tile_count(),
            "resident_window_bytes": store.resident_window_bytes(),
            "query_keypoints": qry_feat.len(),
            "lighterglue_layers": lg_layers,
            "polish_rays_cast": polish_rays,
            "segformer_refreshed": mask_tel.segformer_refreshed,
            "segformer_submitted": mask_tel.segformer_submitted,
            "segformer_ms": mask_tel.segformer_ms,
            "valid_mask_fraction": mask_tel.valid_fraction,
            "propeller_pixels": mask_tel.propeller_pixels,
            "mask_ms": mask_tel.mask_ms,
            "query_xfeat_ms": query_xfeat_ms,
            "solve_ms": solve_ms,
            "total_case_ms": total_case_ms,
        }));

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
            let _ = AnchorArrival {
                state: &mut state,
                independent_fixes: &mut independent_fixes,
                latencies: &mut latencies,
                results: &mut anchor_results,
                catalog_sha256: &surface.catalog_sha256,
                cell_m,
            }
            .apply(outcome, None);
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

    let mut summary = json!({
        "evaluated": evaluated,
        "accepted": accepted,
        "acquiring_anchor_frames": acquiring_frames,
        "warm_tracked_frames": warm_tracked_frames,
        "independent_ekf_fixes": independent_fixes,
        "segformer_refreshes": seg_refreshes,
        "xfeat_runtime": xfeat.runtime_name(),
        "corridor_tiles": store.indexed_tile_count(),
        "cached_tiles": store.cached_tile_count(),
        "resident_window_bytes": store.resident_window_bytes(),
        "tiles_built_live": tiles_built_live,
        "tile_build_p50_ms": percentile(&tile_build_ms, 50.0),
        "shader_warmup_ms": shader_warmup_ms,
        "output": opts.output_dir.display().to_string(),
        "telemetry": telemetry,
        "background_refresh": opts.background_refresh,
        "anchor_results": anchor_results,
    });
    if let Some(map) = summary.as_object_mut() {
        latencies.write_summary(map);
    }
    Ok(summary)
}
