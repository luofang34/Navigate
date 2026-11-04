//! Thin CLI replay adapter routing `PhotoCase` datasets into [`FlightVisualEngine`].

pub(crate) mod inputs;
pub(crate) mod reporting;

pub(crate) use inputs::{
    apply_case_overrides, build_live_frame_input, build_search_opts, matcher_runtime_for_options,
    mount_profile_for_dynamics,
};

use super::{CorridorTrackOptions, latency::StageLatencies};
use crate::{
    PipelineError,
    json_file::write_json,
    stages::{
        estimate_intrinsics_with_estimator, focal::FocalLengthEstimator, load_case_mask,
        load_case_rgb, load_profile_cases, scale_k,
    },
    testset::PhotoCase,
};
use inputs::{advance_monotonic_clock, pace_if_realtime};
use navigate_match::DEFAULT_WINDOW_TILES;
use navigate_visual_flight::{
    ExecutionPolicy, FlightEngineConfig, FlightEngineResourcePaths, FlightEngineResources,
    FlightVisualEngine, TrackingMode, tracking_render_size,
};
use reporting::{
    RunSummaryParams, build_run_summary, emit_corridor_case_report, format_anchor_arrival_json,
    record_frame_telemetry,
};
use serde_json::{Value, json};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

/// Localize and track sequential frames against a `CorridorTileStore` (`corridor-track`).
///
/// # Errors
///
/// Returns [`PipelineError`] when bank verification, masking, matching, or report writing fails.
pub fn run_corridor_track(opts: &CorridorTrackOptions) -> Result<Value, PipelineError> {
    let matcher_rt = matcher_runtime_for_options(
        opts.lighterglue_weights_path.as_deref(),
        &opts.xfeat_runtime,
    );
    let resources = Arc::new(FlightEngineResources::open(&FlightEngineResourcePaths {
        catalog_path: &opts.catalog_path,
        corridor_bank_path: &opts.corridor_bank_path,
        xfeat_weights_path: &opts.xfeat_weights_path,
        xfeat_runtime: &opts.xfeat_runtime,
        matcher_weights_path: opts.lighterglue_weights_path.as_deref(),
        matcher_runtime: &matcher_rt,
        dinov2_weights_path: opts.dinov2_weights_path.as_deref(),
        place_index_path: opts.place_index_path.as_deref(),
        max_resident_tiles: DEFAULT_WINDOW_TILES,
        prewarm_gpu: false,
    })?);
    run_corridor_track_with_resources(opts, None, resources, Some(&opts.output_dir))
}

pub(crate) fn run_corridor_track_with_resources(
    opts: &CorridorTrackOptions,
    cases_override: Option<&[PhotoCase]>,
    resources: Arc<FlightEngineResources>,
    persist_dir: Option<&Path>,
) -> Result<Value, PipelineError> {
    let loaded_cases;
    let cases: &[PhotoCase] = if let Some(explicit) = cases_override {
        explicit
    } else {
        loaded_cases = load_profile_cases(&opts.profile_path)?;
        &loaded_cases
    };
    let effective_cases = cases
        .iter()
        .filter(|c| {
            opts.only_cases
                .as_ref()
                .is_none_or(|allow| allow.contains(&c.id))
        })
        .count();
    let use_bg = opts.background_refresh && effective_cases > 1;

    if let Some(dir) = persist_dir {
        fs::create_dir_all(dir).map_err(|source| PipelineError::Io {
            path: dir.to_path_buf(),
            source,
        })?;
        write_json(
            &dir.join("run.json"),
            &json!({ "catalog_sha256": resources.surface.catalog_sha256 }),
        )?;
    }

    let first_eff = cases
        .iter()
        .find(|c| {
            opts.only_cases
                .as_ref()
                .is_none_or(|allow| allow.contains(&c.id))
        })
        .or_else(|| cases.first())
        .map(|c| apply_case_overrides(c, opts));
    let focal_est = FocalLengthEstimator::default();
    let init_rs = first_eff
        .as_ref()
        .map_or([800, 600], |c| tracking_render_size(c.size));
    let init_k = first_eff.as_ref().map_or_else(
        || nalgebra::Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0),
        |c| {
            let kf = estimate_intrinsics_with_estimator(c, &focal_est);
            let sx = f64::from(init_rs[0]) / f64::from(c.size[0].max(1));
            let sy = f64::from(init_rs[1]) / f64::from(c.size[1].max(1));
            scale_k(&kf, sx, sy)
        },
    );
    let init_prior_pose = first_eff
        .as_ref()
        .and_then(|c| c.prior.as_ref())
        .and_then(|p| {
            let pos = p.position.as_ref()?;
            let dem = resources
                .surface
                .sample_dem(pos.latitude, pos.longitude)
                .unwrap_or(0.0);
            let alt = pos.altitude_m.unwrap_or(dem + 120.0).max(dem + 25.0);
            let eye = resources.surface.local(pos.latitude, pos.longitude, alt);
            let hdg = p.direction_deg.unwrap_or(0.0);
            let rot =
                resources
                    .surface
                    .camera_rotation(pos.latitude, pos.longitude, hdg, 55.0, 0.0);
            Some((eye, rot))
        });

    let mask_interval = opts
        .mask_interval_s
        .map_or(opts.mask_interval, |s| ((s * 30.0).round() as usize).max(1));
    let policy = if opts.pace_realtime {
        ExecutionPolicy::LiveFlight
    } else {
        ExecutionPolicy::DeterministicReplay {
            anchor_budget_ms: 200,
            tile_budget_ms: 150,
        }
    };
    let need_live_tiles = init_prior_pose.is_none_or(|(eye, _)| {
        !resources
            .store
            .has_tile(resources.store.tile_key_at(eye[0], eye[1]))
    });
    let config = FlightEngineConfig {
        policy,
        dynamics_profile: opts.dynamics_profile,
        search_opts: build_search_opts(opts),
        seg_model_dir: opts.seg_model_dir.clone(),
        mask_policy: opts.mask_policy.clone(),
        mask_interval_frames: mask_interval.max(1),
        anchor_interval_frames: opts.anchor_interval,
        anchor_interval_s: opts.anchor_interval_s,
        anchor_on_cell_boundary: opts.anchor_on_cell_boundary,
        min_track_inliers: opts.min_track_inliers.max(24),
        map_cell_m: opts.map_cell_m.clamp(50.0, 20_000.0),
        background_workers: use_bg,
        enable_live_tile_synth: true,
        prewarm_tile_gpu: need_live_tiles,
        init_render_size: init_rs,
        init_k_render: init_k,
        init_prior_pose,
        attitude_budget: None,
    };

    let mut engine = FlightVisualEngine::new(Arc::clone(&resources), config)?;
    let mount = mount_profile_for_dynamics(&opts.dynamics_profile);

    let mut latencies = StageLatencies::default();
    let mut tile_build_ms = Vec::new();
    let mut anchor_results = Vec::new();
    let mut telemetry = Vec::new();
    let (mut evaluated, mut accepted, mut acquiring_frames) = (0_usize, 0_usize, 0_usize);
    let (mut warm_tracked_frames, mut seg_refreshes) = (0_usize, 0_usize);
    let mut pace_origin: Option<(f64, Instant)> = None;
    let mut prev_raw_time_s: Option<f64> = None;
    let mut cur_monotonic_ns = 1_000_000_000_u64;

    for raw_case in cases {
        if let Some(allow) = &opts.only_cases
            && !allow.contains(&raw_case.id)
        {
            continue;
        }
        evaluated += 1;
        pace_if_realtime(opts.pace_realtime, raw_case, &mut pace_origin);
        let case = apply_case_overrides(raw_case, opts);
        let query_rgb = load_case_rgb(&opts.dataset_dir, &case.image)?;
        let disk_mask = (!case.valid_mask.is_empty())
            .then(|| load_case_mask(&opts.dataset_dir, &case.valid_mask, case.size))
            .transpose()?;
        let stamp_ns = advance_monotonic_clock(
            &case,
            &mut prev_raw_time_s,
            &mut cur_monotonic_ns,
            evaluated == 1,
        );
        let live_input = build_live_frame_input(
            &opts.dataset_dir,
            &case,
            &query_rgb,
            disk_mask.as_ref(),
            mount,
            (evaluated - 1) as u32,
            stamp_ns,
            &focal_est,
        );

        let (outcome, diag) = engine.process_frame_with_diagnostics(&live_input, &case.id)?;
        let tel = &outcome.telemetry;
        if evaluated == 1
            && opts.pace_realtime
            && tel.segformer_refreshed
            && let Some((_, ref mut wall0)) = pace_origin
        {
            *wall0 = Instant::now();
        }
        seg_refreshes += usize::from(tel.segformer_refreshed);
        latencies.seg_refresh_ms.extend(tel.segformer_ms);
        tile_build_ms.extend_from_slice(&diag.tile_build_ms);
        for arr in &diag.anchor_arrivals {
            latencies.anchor_ms.push(arr.anchor_ms);
            anchor_results.push(format_anchor_arrival_json(arr));
        }
        match tel.tracking_mode {
            TrackingMode::WarmTrack => {
                warm_tracked_frames += 1;
                latencies
                    .warm_track_ms
                    .push(tel.mask_ms + tel.query_xfeat_ms + tel.solve_ms);
            }
            TrackingMode::AcquiringAnchor => {
                acquiring_frames += 1;
            }
            TrackingMode::CorridorAnchor | TrackingMode::Unresolved if !use_bg => {
                latencies.anchor_ms.push(tel.solve_ms);
            }
            _ => {}
        }

        record_frame_telemetry(
            &mut telemetry,
            &case.id,
            resources.xfeat.runtime_name(),
            tel,
            &diag.filter_snapshot,
        );
        let _ = emit_corridor_case_report(
            persist_dir,
            &resources.surface,
            &case.id,
            tel,
            outcome.anchor_confidence,
            &diag,
            &mut accepted,
        )?;
    }

    let (drained_arrivals, drained_tile_ms) = engine.shutdown_drain()?;
    for arr in &drained_arrivals {
        latencies.anchor_ms.push(arr.anchor_ms);
        anchor_results.push(format_anchor_arrival_json(arr));
    }
    tile_build_ms.extend(drained_tile_ms);
    let final_filter_snap = engine.filter_snapshot();
    let out_label = persist_dir.map_or_else(
        || PathBuf::new().display().to_string(),
        |d| d.display().to_string(),
    );

    Ok(build_run_summary(RunSummaryParams {
        evaluated,
        accepted,
        acquiring_frames,
        warm_tracked_frames,
        independent_fixes: engine.independent_fixes(),
        seg_refreshes,
        runtime_name: resources.xfeat.runtime_name(),
        indexed_tile_count: resources.store.indexed_tile_count(),
        cached_tile_count: resources.store.cached_tile_count(),
        resident_window_bytes: resources.store.resident_window_bytes(),
        shared_gpu_texture_bytes: resources.shared_gpu_texture_bytes(),
        tiles_built_live: engine.tiles_built_live(),
        tile_build_ms: &tile_build_ms,
        shader_warmup_ms: engine.shader_warmup_ms(),
        output_dir: &out_label,
        telemetry,
        background_refresh: opts.background_refresh,
        anchor_results,
        latencies: &latencies,
        filter_snap: &final_filter_snap,
        locked_focal_px: engine.locked_focal_px(),
    }))
}
