//! Lookahead tile synthesis, frame timestamping, and discontinuity resets for corridor tracker.

use super::super::{
    CorridorTrackOptions,
    admission::{apply_case_overrides, estimate_lookahead_enu},
    anchor::multitier::SearchTierMatcher,
    anchor_worker::{AnchorContext, AnchorOutcome, BackgroundAnchor},
    kinematics::{VehicleDynamicsProfile, compute_dt_s},
    motion::TrackState,
    tile_builder::{
        BackgroundTileBuilder, TileBuilderContext, TileSynthParams, synthesize_corridor_tiles,
    },
};
use super::solve::tracking_render_size;
use crate::{
    PipelineError,
    realtime::helpers::rot_to_row_major,
    stages::{
        estimate_intrinsics_with_estimator, focal::FocalLengthEstimator,
        has_calibrated_focal_prior, scale_k,
    },
    surface::gpu::GpuQueueRole,
    surface::{Surface, SurfaceGpuRenderer},
    testset::PhotoCase,
    util::parse_iso8601_ms,
    xfeat_runtime::LoadedXFeat,
};
use navigate_match::{CorridorTileStore, HadamardProjector, LighterGlue, VirtualViewCamera};
use navigate_visual::CameraModel;
use std::{
    sync::{Arc, Mutex},
    time::Instant,
};

pub(super) struct CorridorRuntimeWorkers {
    pub sync_gpu_renderer: Mutex<Option<SurfaceGpuRenderer>>,
    pub tile_params: TileSynthParams,
    pub background_anchor: Option<BackgroundAnchor>,
    pub tile_builder: Option<BackgroundTileBuilder>,
    pub shader_warmup_ms: f64,
}

#[expect(
    clippy::too_many_arguments,
    reason = "worker startup binds GPU renderer, neural models, tile store, and corridor options"
)]
pub(super) fn spawn_corridor_workers(
    surface: &Arc<Surface>,
    store: &Arc<CorridorTileStore>,
    projector: HadamardProjector,
    xfeat: &LoadedXFeat,
    lighterglue: Option<&LighterGlue>,
    search_tier_matcher: Option<&Arc<SearchTierMatcher>>,
    cases: &[PhotoCase],
    opts: &CorridorTrackOptions,
) -> Result<CorridorRuntimeWorkers, PipelineError> {
    let first_eff = cases.first().map(|c| apply_case_overrides(c, opts));
    let init_rs = first_eff
        .as_ref()
        .map_or([800, 600], |c| tracking_render_size(c.size));
    let init_k = first_eff.as_ref().map_or_else(
        || nalgebra::Matrix3::new(600.0, 0.0, 399.5, 0.0, 600.0, 299.5, 0.0, 0.0, 1.0),
        |c| {
            let kf = estimate_intrinsics_with_estimator(c, &FocalLengthEstimator::default());
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
            let dem = surface
                .sample_dem(pos.latitude, pos.longitude)
                .unwrap_or(0.0);
            let alt = pos.altitude_m.unwrap_or(dem + 120.0).max(dem + 25.0);
            let hdg = p.direction_deg.filter(|d| d.is_finite()).unwrap_or(0.0);
            let pitch = if alt - dem <= 250.0 { 88.0 } else { 15.0 };
            Some((
                surface.local(pos.latitude, pos.longitude, alt),
                surface.camera_rotation(pos.latitude, pos.longitude, hdg, pitch, 0.0),
            ))
        });
    let t_warmup = Instant::now();
    let _ = xfeat.warmup(lighterglue)?;
    if let Some((eye, rot)) = init_prior_pose {
        store
            .project_virtual_kashin(&VirtualViewCamera {
                eye_enu: eye,
                world_to_cam: rot_to_row_major(&rot),
                intrinsics: [
                    init_k[(0, 0)],
                    init_k[(1, 1)],
                    init_k[(0, 2)],
                    init_k[(1, 2)],
                ],
                image_size: init_rs,
                max_landmarks: 256,
                min_cos_incidence: 0.45,
                min_cos_orientation: -1.0,
            })
            .ok();
    }
    let sync_gpu = (!opts.background_refresh && opts.polish_anchors)
        .then(|| {
            let mut g = SurfaceGpuRenderer::try_new_with_role(
                surface,
                &init_k,
                init_rs,
                GpuQueueRole::ForegroundTracking,
            )?;
            if let Some((eye, ref rot)) = init_prior_pose {
                g.render_view(surface, eye, rot, &init_k, init_rs).ok();
            }
            Some(g)
        })
        .flatten();
    let sync_gpu_renderer = Mutex::new(sync_gpu);
    let tile_params = TileSynthParams {
        tile_m: f64::from(store.tile_size_m.max(10.0)),
        agl_m: 480.0,
        max_per_tile: 4096,
        stride: 4,
        include_obliques: !cfg!(debug_assertions),
    };
    let need_live_tiles = cases.iter().any(|c| {
        estimate_lookahead_enu(surface, &TrackState::new(), &apply_case_overrides(c, opts))
            .is_some_and(|la| !store.has_tile(store.tile_key_at(la[0], la[1])))
    });
    let background_anchor = opts
        .background_refresh
        .then(|| {
            let init_gpu = opts
                .polish_anchors
                .then(|| {
                    let mut g = SurfaceGpuRenderer::try_new_with_role(
                        surface,
                        &init_k,
                        init_rs,
                        GpuQueueRole::BackgroundAnchor,
                    )?;
                    if let Some((eye, ref rot)) = init_prior_pose {
                        g.render_view(surface, eye, rot, &init_k, init_rs).ok();
                    }
                    Some(g)
                })
                .flatten();
            BackgroundAnchor::spawn(
                AnchorContext {
                    surface: Arc::clone(surface),
                    store: Arc::clone(store),
                    projector,
                    xfeat: xfeat.clone(),
                    lighterglue: lighterglue.cloned(),
                    search_tier_matcher: search_tier_matcher.cloned(),
                    opts: opts.clone(),
                },
                init_gpu,
            )
        })
        .transpose()?;
    let tile_builder = opts
        .background_refresh
        .then(|| {
            BackgroundTileBuilder::spawn(
                TileBuilderContext {
                    surface: Arc::clone(surface),
                    store: Arc::clone(store),
                    projector,
                    xfeat: xfeat.clone(),
                    params: tile_params,
                },
                need_live_tiles,
            )
        })
        .transpose()?;
    Ok(CorridorRuntimeWorkers {
        sync_gpu_renderer,
        tile_params,
        background_anchor,
        tile_builder,
        shader_warmup_ms: t_warmup.elapsed().as_secs_f64() * 1000.0,
    })
}

/// Initialize session camera calibration and map cell size from the first case.
pub(super) fn init_tracker_session(
    state: &mut TrackState,
    cases: &[PhotoCase],
    opts: &CorridorTrackOptions,
    focal_estimator: &FocalLengthEstimator,
    cell_m: f64,
) -> Result<(), PipelineError> {
    if let Some(first_raw) = cases.first() {
        let first_case = apply_case_overrides(first_raw, opts);
        let rs = tracking_render_size(first_case.size);
        let kf = estimate_intrinsics_with_estimator(&first_case, focal_estimator);
        let sx = f64::from(rs[0]) / f64::from(first_case.size[0].max(1));
        let sy = f64::from(rs[1]) / f64::from(first_case.size[1].max(1));
        let kr = scale_k(&kf, sx, sy);
        let cam_model = CameraModel {
            width: rs[0].max(32),
            height: rs[1].max(32),
            fx: kr[(0, 0)].max(10.0),
            fy: kr[(1, 1)].max(10.0),
            cx: kr[(0, 2)],
            cy: kr[(1, 2)],
        };
        state
            .configure_session(cam_model, cell_m)
            .map_err(PipelineError::Invalid)?;
    }
    Ok(())
}

/// Sleep until capture timestamp `time_s` when `--pace-realtime` is active.
pub(super) fn pace_if_realtime(
    pace_realtime: bool,
    raw_case: &PhotoCase,
    pace_origin: &mut Option<(f64, Instant)>,
) {
    if pace_realtime && let Some(t_s) = raw_case.time_s.filter(|t| t.is_finite()) {
        if let Some((t0_s, wall0)) = *pace_origin {
            let target_s = (t_s - t0_s).max(0.0);
            let elapsed_s = wall0.elapsed().as_secs_f64();
            if target_s > elapsed_s {
                std::thread::sleep(std::time::Duration::from_secs_f64(target_s - elapsed_s));
            }
        } else {
            *pace_origin = Some((t_s, Instant::now()));
        }
    }
}

/// Return true when an unpaced run should wait for an in-flight cold anchor before advancing.
pub(super) fn should_wait_cold_anchor(
    pace_realtime: bool,
    evaluated: usize,
    raw_case: &PhotoCase,
    state: &TrackState,
) -> bool {
    if pace_realtime || !state.active_tracks.is_empty() || state.prev_eye.is_some() {
        return false;
    }
    if evaluated == 2 {
        return true;
    }
    let continuous_video = raw_case
        .time_s
        .zip(state.prev_time_s)
        .is_some_and(|(cur, prev)| cur >= prev && (cur - prev) <= 1.0);
    continuous_video && (evaluated <= 16 || evaluated.is_multiple_of(15))
}

/// Poll or wait on `background_anchor` before frame feature extraction and lock focal length early.
pub(super) fn poll_anchor_before_frame(
    background_anchor: &mut Option<BackgroundAnchor>,
    focal_estimator: &mut FocalLengthEstimator,
    pace_realtime: bool,
    evaluated: usize,
    raw_case: &PhotoCase,
    state: &TrackState,
) -> Result<Option<AnchorOutcome>, PipelineError> {
    let outcome = if should_wait_cold_anchor(pace_realtime, evaluated, raw_case, state)
        && let Some(worker) = background_anchor.as_mut()
    {
        worker.wait_for_shutdown()?
    } else if let Some(worker) = background_anchor.as_mut() {
        worker.poll()?
    } else {
        None
    };
    if let Some(ref done) = outcome
        && let Some(locked) = done.locked_focal_px
    {
        focal_estimator.locked_focal_px = Some(locked);
    }
    Ok(outcome)
}

/// Resolve capture time in seconds from `time_s` or ISO-8601 `prior.capture_time`.
pub(super) fn effective_case_time_s(case: &PhotoCase) -> Option<f64> {
    case.time_s.filter(|t| t.is_finite()).or_else(|| {
        case.prior
            .as_ref()?
            .capture_time
            .as_deref()
            .and_then(parse_iso8601_ms)
            .map(|ms| (ms as f64) * 1e-3)
    })
}

/// Compute `dt_s` and reset cross-frame track state across `> 5 s` time gaps or large prior jumps.
pub(super) fn advance_case_timing_and_continuity(
    surface: &Surface,
    state: &mut TrackState,
    focal_estimator: &mut FocalLengthEstimator,
    case: &PhotoCase,
    profile: &VehicleDynamicsProfile,
) -> (Option<f64>, f64, bool) {
    let cur_time_s = effective_case_time_s(case);
    let mut discontinuous = false;
    let dt_s = match (cur_time_s, state.prev_time_s) {
        (Some(cur), Some(prev)) if cur < prev || (cur - prev) > 5.0 => {
            discontinuous = true;
            (cur - prev).abs().max(5.001)
        }
        _ => compute_dt_s(cur_time_s, state.prev_time_s, 30.0),
    };
    let cur_prior_enu = case
        .prior
        .as_ref()
        .and_then(|p| p.position.as_ref())
        .map(|pos| surface.local(pos.latitude, pos.longitude, pos.altitude_m.unwrap_or(0.0)));
    if let (Some(cp), Some(pp)) = (cur_prior_enu, state.prev_prior_enu) {
        let jump_m = (cp[0] - pp[0]).hypot(cp[1] - pp[1]);
        if jump_m > 3.0 * profile.max_displacement(dt_s) {
            discontinuous = true;
        }
    }
    state.prev_prior_enu = cur_prior_enu;
    state.prev_time_s = cur_time_s;
    if discontinuous {
        state.reset_discontinuous_track();
        if !has_calibrated_focal_prior(case) {
            focal_estimator.locked_focal_px = None;
        }
    } else {
        state.time_since_anchor_s += dt_s;
    }
    (cur_time_s, dt_s, discontinuous)
}

/// Ensure corridor tiles are indexed or synthesized for lookahead position.
///
/// # Errors
/// Returns [`PipelineError`] if tile synthesis fails.
#[expect(
    clippy::too_many_arguments,
    reason = "lookahead tile preloading requires terrain, tile store, builders, and inference state"
)]
pub(crate) fn ensure_lookahead_tiles(
    surface: &Surface,
    store: &CorridorTileStore,
    tile_builder: &mut Option<BackgroundTileBuilder>,
    sync_gpu_renderer: &Mutex<Option<SurfaceGpuRenderer>>,
    xfeat: &LoadedXFeat,
    projector: &HadamardProjector,
    tile_params: TileSynthParams,
    lookahead: [f64; 2],
    tile_build_ms: &mut Vec<f64>,
    sync_tiles_built: &mut usize,
) -> Result<(), PipelineError> {
    if let Some(tb) = tile_builder.as_mut() {
        tb.poll(tile_build_ms)?;
        tb.request_neighborhood(surface, store, lookahead)?;
    } else {
        let ck = store.tile_key_at(lookahead[0], lookahead[1]);
        if !store.has_tile(ck) {
            let t_tb = Instant::now();
            let entries = synthesize_corridor_tiles(
                surface,
                Some(sync_gpu_renderer),
                xfeat,
                projector,
                ck,
                tile_params,
            )?;
            if !entries.is_empty() {
                for entry in entries {
                    store.insert_tile(entry, true)?;
                    *sync_tiles_built += 1;
                }
                tile_build_ms.push(t_tb.elapsed().as_secs_f64() * 1000.0);
            }
        }
    }
    Ok(())
}
