//! Lookahead tile synthesis, frame timestamping, and discontinuity resets for corridor tracker.

use super::super::{
    CorridorTrackOptions,
    admission::apply_case_overrides,
    kinematics::{VehicleDynamicsProfile, compute_dt_s},
    motion::TrackState,
    tile_builder::{BackgroundTileBuilder, TileSynthParams, synthesize_corridor_tiles},
};
use super::solve::tracking_render_size;
use crate::{
    PipelineError,
    stages::{
        estimate_intrinsics_with_estimator, focal::FocalLengthEstimator,
        has_calibrated_focal_prior, scale_k,
    },
    surface::{Surface, SurfaceGpuRenderer},
    testset::PhotoCase,
    util::parse_iso8601_ms,
    xfeat_runtime::LoadedXFeat,
};
use navigate_match::{CorridorTileStore, HadamardProjector};
use navigate_visual::CameraModel;
use std::{sync::Mutex, time::Instant};

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
