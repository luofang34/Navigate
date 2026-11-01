//! Telemetry and summary formatting for corridor tracker.

use super::super::{
    admission::FusionAdmission,
    latency::{StageLatencies, percentile},
    masking::MaskFrameTelemetry,
    session::FilterIntegritySnapshot,
};
use serde_json::{Value, json};

/// Append telemetry JSON for one processed frame.
#[allow(clippy::too_many_arguments)]
pub(crate) fn record_frame_telemetry(
    telemetry: &mut Vec<Value>,
    case_id: &str,
    is_accepted: bool,
    runtime_name: &str,
    admission: &FusionAdmission,
    filter_snap: &FilterIntegritySnapshot,
    active_tracks: usize,
    cached_tiles: usize,
    resident_bytes: usize,
    query_keypoints: usize,
    lg_layers: usize,
    polish_rays: usize,
    mask_tel: &MaskFrameTelemetry,
    query_xfeat_ms: f64,
    solve_ms: f64,
    total_case_ms: f64,
) {
    telemetry.push(json!({
        "id": case_id,
        "accepted": is_accepted,
        "xfeat_runtime": runtime_name,
        "tracking_mode": admission.tracking_mode,
        "evidence_kind": admission.evidence_kind,
        "independent_ekf_fix": admission.independent_ekf_fix,
        "fusion_eligibility": admission.fusion_eligibility,
        "map_cell": admission.map_cell,
        "filter_integrity": {
            "quality": filter_snap.quality,
            "redundancy": filter_snap.redundancy,
            "fault_detection": filter_snap.fault_detection,
            "horizontal_1sigma_m": filter_snap.horizontal_1sigma_m,
            "vertical_1sigma_m": filter_snap.vertical_1sigma_m,
        },
        "session_revision": filter_snap.session_revision,
        "trajectory_revisions": filter_snap.trajectory_revisions,
        "active_tracks": active_tracks,
        "cached_tiles": cached_tiles,
        "resident_window_bytes": resident_bytes,
        "query_keypoints": query_keypoints,
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
}

/// Input parameters for tracker run summary serialization.
pub(crate) struct RunSummaryParams<'a> {
    pub(crate) evaluated: usize,
    pub(crate) accepted: usize,
    pub(crate) acquiring_frames: usize,
    pub(crate) warm_tracked_frames: usize,
    pub(crate) independent_fixes: usize,
    pub(crate) seg_refreshes: usize,
    pub(crate) runtime_name: &'a str,
    pub(crate) indexed_tile_count: usize,
    pub(crate) cached_tile_count: usize,
    pub(crate) resident_window_bytes: usize,
    pub(crate) tiles_built_live: usize,
    pub(crate) tile_build_ms: &'a [f64],
    pub(crate) shader_warmup_ms: f64,
    pub(crate) output_dir: &'a str,
    pub(crate) telemetry: Vec<Value>,
    pub(crate) background_refresh: bool,
    pub(crate) anchor_results: Vec<Value>,
    pub(crate) latencies: &'a StageLatencies,
    pub(crate) filter_snap: &'a FilterIntegritySnapshot,
    pub(crate) locked_focal_px: Option<f64>,
}

/// Build summary JSON for the complete tracker run.
#[must_use]
pub(crate) fn build_run_summary(p: RunSummaryParams<'_>) -> Value {
    let mut summary = json!({
        "evaluated": p.evaluated,
        "accepted": p.accepted,
        "acquiring_anchor_frames": p.acquiring_frames,
        "warm_tracked_frames": p.warm_tracked_frames,
        "independent_ekf_fixes": p.independent_fixes,
        "segformer_refreshes": p.seg_refreshes,
        "xfeat_runtime": p.runtime_name,
        "corridor_tiles": p.indexed_tile_count,
        "cached_tiles": p.cached_tile_count,
        "resident_window_bytes": p.resident_window_bytes,
        "tiles_built_live": p.tiles_built_live,
        "tile_build_p50_ms": percentile(p.tile_build_ms, 50.0),
        "shader_warmup_ms": p.shader_warmup_ms,
        "output": p.output_dir,
        "telemetry": p.telemetry,
        "background_refresh": p.background_refresh,
        "anchor_results": p.anchor_results,
        "locked_focal_px": p.locked_focal_px,
        "session_revision": p.filter_snap.session_revision,
        "trajectory_revisions": p.filter_snap.trajectory_revisions,
        "filter_integrity": {
            "quality": p.filter_snap.quality,
            "redundancy": p.filter_snap.redundancy,
            "fault_detection": p.filter_snap.fault_detection,
            "horizontal_1sigma_m": p.filter_snap.horizontal_1sigma_m,
            "vertical_1sigma_m": p.filter_snap.vertical_1sigma_m,
        },
    });
    if let Some(map) = summary.as_object_mut() {
        p.latencies.write_summary(map);
    }
    summary
}
