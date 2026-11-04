//! JSON telemetry, attempt serialization, and debrief report emission for `corridor-track`.

use super::super::{
    latency::{StageLatencies, percentile},
    session::FilterIntegritySnapshot,
};
use crate::{PipelineError, json_file::write_json, pose::save_candidate_npz, surface::Surface};
use navigate_visual_flight::{
    AnchorArrivalRecord, AttemptRecord, FrameDiagnostics, FrameTelemetry, TrackingMode,
};
use serde_json::{Value, json};
use std::{fs, path::Path};

fn evidence_kind_for_mode(mode: TrackingMode) -> &'static str {
    match mode {
        TrackingMode::WarmTrack => "surface_track_odometry",
        TrackingMode::AcquiringAnchor => "visual_odometry",
        TrackingMode::CorridorAnchor | TrackingMode::Unresolved => "corridor_map_anchor",
    }
}

/// Convert a typed [`AttemptRecord`] into the debrief JSON attempt format.
#[must_use]
pub(crate) fn format_attempt_json(rec: &AttemptRecord) -> Value {
    let mut obj = json!({
        "backend": rec.backend,
        "seed": rec.seed,
        "step": rec.step,
        "sketch_score": rec.sketch_score,
        "pairs": rec.pairs,
        "score": rec.score,
    });
    if let Some(map) = obj.as_object_mut() {
        if let Some(vlm) = rec.virtual_landmarks {
            map.insert("virtual_landmarks".into(), vlm.into());
        }
        if let Some(vt) = rec.visible_tiles {
            map.insert("visible_tiles".into(), vt.into());
        }
        if let Some(fm) = rec.focal_multiplier {
            map.insert("focal_multiplier".into(), fm.into());
        }
    }
    obj
}

/// Convert a typed [`AnchorArrivalRecord`] into the `anchor_results` summary JSON entry.
#[must_use]
pub(crate) fn format_anchor_arrival_json(arr: &AnchorArrivalRecord) -> Value {
    let attempts: Vec<Value> = arr.attempts.iter().map(format_attempt_json).collect();
    let mut obj = json!({
        "anchor_result_for": arr.sent_frame_label,
        "arrived_at": arr.arrived_frame_label,
        "propagated_frames": arr.propagated_frames,
        "capture_lag_ns": arr.capture_lag_ns,
        "accepted": arr.accepted,
        "confidence": arr.confidence.map(|c| c.as_str()),
        "attempt_count": attempts.len(),
        "attempts": attempts,
        "anchor_ms": arr.anchor_ms,
    });
    if arr.accepted
        && let Some(map) = obj.as_object_mut()
    {
        map.insert("independent_ekf_fix".into(), arr.independent_ekf_fix.into());
        map.insert(
            "fusion_eligibility".into(),
            arr.fusion_eligibility.as_str().into(),
        );
        map.insert("map_cell".into(), json!(arr.map_cell));
        map.insert("active_tracks".into(), arr.active_tracks.into());
        map.insert(
            "merged_into_healthy_foreground".into(),
            arr.merged_into_healthy_foreground.into(),
        );
    }
    obj
}

/// Append one frame's JSON telemetry object to `telemetry`.
pub(crate) fn record_frame_telemetry(
    telemetry: &mut Vec<Value>,
    case_id: &str,
    runtime_name: &str,
    tel: &FrameTelemetry,
    filter_snap: &FilterIntegritySnapshot,
) {
    telemetry.push(json!({
        "id": case_id,
        "accepted": tel.accepted,
        "xfeat_runtime": runtime_name,
        "tracking_mode": tel.tracking_mode.as_str(),
        "evidence_kind": evidence_kind_for_mode(tel.tracking_mode),
        "independent_ekf_fix": tel.independent_ekf_fix,
        "fusion_eligibility": tel.fusion_eligibility.as_str(),
        "map_cell": tel.map_cell,
        "capture_lag_ns": tel.capture_lag_ns,
        "filter_integrity": {
            "quality": filter_snap.quality,
            "redundancy": filter_snap.redundancy,
            "fault_detection": filter_snap.fault_detection,
            "horizontal_1sigma_m": filter_snap.horizontal_1sigma_m,
            "vertical_1sigma_m": filter_snap.vertical_1sigma_m,
        },
        "session_revision": filter_snap.session_revision,
        "trajectory_revisions": filter_snap.trajectory_revisions,
        "active_tracks": tel.active_tracks,
        "cached_tiles": tel.cached_tiles,
        "resident_window_bytes": tel.resident_window_bytes,
        "query_keypoints": tel.query_keypoints,
        "lighterglue_layers": tel.lighterglue_layers,
        "polish_rays_cast": tel.polish_rays_cast,
        "segformer_refreshed": tel.segformer_refreshed,
        "segformer_submitted": tel.segformer_submitted,
        "segformer_ms": tel.segformer_ms,
        "valid_mask_fraction": f64::from(tel.valid_mask_fraction),
        "luma_p50": tel.luma_p50,
        "propeller_pixels": tel.propeller_pixels,
        "mask_ms": tel.mask_ms,
        "query_xfeat_ms": tel.query_xfeat_ms,
        "solve_ms": tel.solve_ms,
        "total_case_ms": tel.total_frame_ms,
    }));
}

/// Build the per-case JSON report and optionally persist `{case_id}.json` + `candidate.npz`.
pub(crate) fn emit_corridor_case_report(
    output_dir: Option<&Path>,
    surface: &Surface,
    case_id: &str,
    tel: &FrameTelemetry,
    anchor_confidence: Option<navigate_visual_flight::AnchorConfidence>,
    diag: &FrameDiagnostics,
    accepted_count: &mut usize,
) -> Result<Value, PipelineError> {
    let filter_snap = &diag.filter_snapshot;
    let filter_integrity = json!({
        "quality": filter_snap.quality,
        "redundancy": filter_snap.redundancy,
        "fault_detection": filter_snap.fault_detection,
        "horizontal_1sigma_m": filter_snap.horizontal_1sigma_m,
        "vertical_1sigma_m": filter_snap.vertical_1sigma_m,
    });
    let attempts_json: Vec<Value> = diag.attempts.iter().map(format_attempt_json).collect();
    let evidence_kind = evidence_kind_for_mode(tel.tracking_mode);

    let report = if let Some((seed, step, ref cand)) = diag.candidate
        && cand.score.geometry_accepted
    {
        *accepted_count += 1;
        if let Some(dir) = output_dir {
            let case_dir = dir.join(case_id);
            fs::create_dir_all(&case_dir).map_err(|source| PipelineError::Io {
                path: case_dir.clone(),
                source,
            })?;
            save_candidate_npz(&case_dir.join("candidate.npz"), cand)?;
        }
        let [lat, lon, alt] = surface.geodetic(cand.eye);
        json!({
            "id": case_id,
            "status": "candidate_requires_review",
            "attempts": attempts_json,
            "filter_integrity": filter_integrity,
            "session_revision": filter_snap.session_revision,
            "trajectory_revisions": filter_snap.trajectory_revisions,
            "candidate": {
                "latitude": lat,
                "longitude": lon,
                "altitude_m": alt,
                "evidence_kind": evidence_kind,
                "tracking_mode": tel.tracking_mode.as_str(),
                "anchor_confidence": anchor_confidence.map(|c| c.as_str()),
                "independent_ekf_fix": tel.independent_ekf_fix,
                "fusion_eligibility": tel.fusion_eligibility.as_str(),
                "map_cell": tel.map_cell,
                "earlier_anchor_id": diag.earlier_anchor_id,
                "capture_lag_ns": tel.capture_lag_ns,
                "seed": seed,
                "step": step,
                "score": cand.score,
            },
        })
    } else {
        let status = if tel.tracking_mode == TrackingMode::AcquiringAnchor {
            "acquiring_anchor"
        } else {
            "unresolved"
        };
        json!({
            "id": case_id,
            "status": status,
            "tracking_mode": tel.tracking_mode.as_str(),
            "evidence_kind": evidence_kind,
            "fusion_eligibility": tel.fusion_eligibility.as_str(),
            "filter_integrity": filter_integrity,
            "session_revision": filter_snap.session_revision,
            "trajectory_revisions": filter_snap.trajectory_revisions,
            "attempts": attempts_json,
            "candidate": null,
        })
    };

    if let Some(dir) = output_dir {
        write_json(&dir.join(format!("{case_id}.json")), &report)?;
    }
    Ok(report)
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
    pub(crate) shared_gpu_texture_bytes: usize,
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

/// Build summary JSON for a `corridor-track` or `corridor-serve` run.
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
        "shared_gpu_texture_bytes": p.shared_gpu_texture_bytes,
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
