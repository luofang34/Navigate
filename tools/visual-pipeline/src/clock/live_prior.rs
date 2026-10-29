//! Sokoly live survey seed builder (`live_prior`).

use super::{
    PriorTrackEntry, capture_anchor, mercator_local, movie_duration_s, rotation_from_heading,
};
use crate::{
    PipelineError,
    util::{read_json, unix_ms_to_iso8601},
};
use serde_json::{Value, json};
use std::path::Path;

fn heading_from_corners(anchor: [f64; 2], corners: &[[f64; 2]; 4]) -> Option<f64> {
    let tl = mercator_local(anchor, corners[0][1], corners[0][0], 0.0);
    let bl = mercator_local(anchor, corners[3][1], corners[3][0], 0.0);
    let east = tl[0] - bl[0];
    let north = tl[1] - bl[1];
    (east.hypot(north) >= 1e-6).then(|| east.atan2(north).to_degrees().rem_euclid(360.0))
}

/// Seed `acquisition.json` and `navigate-prior-track` v1 from a Sokoly live survey.
///
/// # Errors
///
/// Returns [`PipelineError`] when replay checks, clock alignment, or manifest reads fail.
#[allow(clippy::too_many_arguments)]
pub fn seed_live_prior(
    replay_dir: &Path,
    video_path: &Path,
    map_manifest_path: &Path,
    offset_s: f64,
    size: [u32; 2],
    sensor_fov_deg: f64,
    terrain_m: f64,
    max_gap_s: f64,
) -> Result<(Value, Value), PipelineError> {
    let doc: Value = read_json(&replay_dir.join("replay.json"))?;
    if doc.get("version").and_then(Value::as_i64) != Some(1) {
        return Err(PipelineError::Invalid(
            "live replay must be version 1".into(),
        ));
    }
    let anchor_obj = doc.get("time_anchor").ok_or_else(|| {
        PipelineError::Invalid("live replay needs a utc or unix time anchor".into())
    })?;
    let scale_str = anchor_obj.get("source_scale").and_then(Value::as_str);
    if !matches!(scale_str, Some("utc" | "unix")) {
        return Err(PipelineError::Invalid(
            "live replay needs a utc or unix time anchor".into(),
        ));
    }
    let replay_unix_ms = anchor_obj
        .get("unix_ms")
        .and_then(Value::as_f64)
        .ok_or_else(|| PipelineError::Invalid("missing time_anchor.unix_ms".into()))?;
    let replay_at_ms = anchor_obj.get("at").and_then(Value::as_f64).unwrap_or(0.0);

    let manifest: Value = read_json(map_manifest_path)?;
    let anchor_arr = manifest
        .get("anchor_lat_lon")
        .and_then(Value::as_array)
        .ok_or_else(|| PipelineError::Invalid("missing anchor_lat_lon in map manifest".into()))?;
    let anchor = [
        anchor_arr.first().and_then(Value::as_f64).unwrap_or(0.0),
        anchor_arr.get(1).and_then(Value::as_f64).unwrap_or(0.0),
    ];

    let video_anchor = capture_anchor(video_path)?
        .ok_or_else(|| PipelineError::Invalid("the video has no nonzero movie clock".into()))?;
    let video_start_ms = (video_anchor.unix_ms as f64) + offset_s * 1000.0;
    let duration_s = movie_duration_s(video_path)?;

    let raw_frames = doc
        .get("frames")
        .and_then(Value::as_array)
        .ok_or_else(|| PipelineError::Invalid("missing frames in live replay".into()))?;
    let mut reg = Vec::new();
    for f in raw_frames {
        if f.get("corners").is_none_or(Value::is_null)
            || f.get("position").is_none_or(Value::is_null)
        {
            continue;
        }
        let img_rel = f
            .get("image")
            .and_then(Value::as_str)
            .ok_or_else(|| PipelineError::Invalid("missing frame.image".into()))?;
        let rel_path = Path::new(img_rel);
        if rel_path.is_absolute()
            || rel_path
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(PipelineError::Invalid(format!(
                "frame image escapes the live replay: {img_rel}"
            )));
        }
        if !replay_dir.join(rel_path).is_file() {
            return Err(PipelineError::Invalid(format!(
                "live frame is missing: {img_rel}"
            )));
        }
        reg.push(f);
    }
    if reg.is_empty() {
        return Err(PipelineError::Invalid(
            "the live replay has no registered frame".into(),
        ));
    }

    let frame_unix = |f: &Value| -> f64 {
        replay_unix_ms + f.get("time_s").and_then(Value::as_f64).unwrap_or(0.0) * 1000.0
            - replay_at_ms
    };
    let best = reg
        .iter()
        .copied()
        .min_by(|a, b| {
            (frame_unix(a) - video_start_ms)
                .abs()
                .total_cmp(&(frame_unix(b) - video_start_ms).abs())
        })
        .ok_or_else(|| PipelineError::Invalid("no registered frames".into()))?;
    let gap_ms = (frame_unix(best) - video_start_ms).abs();
    if gap_ms > max_gap_s * 1000.0 {
        return Err(PipelineError::Invalid(format!(
            "nearest registered live frame is {:.1} s from the video start",
            gap_ms / 1000.0
        )));
    }

    let agl_m = doc.get("agl_m").and_then(Value::as_f64).unwrap_or(100.0);
    let default_heading = doc
        .get("heading_deg")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let corners: [[f64; 2]; 4] =
        serde_json::from_value(best["corners"].clone()).map_err(|source| PipelineError::Json {
            context: "frame.corners".into(),
            source,
        })?;
    let pos_arr: [f64; 2] =
        serde_json::from_value(best["position"].clone()).map_err(|source| PipelineError::Json {
            context: "frame.position".into(),
            source,
        })?;
    let (lon, lat) = (pos_arr[0], pos_arr[1]);
    let heading = heading_from_corners(anchor, &corners).unwrap_or(default_heading);
    let local_pos = mercator_local(anchor, lat, lon, terrain_m + agl_m);

    let mut entries = Vec::new();
    for f in &reg {
        let vt = (frame_unix(f) - video_start_ms) / 1000.0;
        let Some(dur) = duration_s else { continue };
        if !(0.0..=dur).contains(&vt) {
            continue;
        }
        let fc: [[f64; 2]; 4] =
            serde_json::from_value(f["corners"].clone()).map_err(|source| PipelineError::Json {
                context: "track.corners".into(),
                source,
            })?;
        let fp: [f64; 2] = serde_json::from_value(f["position"].clone()).map_err(|source| {
            PipelineError::Json {
                context: "track.position".into(),
                source,
            }
        })?;
        let fh = heading_from_corners(anchor, &fc).unwrap_or(default_heading);
        entries.push(PriorTrackEntry {
            video_time_s: vt,
            replay_time_s: f.get("time_s").and_then(Value::as_f64).unwrap_or(0.0),
            longitude: fp[0],
            latitude: fp[1],
            agl_m,
            heading_deg: fh,
            corners: fc,
        });
    }
    entries.sort_by(|a, b| a.video_time_s.total_cmp(&b.video_time_s));

    let focal = f64::from(size[0]) * 1.25 / (2.0 * (sensor_fov_deg.to_radians() * 0.5).tan());
    let track_json = json!({
        "format": "navigate-prior-track",
        "version": 1,
        "video": {
            "start_unix_ms": video_start_ms,
            "duration_s": duration_s,
            "offset_s": offset_s,
            "replay_anchor_unix_ms": replay_unix_ms as i64,
        },
        "source": "Sokoly live survey: flight-controller pose per registered frame; nadir assumption, no visual registration; camera clock offset confirmed by the operator",
        "frames": entries,
    });

    let replay_abs = replay_dir
        .canonicalize()
        .unwrap_or_else(|_| replay_dir.to_path_buf());
    let map_abs = map_manifest_path
        .canonicalize()
        .unwrap_or_else(|_| map_manifest_path.to_path_buf());
    let frame_img = best["image"].as_str().unwrap_or("");
    let frame_t = best["time_s"].as_f64().unwrap_or(0.0);

    let acquisition = json!({
        "initial_user_prior": {
            "longitude": lon,
            "latitude": lat,
            "source": format!(
                "Sokoly live survey: flight-controller position of the registered frame nearest the video start; frame {frame_img} at {frame_t} s; camera clock offset {offset_s} s confirmed by the operator"
            ),
        },
        "camera": {
            "size": size,
            "focal": [focal, focal],
            "principal": [(f64::from(size[0]) - 1.0) * 0.5, (f64::from(size[1]) - 1.0) * 0.5],
        },
        "pose": {
            "position": local_pos,
            "rotation": rotation_from_heading(heading),
        },
        "map_manifest": map_abs.display().to_string(),
        "live_replay": {
            "directory": replay_abs.display().to_string(),
            "frame": frame_img,
            "frame_time_s": frame_t,
            "replay_anchor_unix_ms": replay_unix_ms as i64,
            "video_start_unix_ms": video_start_ms,
            "offset_s": offset_s,
            "terrain_m": terrain_m,
            "height_above_launch_m": agl_m,
            "heading_deg": heading,
        },
        "prior_track": track_json,
    });

    let summary = json!({
        "video_start_utc": unix_ms_to_iso8601(video_anchor.unix_ms),
        "replay_anchor_utc": unix_ms_to_iso8601(replay_unix_ms as i64),
        "offset_s": offset_s,
        "frame_time_s": frame_t,
        "frame_position": [lon, lat],
        "gap_s": gap_ms / 1000.0,
        "video_duration_s": duration_s,
        "frames_inside_video": entries.len(),
        "registered_frames": reg.len(),
        "track_frames": entries.len(),
    });
    Ok((acquisition, summary))
}
