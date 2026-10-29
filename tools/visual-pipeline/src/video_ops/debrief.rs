//! Direct in-process Navigate session debrief builder (`prepare-navigate-debrief`).

use crate::{
    PipelineError,
    util::{read_json, write_json},
};
use navigate_recording::{Configuration, Observation, Processor, ResultFrame};
use serde_json::{Value, json};
use std::path::Path;

/// Drive `navigate_recording::Processor` over an evidence directory and write `debrief.json`.
///
/// Reads `configuration.json` and `observations.json` (or JSONL `observations.jsonl`) inside
/// `evidence_dir`, feeds each [`Observation`] into [`Processor::process_blocking`], and writes
/// `results.json` plus a `sokoly-debrief` v1 `debrief.json` to `output_dir`.
///
/// # Errors
///
/// Returns [`PipelineError`] when configuration validation, frame processing, or writing fails.
pub fn prepare_navigate_debrief(
    evidence_dir: &Path,
    output_dir: &Path,
) -> Result<Value, PipelineError> {
    std::fs::create_dir_all(output_dir).map_err(|source| PipelineError::Io {
        path: output_dir.to_path_buf(),
        source,
    })?;
    let cfg_path = evidence_dir.join("configuration.json");
    let config: Configuration = read_json(&cfg_path)?;
    let observations = load_observations(evidence_dir)?;

    let mut processor =
        Processor::new(config.clone()).map_err(|e| PipelineError::Invalid(e.to_string()))?;

    let mut results: Vec<ResultFrame> = Vec::with_capacity(observations.len());
    for obs in observations {
        let frame_res = processor
            .process_blocking(evidence_dir, obs)
            .map_err(|e| PipelineError::Invalid(e.to_string()))?;
        results.push(frame_res);
    }

    write_json(&output_dir.join("results.json"), &results)?;

    let located = results.iter().filter(|r| r.pose.is_some()).count();
    let confirmed = results.iter().filter(|r| r.confirmed).count();
    let nav_ready = results
        .iter()
        .filter(|r| r.support.as_ref().is_some_and(|s| s.navigation))
        .count();

    let items: Vec<Value> = results
        .iter()
        .map(|r| {
            json!({
                "sequence": r.sequence,
                "media_ns": r.media_ns,
                "observation_sha256": r.observation_sha256,
                "pose": r.pose,
                "geodetic": r.geodetic,
                "camera_agl": r.camera_agl,
                "position_bound_m": r.position_bound_m,
                "confirmed": r.confirmed,
                "map_observation_accepted": r.map_observation_accepted,
                "support": r.support,
                "revision": r.revision,
                "diagnostics": r.diagnostics,
            })
        })
        .collect();

    let debrief = json!({
        "format": "sokoly-debrief",
        "version": 1,
        "kind": "navigate-session",
        "anchor_lat_lon": config.anchor,
        "map_release": config.map_release,
        "map_sha256": config.map_sha256,
        "matcher": config.matcher,
        "summary": {
            "frames": results.len(),
            "located": located,
            "confirmed": confirmed,
            "navigation_permitted": nav_ready,
        },
        "items": items,
    });
    write_json(&output_dir.join("debrief.json"), &debrief)?;
    Ok(debrief["summary"].clone())
}

fn load_observations(evidence_dir: &Path) -> Result<Vec<Observation>, PipelineError> {
    let json_path = evidence_dir.join("observations.json");
    if json_path.is_file() {
        return read_json(&json_path);
    }
    let jsonl_path = evidence_dir.join("observations.jsonl");
    let text = std::fs::read_to_string(&jsonl_path).map_err(|source| PipelineError::Io {
        path: jsonl_path.clone(),
        source,
    })?;
    let mut out = Vec::new();
    for (idx, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let obs: Observation =
            serde_json::from_str(trimmed).map_err(|source| PipelineError::Json {
                context: format!("{}:{}", jsonl_path.display(), idx + 1),
                source,
            })?;
        out.push(obs);
    }
    Ok(out)
}
