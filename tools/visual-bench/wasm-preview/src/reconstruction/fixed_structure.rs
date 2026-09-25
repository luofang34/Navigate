//! Conditional point triangulation from verified source links and estimated poses.
use super::{Scene, graph};
use crate::error::PreviewError;
use navigate_visual::reconstruction::triangulate_scene_tracks;
use serde_json::json;
use wasm_bindgen::prelude::*;

/// Triangulate source image tracks with unchanged estimated camera poses.
/// Run this synchronous operation in a worker. No geographic pose is accepted.
#[wasm_bindgen]
pub fn triangulate_scene_points(
    camera_json: &str,
    graph_json: &str,
    scene_json: &str,
) -> Result<String, JsValue> {
    run(camera_json, graph_json, scene_json).map_err(JsValue::from)
}

fn run(camera_json: &str, graph_json: &str, scene_json: &str) -> Result<String, PreviewError> {
    let camera = graph::camera(camera_json)?;
    let tracks = graph::decode(graph_json)?;
    let source: Scene = serde_json::from_str(scene_json)?;
    let scene = source.cameras_only().model()?;
    let result = triangulate_scene_tracks(&camera, &tracks, &scene.cameras)?;
    Ok(json!({"stage":"conditional_scene_triangulation","geographic_acceptance":false,
        "uncertainty":"unknown; conditional on estimated camera poses and calibration",
        "evidence_correlation":"unknown; image tracks and camera estimates share observations",
        "unresolved_feature_ids":result.unresolved_feature_ids.iter().map(u64::to_string).collect::<Vec<_>>(),
        "scene":source.with_points_from(&result.scene)}).to_string())
}

#[cfg(test)]
mod tests;
