//! Initialization from traced scene estimates, separate from geometric checks.
use super::{Scene, graph};
use crate::error::PreviewError;
use navigate_visual::reconstruction::{ScenePointSeed, initialize_scene_tracks};
use serde::Deserialize;
use serde_json::json;
use wasm_bindgen::prelude::*;

#[derive(Deserialize)]
struct Seed {
    feature_id: String,
    position_scene_units: [f64; 3],
}

/// Initialize conditional points from source tracks and prior scene estimates.
/// Run in a worker. The host retains source scene and association identities.
#[wasm_bindgen]
pub fn initialize_scene_points(
    camera_json: &str,
    graph_json: &str,
    scene_json: &str,
    seeds_json: &str,
) -> Result<String, JsValue> {
    run(camera_json, graph_json, scene_json, seeds_json).map_err(JsValue::from)
}
fn run(
    camera_json: &str,
    graph_json: &str,
    scene_json: &str,
    seeds_json: &str,
) -> Result<String, PreviewError> {
    let camera = graph::camera(camera_json)?;
    let tracks = graph::decode(graph_json)?;
    let source: Scene = serde_json::from_str(scene_json)?;
    let scene = source.cameras_only().model()?;
    let seeds: Vec<Seed> = serde_json::from_str(seeds_json)?;
    let seeds = seeds
        .into_iter()
        .map(|s| {
            let feature_id = s
                .feature_id
                .parse()
                .map_err(|source| PreviewError::ScenePointId {
                    id: s.feature_id,
                    source,
                })?;
            Ok(ScenePointSeed {
                feature_id,
                position: s.position_scene_units.into(),
            })
        })
        .collect::<Result<Vec<_>, PreviewError>>()?;
    let result = initialize_scene_tracks(&camera, &tracks, &scene.cameras, &seeds)?;
    Ok(json!({"stage":"conditional_scene_initialization","geographic_acceptance":false,
        "uncertainty":"unknown; estimated cameras, calibration, and point seeds",
        "evidence_correlation":"unknown; retained point estimates and tracks reuse source observations",
        "unresolved_feature_ids":result.unresolved_feature_ids.iter().map(u64::to_string).collect::<Vec<_>>(),
        "scene":source.with_points_from(&result.scene)}).to_string())
}

#[cfg(test)]
mod tests;
