//! Run geometry from verified saved groups without repeating image inference.
use super::*;
use navigate_visual::reconstruction::{AlignmentConfig, align_scenes};
/// Retrieve initialization pairs. These are scheduling proposals, not poses.
#[wasm_bindgen]
pub fn scene_seed_pairs(camera_json: &str, graph_json: &str) -> Result<String, JsValue> {
    let run = || -> Result<String, PreviewError> {
        let pairs = navigate_visual::reconstruction::candidate_seed_pairs(
            &graph::camera(camera_json)?,
            &graph::decode(graph_json)?,
        )?;
        Ok(serde_json::to_string(&pairs)?)
    };
    run().map_err(JsValue::from)
}
/// Propose initial poses from a saved image graph. Call in a worker.
#[wasm_bindgen]
pub fn scene_proposals(
    camera_json: &str,
    graph_json: &str,
    first: u32,
    second: u32,
) -> Result<String, JsValue> {
    let run = || -> Result<String, PreviewError> {
        proposals(
            &graph::camera(camera_json)?,
            &graph::decode(graph_json)?,
            [first as usize, second as usize],
        )
    };
    run().map_err(JsValue::from)
}
/// Reconstruct a saved image graph with one supplied conditional seed.
#[wasm_bindgen]
pub fn reconstruct_scene(
    camera_json: &str,
    graph_json: &str,
    seed_json: &str,
) -> Result<String, JsValue> {
    let run = || -> Result<String, PreviewError> {
        solve(
            &graph::camera(camera_json)?,
            &graph::decode(graph_json)?,
            seed_json,
        )
    };
    run().map_err(JsValue::from)
}
/// Initialize a group from two shared estimated cameras with the largest baseline.
#[wasm_bindgen]
pub fn seed_from_scene(graph_json: &str, previous_json: &str) -> Result<String, JsValue> {
    continuation(graph_json, previous_json).map_err(JsValue::from)
}
fn continuation(graph_json: &str, previous_json: &str) -> Result<String, PreviewError> {
    let graph = graph::decode(graph_json)?;
    let previous: Scene = serde_json::from_str(previous_json)?;
    let previous = previous.model()?;
    let mut poses = std::collections::BTreeMap::new();
    for camera in previous.cameras {
        if poses
            .insert(camera.observation_sha256.to_ascii_lowercase(), camera.pose)
            .is_some()
        {
            return Err(PreviewError::Input {
                reason: "repeated source camera in reconstruction continuation".into(),
            });
        }
    }
    let shared: Vec<_> = graph
        .observation_sha256
        .iter()
        .enumerate()
        .filter_map(|(i, id)| poses.get(&id.to_ascii_lowercase()).map(|p| (i, *p)))
        .collect();
    let mut best = None;
    let mut baseline = 1e-6;
    for (i, &a) in shared.iter().enumerate() {
        for &b in &shared[i + 1..] {
            let distance = (a.1.position - b.1.position).norm();
            if distance > baseline {
                baseline = distance;
                best = Some((a, b));
            }
        }
    }
    let (a, b) = best.ok_or_else(|| PreviewError::Input {
        reason: "shared camera poses do not supply a nonzero baseline".into(),
    })?;
    let seed = ReconstructionSeed::between([a.0, b.0], [a.1, b.1])?;
    Ok(serde_json::to_string(&Seed {
        camera_indices: seed.camera_indices,
        position_scene_units: seed.second_pose.position.into(),
        eye_to_scene_xyzw: seed.second_pose.orientation.coords.into(),
    })?)
}
/// Align two estimated scene groups by shared observation identity.
/// The result does not supply a navigation fix or independent confidence.
#[wasm_bindgen]
pub fn align_scene_groups(source_json: &str, target_json: &str) -> Result<String, JsValue> {
    alignment(source_json, target_json).map_err(JsValue::from)
}
fn alignment(source_json: &str, target_json: &str) -> Result<String, PreviewError> {
    let mut source: Scene = serde_json::from_str(source_json)?;
    let target: Scene = serde_json::from_str(target_json)?;
    let result = align_scenes(
        &source.model()?,
        &target.model()?,
        AlignmentConfig::default(),
    )?;
    source.transform(result.transform)?;
    Ok(json!({"scene":source,"stage":"conditional_scene_alignment","geographic_acceptance":false,
        "scale":result.transform.scale,"source_to_target_xyzw":result.transform.rotation.coords.as_slice(),"translation_target_scene_units":result.transform.translation.as_slice(),
        "observation_sha256":result.observation_sha256,"excluded_observation_sha256":result.excluded_observation_sha256,
        "position_rms_scene_units":result.position_rms_scene_units,"rotation_rms_rad":result.rotation_rms_rad,"target_baseline_scene_units":result.target_baseline_scene_units,
        "uncertainty":"unknown; correlated estimated camera poses and arbitrary scene scale",
        "evidence_correlation":"shared image identities; this is a coordinate fit, not independent evidence"}).to_string())
}
#[cfg(test)]
mod tests;
