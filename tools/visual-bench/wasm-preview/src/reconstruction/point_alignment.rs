//! Point-supported alignment for verified conditional scene records.
use super::*;
use navigate_visual::reconstruction::{
    ScenePointAlignment, align_scenes_with_points, associate_scene_points,
};
/// Propose alignments from shared image pixels and estimated scene points.
/// Call in a worker. Different scale proposals remain separate; no geographic
/// acceptance or independent confidence is supplied.
#[wasm_bindgen]
pub fn align_scene_groups_with_points(
    source_json: &str,
    target_json: &str,
) -> Result<String, JsValue> {
    align(source_json, target_json).map_err(JsValue::from)
}
fn align(source_json: &str, target_json: &str) -> Result<String, PreviewError> {
    let source: Scene = serde_json::from_str(source_json)?;
    let target: Scene = serde_json::from_str(target_json)?;
    let a = source.model()?;
    let b = target.model()?;
    let links = associate_scene_points(&a, &b, Default::default())?;
    let proposals = align_scenes_with_points(&a, &b, &links, Default::default())?;
    let candidates = proposals
        .candidates
        .iter()
        .map(|c| candidate(&source, c))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({"candidates":candidates,"point_proposals":links.len(),"candidate_budget_exhausted":proposals.candidate_budget_exhausted,
        "stage":"conditional_scene_alignment","geographic_acceptance":false}).to_string())
}
fn candidate(
    source: &Scene,
    result: &ScenePointAlignment,
) -> Result<serde_json::Value, PreviewError> {
    let mut scene = source.cameras_only();
    let alignment = &result.alignment;
    scene.transform(alignment.transform)?;
    let pairs = |items: &[navigate_visual::reconstruction::ScenePointAssociation]| {
        items.iter().map(|p|json!({"source_feature_id":p.source_feature_id.to_string(),"target_feature_id":p.target_feature_id.to_string()})).collect::<Vec<_>>()
    };
    Ok(
        json!({"scene":scene,"stage":"conditional_scene_alignment","geographic_acceptance":false,
        "alignment_support":"shared_cameras_and_scene_points",
        "scale":alignment.transform.scale,"source_to_target_xyzw":alignment.transform.rotation.coords.as_slice(),
        "translation_target_scene_units":alignment.transform.translation.as_slice(),
        "observation_sha256":alignment.observation_sha256,"excluded_observation_sha256":alignment.excluded_observation_sha256,
        "position_rms_scene_units":alignment.position_rms_scene_units,"rotation_rms_rad":alignment.rotation_rms_rad,
        "target_baseline_scene_units":alignment.target_baseline_scene_units,
        "point_associations":pairs(&result.associations),"excluded_point_associations":pairs(&result.excluded_associations),
        "point_rms_scene_units":result.point_rms_scene_units,"target_point_distance_scene_units":result.target_point_distance_scene_units,
        "uncertainty":"unknown; correlated estimated cameras, calibration and scene points; arbitrary scene scale",
        "evidence_correlation":"shared image identities; point and camera fits reuse image evidence"}),
    )
}
#[cfg(test)]
mod tests;
