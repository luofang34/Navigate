//! Apply a stored conditional alignment without fitting the shared poses again.
use super::*;
use navigate_visual::reconstruction::SceneTransform;

#[derive(Deserialize)]
struct Alignment {
    scale: f64,
    source_to_target_xyzw: [f64; 4],
    translation_target_scene_units: [f64; 3],
}

/// Restore a cloud in its saved parent coordinates. This adds no evidence.
/// The caller must verify the source record and alignment record identities.
#[wasm_bindgen]
pub fn restore_aligned_scene(scene_json: &str, alignment_json: &str) -> Result<String, JsValue> {
    restore(scene_json, alignment_json).map_err(JsValue::from)
}

fn restore(scene_json: &str, alignment_json: &str) -> Result<String, PreviewError> {
    let mut scene: Scene = serde_json::from_str(scene_json)?;
    let alignment: Alignment = serde_json::from_str(alignment_json)?;
    let [x, y, z, w] = alignment.source_to_target_xyzw;
    let rotation = Quaternion::new(w, x, y, z);
    if !alignment.scale.is_finite()
        || alignment.scale <= 0.0
        || !rotation.coords.iter().all(|v| v.is_finite())
        || (rotation.norm_squared() - 1.0).abs() > 1e-8
        || !alignment
            .translation_target_scene_units
            .iter()
            .all(|v| v.is_finite())
    {
        return Err(PreviewError::Input {
            reason: "invalid saved scene transform".into(),
        });
    }
    scene.transform(SceneTransform {
        scale: alignment.scale,
        rotation: UnitQuaternion::new_normalize(rotation),
        translation: alignment.translation_target_scene_units.into(),
    })?;
    Ok(serde_json::to_string(&scene)?)
}

#[cfg(test)]
mod tests;
