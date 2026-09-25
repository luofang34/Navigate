//! JSON boundary for conditional camera resection.
use crate::{
    error::PreviewError,
    model::{Camera, Pose},
};
use navigate_visual::{
    LocalScenePose,
    reconstruction::{SceneCameraFit, ScenePointMatch, refit_scene_camera},
};
use serde::Deserialize;
use serde_json::json;
use wasm_bindgen::prelude::*;

#[derive(Deserialize)]
struct PointMatch {
    feature_id: String,
    position_scene_units: [f64; 3],
    pixel: [f64; 2],
}
#[derive(Deserialize)]
struct Request {
    scene_sha256: String,
    observation_sha256: String,
    position_scene_units: [f64; 3],
    eye_to_scene_xyzw: [f64; 4],
    matches: Vec<PointMatch>,
}
/// Fit a camera to one conditional scene. Run synchronous work in a worker.
/// Source identities remain explicit. The result is not a navigation fix.
#[wasm_bindgen]
pub fn resect_scene_camera(camera_json: &str, request_json: &str) -> Result<String, JsValue> {
    run(camera_json, request_json).map_err(JsValue::from)
}
fn run(camera_json: &str, request_json: &str) -> Result<String, PreviewError> {
    let camera: Camera = serde_json::from_str(camera_json)?;
    camera.validate()?;
    let request: Request = serde_json::from_str(request_json)?;
    let pose = Pose {
        position_enu_m: request.position_scene_units,
        eye_to_enu_xyzw: request.eye_to_scene_xyzw,
    }
    .model()?;
    let points = request
        .matches
        .iter()
        .map(|p| {
            let feature_id = p
                .feature_id
                .parse()
                .map_err(|source| PreviewError::ScenePointId {
                    id: p.feature_id.clone(),
                    source,
                })?;
            Ok(ScenePointMatch {
                feature_id,
                position: p.position_scene_units.into(),
                pixel: p.pixel.into(),
            })
        })
        .collect::<Result<Vec<_>, PreviewError>>()?;
    let fit = refit_scene_camera(
        &camera.model(),
        &request.scene_sha256,
        &request.observation_sha256,
        LocalScenePose {
            position: pose.position,
            orientation: pose.orientation,
        },
        &points,
    )?;
    Ok(report(&request.scene_sha256, &request.observation_sha256, fit).to_string())
}
pub(super) fn report(
    scene_sha256: &str,
    observation_sha256: &str,
    fit: Option<SceneCameraFit>,
) -> serde_json::Value {
    json!({"stage":"conditional_scene_resection","geographic_acceptance":false,
        "scene_sha256":scene_sha256,"observation_sha256":observation_sha256,
        "uncertainty":"unknown; conditional on estimated scene and calibration",
        "evidence_correlation":"unknown; scene and camera estimates share source observations",
        "fit":fit.map(|fit|json!({"position_scene_units":fit.pose.position.as_slice(),
            "eye_to_scene_xyzw":fit.pose.orientation.coords.as_slice(),
            "inlier_feature_ids":fit.inlier_feature_ids.iter().map(u64::to_string).collect::<Vec<_>>(),
            "reprojection_rms_px":fit.reprojection_rms_px}))})
}

#[cfg(test)]
mod tests;
