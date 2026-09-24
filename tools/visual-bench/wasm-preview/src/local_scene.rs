//! JSON boundary for conditional Rust scene refinement in a worker.
use crate::{error::PreviewError, model::Camera};
use nalgebra::{Quaternion, UnitQuaternion};
use navigate_visual::{
    LocalScene, LocalSceneCamera, LocalScenePoint, LocalScenePose, SceneCoordinateGauge,
    ScenePointObservation, refine_local_scene, refine_local_scene_with_gauge,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use wasm_bindgen::prelude::*;
#[derive(Deserialize, Serialize)]
struct Scene {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    coordinate_gauge: Option<CoordinateGauge>,
    cameras: Vec<SceneCamera>,
    points: Vec<ScenePoint>,
}
#[derive(Deserialize, Serialize)]
struct CoordinateGauge {
    origin_camera: usize,
    scale_camera: usize,
}
#[derive(Deserialize, Serialize)]
struct SceneCamera {
    observation_sha256: String,
    position_scene_units: [f64; 3],
    eye_to_scene_xyzw: [f64; 4],
    fixed: bool,
}
#[derive(Deserialize, Serialize)]
struct ScenePoint {
    feature_id: String,
    position_scene_units: [f64; 3],
    observations: Vec<Observation>,
}
#[derive(Deserialize, Serialize)]
struct Observation {
    camera_index: usize,
    pixel: [f64; 2],
}
impl Scene {
    fn model(&self) -> Result<LocalScene, PreviewError> {
        let cameras = self
            .cameras
            .iter()
            .enumerate()
            .map(|(index, c)| {
                let [x, y, z, w] = c.eye_to_scene_xyzw;
                let q = Quaternion::new(w, x, y, z);
                if !q.coords.iter().all(|v| v.is_finite()) || (q.norm_squared() - 1.0).abs() > 1e-8
                {
                    return Err(PreviewError::Input {
                        reason: format!("invalid local scene rotation at camera {index}"),
                    });
                }
                Ok(LocalSceneCamera {
                    observation_sha256: c.observation_sha256.clone(),
                    pose: LocalScenePose {
                        position: c.position_scene_units.into(),
                        orientation: UnitQuaternion::new_normalize(q),
                    },
                    fixed: c.fixed,
                })
            })
            .collect::<Result<Vec<_>, PreviewError>>()?;
        let points = self
            .points
            .iter()
            .map(|p| {
                let feature_id =
                    p.feature_id
                        .parse()
                        .map_err(|source| PreviewError::ScenePointId {
                            id: p.feature_id.clone(),
                            source,
                        })?;
                Ok(LocalScenePoint {
                    feature_id,
                    position: p.position_scene_units.into(),
                    observations: p
                        .observations
                        .iter()
                        .map(|o| ScenePointObservation {
                            camera_index: o.camera_index,
                            pixel: o.pixel.into(),
                        })
                        .collect(),
                })
            })
            .collect::<Result<Vec<_>, PreviewError>>()?;
        Ok(LocalScene { cameras, points })
    }
    fn update(&mut self, scene: &LocalScene) {
        for (target, source) in self.cameras.iter_mut().zip(&scene.cameras) {
            if !target.fixed {
                target.position_scene_units = source.pose.position.into();
                target.eye_to_scene_xyzw = source.pose.orientation.coords.into();
            }
        }
        for (target, source) in self.points.iter_mut().zip(&scene.points) {
            target.position_scene_units = source.position.into()
        }
    }
}
/// Refine one initialized local scene. Call this synchronous function in a worker.
/// The result is conditional geometry and does not accept a geographic position.
#[wasm_bindgen]
pub fn refine_scene(
    camera_json: String,
    scene_json: String,
    max_iterations: u32,
) -> Result<String, JsValue> {
    run(&camera_json, &scene_json, max_iterations).map_err(JsValue::from)
}
fn run(camera_json: &str, scene_json: &str, max_iterations: u32) -> Result<String, PreviewError> {
    let camera: Camera = serde_json::from_str(camera_json)?;
    camera.validate()?;
    let mut source: Scene = serde_json::from_str(scene_json)?;
    let mut scene = source.model()?;
    let result = if let Some(gauge) = &source.coordinate_gauge {
        refine_local_scene_with_gauge(
            &camera.model(),
            &mut scene,
            SceneCoordinateGauge {
                origin_camera: gauge.origin_camera,
                scale_camera: gauge.scale_camera,
            },
            max_iterations as usize,
        )?
    } else {
        refine_local_scene(&camera.model(), &mut scene, max_iterations as usize)?
    };
    let uncertainty = if source.coordinate_gauge.is_some() {
        "unknown; arbitrary coordinate gauge and estimated calibration; no measured scale"
    } else {
        "unknown; conditional on fixed estimated poses and calibration"
    };
    source.update(&scene);
    Ok(json!({"stage":"local_scene_refinement","geographic_acceptance":false,
        "uncertainty":uncertainty,
        "evidence_correlation":"unknown; retained observations can share evidence",
        "initial_cost":result.initial_cost,"final_cost":result.final_cost,"optimizer_steps":result.steps,
        "scene":source}).to_string())
}
#[cfg(test)]
mod tests;
