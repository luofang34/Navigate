//! Worker boundary for image links and conditional scene reconstruction.
use crate::{error::PreviewError, local_scene::Scene, model::Camera};
use nalgebra::{Quaternion, UnitQuaternion};
use navigate_visual::reconstruction::{
    DenseTrackBuilder, ImageTracks, ReconstructionSeed, propose_seeds, reconstruct,
};
use navigate_visual::{CameraModel, LocalScenePose, PixelMatch};
use serde::{Deserialize, Serialize};
use serde_json::json;
use wasm_bindgen::prelude::*;
mod fixed_structure;
mod graph;
mod point_alignment;
mod resection;
mod restored;
mod seeded_structure;
mod track_merge;
mod transform;
#[derive(Deserialize)]
struct Pair {
    reference: [f64; 2],
    query: [f64; 2],
}
#[derive(Serialize, Deserialize)]
struct Seed {
    camera_indices: [usize; 2],
    position_scene_units: [f64; 3],
    eye_to_scene_xyzw: [f64; 4],
}
impl Seed {
    fn model(&self) -> Result<ReconstructionSeed, PreviewError> {
        let [x, y, z, w] = self.eye_to_scene_xyzw;
        let q = Quaternion::new(w, x, y, z);
        if !q.coords.iter().all(|v| v.is_finite()) || (q.norm_squared() - 1.0).abs() > 1e-8 {
            return Err(PreviewError::Input {
                reason: "invalid reconstruction seed rotation".into(),
            });
        }
        Ok(ReconstructionSeed {
            camera_indices: self.camera_indices,
            second_pose: LocalScenePose {
                position: self.position_scene_units.into(),
                orientation: UnitQuaternion::new_normalize(q),
            },
        })
    }
}
/// A bounded set of uploaded image observations. Use this object in a worker.
#[wasm_bindgen]
pub struct SceneTracks {
    camera: CameraModel,
    builder: DenseTrackBuilder,
}
#[wasm_bindgen]
impl SceneTracks {
    /// Start a group. Dense correspondence proposals stay separate from geometry.
    #[wasm_bindgen(constructor)]
    pub fn new(camera_json: &str) -> Result<SceneTracks, JsValue> {
        Self::create(camera_json).map_err(JsValue::from)
    }
    /// Append a source digest and pairs from the immediately preceding image.
    pub fn push(&mut self, observation_sha256: String, pairs_json: &str) -> Result<(), JsValue> {
        self.append(observation_sha256, pairs_json)
            .map_err(JsValue::from)
    }
    /// Export source links. No confidence, geographic acceptance, or map update.
    pub fn snapshot(&self) -> String {
        graph_json(&self.builder.snapshot())
    }
    /// Keep all supported relative-pose proposals for the selected camera pair.
    pub fn propose(&self, first: u32, second: u32) -> Result<String, JsValue> {
        self.proposals([first as usize, second as usize])
            .map_err(JsValue::from)
    }
    /// Reconstruct one supplied seed; call separately for each alternative.
    pub fn reconstruct(&self, seed_json: &str) -> Result<String, JsValue> {
        self.solve(seed_json).map_err(JsValue::from)
    }
}
impl SceneTracks {
    fn create(camera_json: &str) -> Result<Self, PreviewError> {
        let camera: Camera = serde_json::from_str(camera_json)?;
        camera.validate()?;
        let camera = camera.model();
        Ok(Self {
            camera,
            builder: DenseTrackBuilder::new(camera)?,
        })
    }
    fn append(&mut self, digest: String, pairs_json: &str) -> Result<(), PreviewError> {
        let pairs: Vec<Pair> = serde_json::from_str(pairs_json)?;
        let pairs: Vec<_> = pairs
            .into_iter()
            .map(|p| PixelMatch {
                reference: p.reference.into(),
                query: p.query.into(),
            })
            .collect();
        Ok(self.builder.push(digest, &pairs)?)
    }
    fn proposals(&self, pair: [usize; 2]) -> Result<String, PreviewError> {
        proposals(&self.camera, &self.builder.snapshot(), pair)
    }
    fn solve(&self, seed_json: &str) -> Result<String, PreviewError> {
        solve(&self.camera, &self.builder.snapshot(), seed_json)
    }
}
fn graph_json(graph: &ImageTracks) -> String {
    graph_value(graph).to_string()
}
fn graph_value(graph: &ImageTracks) -> serde_json::Value {
    json!({"observation_sha256":graph.observation_sha256,"tracks":graph.tracks.iter().map(|t|json!({"feature_id":t.feature_id.to_string(),"observations":t.observations.iter().map(|o|json!({"camera_index":o.camera_index,"pixel":[o.pixel.x,o.pixel.y]})).collect::<Vec<_>>()})).collect::<Vec<_>>(),
        "stage":"image_associations","geographic_acceptance":false,
        "evidence_correlation":"unknown; reverse interpolation reuses the same pair proposals"})
}
#[cfg(test)]
mod tests;

fn proposals(
    camera: &CameraModel,
    graph: &ImageTracks,
    pair: [usize; 2],
) -> Result<String, PreviewError> {
    let proposals = propose_seeds(camera, graph, pair)?;
    Ok(json!(proposals.iter().map(|p|json!({"seed":Seed {camera_indices:p.seed.camera_indices,position_scene_units:p.seed.second_pose.position.into(),eye_to_scene_xyzw:p.seed.second_pose.orientation.coords.into()},"triangulated_points":p.triangulated_points})).collect::<Vec<_>>()).to_string())
}
fn solve(
    camera: &CameraModel,
    graph: &ImageTracks,
    seed_json: &str,
) -> Result<String, PreviewError> {
    let seed: Seed = serde_json::from_str(seed_json)?;
    let result = reconstruct(camera, graph, seed.model()?)?;
    Ok(json!({"stage":"local_scene_reconstruction","geographic_acceptance":false,
        "uncertainty":"unknown; arbitrary scale, estimated calibration and scene geometry",
        "evidence_correlation":"unknown; source links and estimated poses share observations",
        "source_camera_indices":result.source_camera_indices,"unresolved_camera_indices":result.unresolved_camera_indices,
        "initial_cost":result.refinement.initial_cost,"final_cost":result.refinement.final_cost,
        "optimizer_steps":result.refinement.steps,"scene":Scene::from_model(&result.scene,Some(result.coordinate_gauge))}).to_string())
}
