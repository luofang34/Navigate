//! Worker-owned image graph for bounded scene refinement and camera fitting.
use super::{Scene, graph, graph_value, resection};
use crate::{error::PreviewError, model::Pose};
use navigate_visual::{
    CameraModel, LocalScenePoint, LocalScenePose,
    reconstruction::{ImageTrackMerger, refit_scene_camera},
};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use wasm_bindgen::prelude::*;
#[derive(Deserialize)]
struct Initial {
    position_scene_units: [f64; 3],
    eye_to_scene_xyzw: [f64; 4],
}
struct FixedPoints {
    scene_sha256: String,
    points: Vec<LocalScenePoint>,
}
/// Source image links shared by one connected reconstruction candidate.
#[wasm_bindgen]
pub struct SceneTrackGraph {
    camera: CameraModel,
    merger: ImageTrackMerger,
    scene: Option<FixedPoints>,
}
#[wasm_bindgen]
impl SceneTrackGraph {
    /// Start a graph. Run this object in a worker.
    #[wasm_bindgen(constructor)]
    pub fn new(camera_json: &str) -> Result<SceneTrackGraph, JsValue> {
        Self::create(camera_json).map_err(JsValue::from)
    }
    /// Append a verified group in source order. Repeated group IDs are rejected.
    pub fn push(&mut self, group_sha256: &str, graph_json: &str) -> Result<(), JsValue> {
        self.append(group_sha256, graph_json).map_err(JsValue::from)
    }
    /// Select bounded camera observations and keep every selected source identity.
    pub fn select(
        &self,
        observations_json: &str,
        max_tracks: u32,
        cell_size_px: u32,
    ) -> Result<String, JsValue> {
        self.selection(observations_json, max_tracks, cell_size_px)
            .map_err(JsValue::from)
    }
    /// Hold estimated scene points for repeated camera fits. Returns the SHA-256
    /// of the exact scene JSON bytes. This does not establish scene accuracy.
    pub fn set_scene(&mut self, scene_json: &str) -> Result<String, JsValue> {
        self.set_points(scene_json).map_err(JsValue::from)
    }
    /// Fit one observation to the selected scene points. Each call is conditional
    /// and retains ambiguous feature IDs. Other scene alternatives stay separate.
    pub fn fit_camera(
        &self,
        observation_sha256: &str,
        initial_json: &str,
    ) -> Result<String, JsValue> {
        self.fit(observation_sha256, initial_json)
            .map_err(JsValue::from)
    }
}
impl SceneTrackGraph {
    fn create(camera_json: &str) -> Result<Self, PreviewError> {
        let camera = graph::camera(camera_json)?;
        Ok(Self {
            camera,
            merger: ImageTrackMerger::new(camera)?,
            scene: None,
        })
    }
    fn append(&mut self, group_sha256: &str, graph_json: &str) -> Result<(), PreviewError> {
        self.merger
            .push(group_sha256, &graph::decode(graph_json)?)?;
        self.scene = None;
        Ok(())
    }
    fn selection(
        &self,
        observations_json: &str,
        max_tracks: u32,
        cell_size_px: u32,
    ) -> Result<String, PreviewError> {
        let observations: Vec<String> = serde_json::from_str(observations_json)?;
        let selected = self
            .merger
            .select(&observations, max_tracks as usize, cell_size_px)?;
        let graph = graph_value(&selected.graph);
        let sources: Vec<_> = selected.graph.tracks.iter().zip(selected.source_tracks).map(|(t, sources)| json!({
            "feature_id":t.feature_id.to_string(),"source_tracks":sources.iter().map(|s| json!({"group_sha256":s.group_sha256,"feature_id":s.feature_id.to_string()})).collect::<Vec<_>>()
        })).collect();
        Ok(json!({"graph":graph,"association_sources":sources,"deferred_tracks":selected.deferred_tracks,
            "stage":"joined_image_associations","geographic_acceptance":false,
            "evidence_correlation":"unknown; overlapping groups reuse source observations"}).to_string())
    }
    fn set_points(&mut self, scene_json: &str) -> Result<String, PreviewError> {
        let scene: Scene = serde_json::from_str(scene_json)?;
        let points = scene.model()?.points;
        let observation =
            self.merger
                .observation_sha256()
                .first()
                .ok_or_else(|| PreviewError::Input {
                    reason: "scene points require a source graph".into(),
                })?;
        self.merger.scene_point_links(observation, &points)?;
        let scene_sha256 = format!("{:x}", Sha256::digest(scene_json.as_bytes()));
        self.scene = Some(FixedPoints {
            scene_sha256: scene_sha256.clone(),
            points,
        });
        Ok(scene_sha256)
    }
    fn fit(&self, observation: &str, initial_json: &str) -> Result<String, PreviewError> {
        let scene = self.scene.as_ref().ok_or_else(|| PreviewError::Input {
            reason: "camera fitting requires selected scene points".into(),
        })?;
        let initial: Initial = serde_json::from_str(initial_json)?;
        let pose = Pose {
            position_enu_m: initial.position_scene_units,
            eye_to_enu_xyzw: initial.eye_to_scene_xyzw,
        }
        .model()?;
        let links = self.merger.scene_point_links(observation, &scene.points)?;
        let fit = refit_scene_camera(
            &self.camera,
            &scene.scene_sha256,
            observation,
            LocalScenePose {
                position: pose.position,
                orientation: pose.orientation,
            },
            &links.matches,
        )?;
        let mut report = resection::report(&scene.scene_sha256, observation, fit);
        report["ambiguous_feature_ids"] = json!(
            links
                .ambiguous_feature_ids
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
        );
        report["match_count"] = json!(links.matches.len());
        Ok(report.to_string())
    }
}
#[cfg(test)]
mod tests;
