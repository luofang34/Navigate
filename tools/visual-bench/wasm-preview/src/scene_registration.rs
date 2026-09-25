//! Conditional scene registration against the active rendered reference.
use crate::{error::PreviewError, local_scene::Scene, model::Pose, preview::Preview};
use navigate_visual::reconstruction::{
    RegistrationConfig, SceneTransform, propose_scene_registration,
};
use navigate_visual::{CameraModel, CameraPose, LocalFrame, LocalScene, PixelMatch, ReferenceView};
use serde::Deserialize;
use serde_json::{Value, json};
use wasm_bindgen::prelude::*;
#[derive(Deserialize)]
struct Pair {
    reference: [f64; 2],
    query: [f64; 2],
}
#[derive(Deserialize)]
struct Registration {
    scale: f64,
    scene_to_map_xyzw: [f64; 4],
    translation_enu_m: [f64; 3],
    anchor_lat_lon: [f64; 2],
}
impl Registration {
    fn model(&self) -> Result<(SceneTransform, LocalFrame), PreviewError> {
        if !self.scale.is_finite() || self.scale <= 0.0 {
            return Err(PreviewError::Input {
                reason: "invalid scene scale".into(),
            });
        }
        let pose = Pose {
            position_enu_m: self.translation_enu_m,
            eye_to_enu_xyzw: self.scene_to_map_xyzw,
        }
        .model()?;
        Ok((
            SceneTransform {
                scale: self.scale,
                rotation: pose.orientation,
                translation: pose.position,
            },
            LocalFrame::anchor_mercator(self.anchor_lat_lon[0], self.anchor_lat_lon[1])?,
        ))
    }
}
#[wasm_bindgen]
impl Preview {
    /// Propose scene registration from matcher pixels and the active reference.
    /// The caller retains scene identity and handles geographic acceptance separately.
    pub fn register_scene(
        &self,
        id: u32,
        scene_json: &str,
        pairs_json: &str,
        backend: &str,
    ) -> Result<String, JsValue> {
        let run = || -> Result<String, PreviewError> {
            let session = self
                .camera_session
                .as_ref()
                .ok_or_else(|| PreviewError::Input {
                    reason: "no active observation".into(),
                })?;
            let observation = session.active_observation(id)?;
            let reference = self.reference.as_ref().ok_or_else(|| PreviewError::Input {
                reason: "no rendered reference".into(),
            })?;
            let scene: Scene = serde_json::from_str(scene_json)?;
            let pairs: Vec<Pair> = serde_json::from_str(pairs_json)?;
            let pairs: Vec<_> = pairs
                .into_iter()
                .map(|p| PixelMatch {
                    reference: p.reference.into(),
                    query: p.query.into(),
                })
                .collect();
            Ok(register(
                &self.camera.model(),
                &scene.model()?,
                &observation,
                reference,
                &pairs,
                backend,
            )?
            .to_string())
        };
        run().map_err(JsValue::from)
    }
}
fn register(
    camera: &CameraModel,
    scene: &LocalScene,
    observation: &str,
    reference: &ReferenceView,
    pairs: &[PixelMatch],
    backend: &str,
) -> Result<Value, PreviewError> {
    let result = propose_scene_registration(
        camera,
        scene,
        observation,
        reference,
        pairs,
        RegistrationConfig::default(),
    )?;
    let anchor = reference.frame.geodetic(nalgebra::Vector3::zeros());
    let candidates = result.candidates.iter().map(|c| {
        let transform = c.transform;
        Ok(json!({"scale":transform.scale,"scene_to_map_xyzw":transform.rotation.coords.as_slice(),
            "translation_enu_m":transform.translation.as_slice(),"anchor_lat_lon":[anchor[0],anchor[1]],
            "inlier_indices":c.inlier_indices,"point_fit_rms_m":c.inlier_rms_m,"occupied_cells":c.occupied_cells,
            "cameras":map_cameras(scene,transform,reference.frame)?,"geographic_acceptance":false}))
    }).collect::<Result<Vec<_>,PreviewError>>()?;
    let associations: Vec<_> = result.associations.iter().map(|a| json!({"feature_id":a.feature_id.to_string(),
        "match_index":a.match_index,"query_pixel":a.query_pixel.as_slice(),"reference_pixel":a.reference_pixel.as_slice(),
        "scene_point":a.scene_point.as_slice(),"map_point":a.map_point.as_slice(),"association_distance_px":a.association_distance_px})).collect();
    let mut report = json!({"stage":"conditional_scene_registration","geographic_acceptance":false,
        "observation_sha256":result.observation_sha256,"map_release_id":result.map.release_id,"map_manifest_sha256":result.map.manifest_sha256,
        "backend":backend,"candidates":candidates,"associations":associations,"candidate_budget_exhausted":result.candidate_budget_exhausted,
        "geographic_accuracy":"not_independently_measured","uncertainty":"unknown; scene, calibration, map registration and surface errors",
        "evidence_correlation":"unknown; image associations and estimated geometry share evidence; no confidence accumulation"});
    crate::session::reference_provenance(&mut report, reference);
    Ok(report)
}
fn map_cameras(
    scene: &LocalScene,
    transform: SceneTransform,
    frame: LocalFrame,
) -> Result<Vec<Value>, PreviewError> {
    if scene.cameras.len() > 129 {
        return Err(PreviewError::Input {
            reason: "too many scene cameras".into(),
        });
    }
    scene.cameras.iter().map(|c| {
        let pose = transform.pose(c.pose);
        CameraPose { position: pose.position, orientation: pose.orientation }.validate()?;
        let [latitude,longitude,altitude] = frame.geodetic(pose.position);
        Ok(json!({"observation_sha256":c.observation_sha256,"position_enu_m":pose.position.as_slice(),
            "eye_to_enu_xyzw":pose.orientation.coords.as_slice(),"latitude_deg":latitude,"longitude_deg":longitude,"altitude_m":altitude}))
    }).collect()
}
/// Apply one conditional registration to cameras in that exact coordinate frame.
/// The host must verify the saved scene lineage before it calls this function.
#[wasm_bindgen]
pub fn registered_scene_cameras(
    scene_json: &str,
    registration_json: &str,
) -> Result<String, JsValue> {
    let run = || -> Result<String, PreviewError> {
        let scene: Scene = serde_json::from_str(scene_json)?;
        let registration: Registration = serde_json::from_str(registration_json)?;
        let (transform, frame) = registration.model()?;
        Ok(serde_json::to_string(&map_cameras(
            &scene.model()?,
            transform,
            frame,
        )?)?)
    };
    run().map_err(JsValue::from)
}
#[cfg(test)]
mod tests;
