//! Offline refinement of explicitly linked, conditional camera observations.
use crate::{Error, assets};
use navigate_recording::{Camera, Pose};
use navigate_visual::{
    CameraPose, LocalScene, LocalSceneCamera, LocalScenePoint, LocalScenePose,
    SceneCoordinateGauge, ScenePointObservation, ScenePointPrior, refine_calibrated_local_scene,
    refine_regularized_local_scene,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Scene {
    cameras: Vec<SceneCamera>,
    points: Vec<ScenePoint>,
    coordinate_gauge: Option<[usize; 2]>,
    iterations: usize,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct SceneCamera {
    observation_sha256: String,
    camera: Camera,
    pose: Pose,
    fixed: bool,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ScenePoint {
    feature_id: u64,
    position: [f64; 3],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    position_prior: Option<PositionPrior>,
    observations: Vec<Observation>,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PositionPrior {
    position: [f64; 3],
    scale: f64,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Observation {
    camera_index: usize,
    pixel: [f64; 2],
}
impl Scene {
    fn model(&self) -> Result<LocalScene, Error> {
        if self.cameras.len() > 128 || self.points.len() > 16384 {
            return Err(Error::Input(
                "scene exceeds 128 cameras or 16384 points".into(),
            ));
        }
        let cameras = self
            .cameras
            .iter()
            .map(|camera| {
                let pose = camera.pose.camera()?;
                Ok(LocalSceneCamera {
                    observation_sha256: camera.observation_sha256.clone(),
                    pose: LocalScenePose {
                        position: pose.position,
                        orientation: pose.orientation,
                    },
                    fixed: camera.fixed,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let points = self
            .points
            .iter()
            .map(|point| LocalScenePoint {
                feature_id: point.feature_id,
                position: point.position.into(),
                observations: point
                    .observations
                    .iter()
                    .map(|o| ScenePointObservation {
                        camera_index: o.camera_index,
                        pixel: o.pixel.into(),
                    })
                    .collect(),
            })
            .collect();
        Ok(LocalScene { cameras, points })
    }
    fn update(&mut self, model: &LocalScene) {
        for (camera, fitted) in self.cameras.iter_mut().zip(&model.cameras) {
            camera.pose = Pose::from(CameraPose {
                position: fitted.pose.position,
                orientation: fitted.pose.orientation,
            });
        }
        for (point, fitted) in self.points.iter_mut().zip(&model.points) {
            point.position = fitted.position.into();
        }
    }
}

pub fn run_blocking(input: &Path, output: &Path) -> Result<(), Error> {
    let bytes = assets::read_blocking(input, 4 * 1024 * 1024)?;
    let mut scene: Scene = serde_json::from_slice(&bytes).map_err(|source| Error::Json {
        path: input.into(),
        source,
    })?;
    let mut model = scene.model()?;
    let calibrations: Vec<_> = scene.cameras.iter().map(|c| c.camera.model()).collect();
    let gauge = scene
        .coordinate_gauge
        .map(|[origin_camera, scale_camera]| SceneCoordinateGauge {
            origin_camera,
            scale_camera,
        });
    let priors: Vec<_> = scene
        .points
        .iter()
        .filter_map(|point| {
            point.position_prior.as_ref().map(|prior| ScenePointPrior {
                feature_id: point.feature_id,
                position: prior.position.into(),
                scale: prior.scale,
            })
        })
        .collect();
    if gauge.is_some() && !priors.is_empty() {
        return Err(Error::Input(
            "position constraints cannot use an arbitrary coordinate gauge".into(),
        ));
    }
    let start = Instant::now();
    let result = if priors.is_empty() {
        refine_calibrated_local_scene(&calibrations, &mut model, scene.iterations, gauge)
    } else {
        refine_regularized_local_scene(&calibrations, &mut model, &priors, scene.iterations)
    }
    .map_err(|source| Error::Scene {
        path: input.into(),
        source,
    })?;
    let solve_ms = start.elapsed().as_secs_f64() * 1000.0;
    scene.update(&model);
    assets::write_json_blocking(
        output,
        &json!({
            "source_sha256":format!("{:x}", Sha256::digest(&bytes)),
            "scene":scene,"initial_cost":result.initial_cost,"final_cost":result.final_cost,
            "steps":result.steps,"solve_ms":solve_ms,
            "scope":"Conditional poses and structure with fixed intrinsics. This does not accept a navigation fix or establish geographic accuracy. Input links and source identities are supplied by the caller."
        }),
    )?;
    tracing::info!(path=%output.display(), solve_ms, "Saved conditional scene refinement");
    Ok(())
}

#[cfg(test)]
mod tests;
