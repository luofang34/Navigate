//! Hash-bound inputs prevent mismatched frames from entering temporal diagnostics.
use super::{digest, geometry};
use crate::{Error, assets, job::Image};
use navigate_recording::{Camera, Pose};
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::Path;
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    pub frames: Vec<Sample>,
    pub runs: Vec<Image>,
    pub job: Image,
    pub lag: usize,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Sample {
    pub image: Image,
    pub capture_time_ns: u64,
}
#[derive(Deserialize)]
pub(super) struct Run {
    query_camera: Camera,
    query_transform: Option<Transform>,
    query_focal_scale: f64,
    frames: Vec<Row>,
}
#[derive(Deserialize)]
struct Transform {
    crop: [u32; 4],
    size: [u32; 2],
}
#[derive(Deserialize)]
struct Row {
    index: usize,
    capture_time_ns: u64,
    observation_sha256: String,
    accepted: bool,
    pose: Option<Pose>,
    evidence_kind: String,
    map_refreshed: bool,
}
impl Manifest {
    pub fn validate(&self) -> Result<(), Error> {
        if self.frames.len() < 2
            || self.frames.len() > 1000
            || self.runs.is_empty()
            || self.runs.len() > 8
            || self.lag == 0
            || self.lag > 10
            || self.lag >= self.frames.len()
            || self
                .frames
                .windows(2)
                .any(|p| p[1].capture_time_ns <= p[0].capture_time_ns)
        {
            return Err(Error::Input("motion probe bounds or timestamps".into()));
        }
        Ok(())
    }
    pub fn image_blocking(&self, root: &Path, index: usize) -> Result<image::GrayImage, Error> {
        let a = &self.frames[index].image;
        assets::image_verified_blocking(&root.join(&a.path), 4_000_000, Some(&a.sha256))
    }
}
impl Run {
    pub fn validate_camera(&self, job: &crate::job::Job) -> Result<(), Error> {
        let mut camera = job.camera.model();
        if let Some(t) = &self.query_transform {
            camera = camera.crop_and_resize(t.crop, t.size)?;
        }
        camera.fx *= self.query_focal_scale;
        camera.fy *= self.query_focal_scale;
        if camera != self.query_camera.model() {
            return Err(Error::Input(
                "pose camera differs from the declared job and image transform".into(),
            ));
        }
        Ok(())
    }
    pub fn camera(&self) -> navigate_visual::CameraModel {
        self.query_camera.model()
    }

    pub fn load_blocking(asset: &Image, root: &Path, count: usize) -> Result<Self, Error> {
        let path = root.join(&asset.path);
        let bytes = assets::read_blocking(&path, 32 * 1024 * 1024)?;
        assets::verify(&bytes, &asset.sha256)?;
        let run: Self =
            serde_json::from_slice(&bytes).map_err(|source| Error::Json { path, source })?;
        run.query_camera.model().validate()?;
        if run.frames.len() != count
            || run
                .frames
                .iter()
                .enumerate()
                .any(|(i, r)| r.index != i || r.accepted != r.pose.is_some())
        {
            return Err(Error::Input("motion probe pose/frame association".into()));
        }
        for row in &run.frames {
            if let Some(p) = &row.pose {
                p.camera()?.validate()?;
            }
        }
        Ok(run)
    }
    pub fn bind(
        &self,
        spec: &Manifest,
        index: usize,
        image: &image::GrayImage,
    ) -> Result<(), Error> {
        let row = &self.frames[index];
        let camera = self.query_camera.model();
        let time = spec.frames[index].capture_time_ns;
        if image.dimensions() != (camera.width, camera.height)
            || row.capture_time_ns != time
            || digest(image, camera, index, time) != row.observation_sha256
        {
            return Err(Error::Input(format!(
                "motion probe observation mismatch at frame {index}"
            )));
        }
        Ok(())
    }
    pub fn score(
        &self,
        a: usize,
        b: usize,
        pairs: &[(nalgebra::Vector2<f64>, nalgebra::Vector2<f64>)],
    ) -> Result<Value, Error> {
        if self.frames[a..=b].iter().any(|row| !row.accepted) {
            return Ok(
                json!({"available":false,"reason":"rejected pose within frame pair; continuity not established"}),
            );
        }
        match (&self.frames[a].pose, &self.frames[b].pose) {
            (Some(p), Some(q)) => {
                let a_pose = p.camera()?;
                let b_pose = q.camera()?;
                let mut doubled = b_pose;
                doubled.position = a_pose.position + 2.0 * (b_pose.position - a_pose.position);
                let mut rotated = b_pose;
                rotated.orientation *=
                    nalgebra::UnitQuaternion::from_euler_angles(0.05, 0.07, 0.03);
                Ok(
                    json!({"available":true,"source_evidence":self.frames[a].evidence_kind,"target_evidence":self.frames[b].evidence_kind,
                    "target_map_refreshed":self.frames[b].map_refreshed,"map_refreshes_in_span":self.frames[a+1..=b].iter().filter(|r|r.map_refreshed).count(),"measured":geometry::score(self.query_camera.model(),a_pose,b_pose,pairs),
                    "double_translation_control":geometry::score(self.query_camera.model(),a_pose,doubled,pairs),
                    "rotation_error_control":geometry::score(self.query_camera.model(),a_pose,rotated,pairs)}),
                )
            }
            _ => Ok(json!({"available":false,"reason":"one or both poses rejected"})),
        }
    }
}

#[cfg(test)]
mod tests;
