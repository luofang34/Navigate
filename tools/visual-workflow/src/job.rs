//! A localization job binds images, depth, calibration, and priors.
use crate::{Error, assets};
use navigate_recording::{Camera, Pose};
use navigate_visual::{Frame, FrameStamp, LocalFrame, MapRevision, PosePrior, ReferenceView};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Job {
    pub query: Image,
    pub camera: Camera,
    pub prior: Pose,
    pub position_radius_m: f64,
    pub attitude_radius_rad: f64,
    pub anchor_lat_lon: [f64; 2],
    pub map_release: String,
    pub map_sha256: String,
    pub references: Vec<Reference>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Image {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    pub id: String,
    pub image: Image,
    pub depth: PathBuf,
    pub depth_sha256: String,
    pub pose: Pose,
    pub features: Option<PathBuf>,
}
impl Job {
    pub fn query_blocking(&self, root: &Path, max_pixels: usize) -> Result<Frame, Error> {
        let image = assets::image_verified_blocking(
            &root.join(&self.query.path),
            max_pixels,
            Some(&self.query.sha256),
        )?;
        Ok(Frame {
            stamp: FrameStamp {
                sequence: 0,
                capture_time_ns: 0,
            },
            camera: self.camera.model(),
            image,
        })
    }
    pub fn prior(&self) -> Result<PosePrior, Error> {
        Ok(PosePrior {
            pose: self.prior.camera()?,
            position_radius_m: self.position_radius_m,
            attitude_radius_rad: self.attitude_radius_rad,
        })
    }
    pub fn reference_blocking(
        &self,
        entry: &Reference,
        root: &Path,
        max_pixels: usize,
    ) -> Result<ReferenceView, Error> {
        let image = assets::image_verified_blocking(
            &root.join(&entry.image.path),
            max_pixels,
            Some(&entry.image.sha256),
        )?;
        let bytes = assets::read_blocking(&root.join(&entry.depth), max_pixels.saturating_mul(4))?;
        assets::verify(&bytes, &entry.depth_sha256)?;
        if bytes.len() != image.as_raw().len() * 4 {
            return Err(Error::Input(format!("depth dimensions for {}", entry.id)));
        }
        let depth_m = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        Ok(ReferenceView {
            map: MapRevision {
                release_id: self.map_release.clone(),
                manifest_sha256: self.map_sha256.clone(),
            },
            frame: LocalFrame::anchor_mercator(self.anchor_lat_lon[0], self.anchor_lat_lon[1])?,
            pose: entry.pose.camera()?,
            image,
            depth_m,
        })
    }
}
