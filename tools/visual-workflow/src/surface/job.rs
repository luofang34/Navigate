//! Explicit local-coordinate observations; query truth is not an input field.
use crate::{Error, assets, config::Config, job::Image};
use navigate_recording::{Camera, Pose};
use navigate_visual::{
    Frame, FrameStamp, MapRevision, OrthographicReference, PosePrior, RasterGeometry,
};
use serde::{Deserialize, Serialize};
use std::path::Path;
mod window;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SurfaceJob {
    pub query: Image,
    pub camera: Camera,
    pub prior: Pose,
    pub position_radius_m: f64,
    pub attitude_radius_rad: f64,
    pub reference: Surface,
    #[serde(default = "default_rotations")]
    pub reference_quarter_turns: Vec<u8>,
    #[serde(default)]
    pub record_matches: bool,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Surface {
    pub image: Image,
    pub heights: Image,
    pub origin: [f64; 2],
    pub column: [f64; 2],
    pub row: [f64; 2],
    pub map_release: String,
    pub map_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<[u32; 4]>,
}
fn default_rotations() -> Vec<u8> {
    vec![0]
}
impl SurfaceJob {
    pub fn load_blocking(
        &self,
        root: &Path,
        config: &Config,
    ) -> Result<(Frame, OrthographicReference, PosePrior), Error> {
        let pixels = config.limits.image_pixels;
        let query = assets::image_verified_blocking(
            &root.join(&self.query.path),
            pixels,
            Some(&self.query.sha256),
        )?;
        self.camera.model().validate()?;
        if query.dimensions() != (self.camera.size[0], self.camera.size[1]) {
            return Err(Error::Input("surface query calibration dimensions".into()));
        }
        let reference = self.reference.load_blocking(root, config)?;
        let prior = PosePrior {
            pose: self.prior.camera()?,
            position_radius_m: self.position_radius_m,
            attitude_radius_rad: self.attitude_radius_rad,
        };
        prior.validate()?;
        Ok((
            Frame {
                stamp: FrameStamp {
                    sequence: 0,
                    capture_time_ns: 0,
                },
                camera: self.camera.model(),
                image: query,
            },
            reference,
            prior,
        ))
    }
}

impl Surface {
    fn load_blocking(&self, root: &Path, config: &Config) -> Result<OrthographicReference, Error> {
        let input = self;
        let image = assets::image_verified_blocking(
            &root.join(&input.image.path),
            config.limits.image_pixels,
            Some(&input.image.sha256),
        )?;
        let needed = image.as_raw().len().saturating_mul(5);
        if needed > config.limits.reference_bytes {
            return Err(Error::Input(
                "orthographic reference exceeds memory limit".into(),
            ));
        }
        let bytes = assets::read_blocking(
            &root.join(&input.heights.path),
            image.as_raw().len().saturating_mul(4),
        )?;
        assets::verify(&bytes, &input.heights.sha256)?;
        if bytes.len() != image.as_raw().len() * 4 {
            return Err(Error::Input("orthographic height dimensions".into()));
        }
        let heights_m = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]))
            .collect();
        let reference = OrthographicReference {
            map: MapRevision {
                release_id: input.map_release.clone(),
                manifest_sha256: input.map_sha256.clone(),
            },
            image,
            heights_m,
            geometry: RasterGeometry {
                origin: input.origin.into(),
                column: input.column.into(),
                row: input.row.into(),
            },
        };
        reference.validate()?;
        match input.window {
            Some(window) => window::crop(reference, window, config.limits.reference_bytes),
            None => Ok(reference),
        }
    }
}

#[cfg(test)]
mod tests;
