//! Verify immutable tile catalogs and upload a bounded resident scene.
mod catalog;
mod tiles;

use crate::{Error, assets};
use navigate_visual::{CameraModel, LocalFrame, MapRevision};
use navigate_visual_render::{GpuRenderer, RenderLimits};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SceneConfig {
    pub catalog: PathBuf,
    pub catalog_sha256: String,
    pub imagery_tile_pixels: u32,
    pub scene_bytes: usize,
    pub far_m: f32,
    pub steps: u32,
    pub refinements: u32,
}

impl SceneConfig {
    pub fn load_blocking(
        &self,
        root: &Path,
        frame: LocalFrame,
        map: MapRevision,
        camera: CameraModel,
        max_pixels: usize,
    ) -> Result<GpuRenderer, Error> {
        if self.scene_bytes == 0
            || self.scene_bytes > 1_024 * 1_024 * 1_024
            || !self.imagery_tile_pixels.is_power_of_two()
            || self.imagery_tile_pixels > 256
        {
            return Err(Error::Input(
                "scene byte limit or imagery tile resolution".into(),
            ));
        }
        let path = root.join(&self.catalog);
        let bytes = assets::read_blocking(&path, 4 * 1024 * 1024)?;
        assets::verify(&bytes, &self.catalog_sha256)?;
        if map.manifest_sha256 != self.catalog_sha256 {
            return Err(Error::Input(
                "render map identity must bind the catalog".into(),
            ));
        }
        let catalog: catalog::Catalog =
            serde_json::from_slice(&bytes).map_err(|source| Error::Json {
                path: path.clone(),
                source,
            })?;
        let root = path
            .parent()
            .ok_or_else(|| Error::Input("catalog parent".into()))?;
        let scene = catalog.load_blocking(root, self, frame, map)?;
        Ok(pollster::block_on(GpuRenderer::new(
            scene,
            camera,
            RenderLimits {
                scene_bytes: self.scene_bytes,
                pixels: max_pixels,
                far_m: self.far_m,
                steps: self.steps,
                refinements: self.refinements,
            },
        ))?)
    }
}
