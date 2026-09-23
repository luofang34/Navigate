use crate::{
    error::{PreviewError, render_error},
    model::Pose,
    preview::Preview,
};
use image::GrayImage;
use maplibre::headless::map::reference::ReferenceTarget;
use nalgebra::Vector2;
use navigate_visual::{MapRevision, ReferenceView};
use wasm_bindgen::prelude::*;
#[wasm_bindgen]
impl Preview {
    /// Render a candidate with optical depth for independent pose verification.
    pub async fn render_reference(
        &mut self,
        id: u32,
        pose_json: String,
    ) -> Result<Vec<u8>, JsValue> {
        self.reference = None;
        let session = self
            .camera_session
            .as_mut()
            .ok_or_else(|| PreviewError::Input {
                reason: "begin an observation first".into(),
            })?;
        session.invalidate(id)?;
        if self.globe {
            return Err(PreviewError::Input {
                reason: "reference geometry requires the package local frame".into(),
            }
            .into());
        }
        let pose: Pose = serde_json::from_str(&pose_json).map_err(PreviewError::from)?;
        let target = ReferenceTarget::new(&self.map, self.camera.intrinsics())
            .map_err(|e| render_error("reference target", e))?;
        target
            .draw(&mut self.map, self.anchor, pose.transform()?)
            .map_err(|e| render_error("reference frame", e))?;
        let render = target
            .read(&self.map)
            .await
            .map_err(|e| render_error("reference readback", e))?;
        let image: Vec<u8> = render
            .rgba
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| {
                ((77 * u32::from(p[0]) + 150 * u32::from(p[1]) + 29 * u32::from(p[2])) >> 8) as u8
            })
            .collect();
        let mut depth = render.depth_m;
        let pose = pose.model()?;
        let camera = self.camera.model();
        self.mask_depth(&mut depth, camera, pose);
        self.reference = Some(ReferenceView {
            map: MapRevision {
                release_id: self.manifest.release_id.clone(),
                manifest_sha256: self.manifest.pack_id.clone(),
            },
            frame: self.frame,
            pose,
            image: GrayImage::from_raw(camera.width, camera.height, image.clone()).ok_or_else(
                || PreviewError::Input {
                    reason: "invalid render length".into(),
                },
            )?,
            depth_m: depth,
        });
        Ok(image)
    }
}

impl Preview {
    fn mask_depth(
        &self,
        depth: &mut [f32],
        camera: navigate_visual::CameraModel,
        pose: navigate_visual::CameraPose,
    ) {
        for (i, d) in depth.iter_mut().enumerate() {
            if *d <= 0.0 {
                continue;
            }
            let p = Vector2::new(
                (i % camera.width as usize) as f64,
                (i / camera.width as usize) as f64,
            );
            let world = camera.unproject(&pose, p, f64::from(*d));
            let xy = self.frame.mercator_xy(world);
            let terrain = self.map.rendered_terrain_sample_cached(xy);
            if !self.coverage.supports(xy) || !terrain.is_some_and(|t| t.covered && t.dem_loaded) {
                *d = 0.0;
            }
        }
    }
}
