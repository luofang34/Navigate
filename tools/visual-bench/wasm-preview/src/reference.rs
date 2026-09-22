use crate::{error::PreviewError, model::Pose, preview::Preview, readback};
use image::GrayImage;
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
        self.draw(pose.transform()?)?;
        let rgba = readback::read(&self.map).await?;
        let bytes =
            readback::read_texture(&self.map, &self.depth, wgpu::TextureAspect::DepthOnly).await?;
        let image: Vec<u8> = rgba
            .as_chunks::<4>()
            .0
            .iter()
            .map(|p| {
                ((77 * u32::from(p[0]) + 150 * u32::from(p[1]) + 29 * u32::from(p[2])) >> 8) as u8
            })
            .collect();
        let frustum = self.camera.frustum();
        let mut depth: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .zip(rgba.as_chunks::<4>().0.iter())
            .map(|(b, c)| {
                let d = f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
                if d <= 0.0 || !d.is_finite() || c[3] != 255 {
                    0.0
                } else {
                    (frustum.near * frustum.far / (frustum.near + d * (frustum.far - frustum.near)))
                        as f32
                }
            })
            .collect();
        let pose = pose.model()?;
        let camera = self.camera.model();
        self.mask_depth(&mut depth, camera, pose);
        self.reference = Some(ReferenceView {
            map: MapRevision {
                release_id: self.manifest.release_id.clone(),
                manifest_sha256: self.manifest.pack_id.clone(),
            },
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
        let [lat, lon] = self.manifest.anchor_lat_lon;
        let ax = (lon + 180.0) / 360.0;
        let ay = (1.0 - lat.to_radians().tan().asinh() / std::f64::consts::PI) / 2.0;
        let scale = std::f64::consts::TAU * 6_371_008.8 * lat.to_radians().cos();
        for (i, d) in depth.iter_mut().enumerate() {
            if *d <= 0.0 {
                continue;
            }
            let p = Vector2::new(
                (i % camera.width as usize) as f64,
                (i / camera.width as usize) as f64,
            );
            let world = camera.unproject(&pose, p, f64::from(*d));
            let xy = [ax + world.x / scale, ay - world.y / scale];
            let terrain = self.map.rendered_terrain_sample_cached(xy);
            if !self.coverage.supports(xy) || !terrain.is_some_and(|t| t.covered && t.dem_loaded) {
                *d = 0.0;
            }
        }
    }
}
