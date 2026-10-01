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
        let valid = |x: usize, y: usize, d: f32| {
            let world = camera.unproject(&pose, Vector2::new(x as f64, y as f64), f64::from(d));
            let xy = self.frame.mercator_xy(world);
            self.coverage.supports(xy)
                && self
                    .map
                    .rendered_terrain_sample_cached(xy)
                    .is_some_and(|t| t.covered && t.dem_loaded)
        };
        mask_blocks(depth, camera.width as usize, camera.height as usize, valid);
    }
}

/// Mask blocks are evaluated at their corners; coverage and DEM availability vary at tile scale.
const MASK_BLOCK: usize = 8;

/// Zeroes depth where `valid` rejects the surface point. A block whose four corners are valid keeps
/// its pixels and a block whose four corners are invalid is cleared; mixed blocks are checked pixel
/// by pixel. Clearing can only drop depth, so no rejected surface point is kept.
fn mask_blocks(
    depth: &mut [f32],
    width: usize,
    height: usize,
    valid: impl Fn(usize, usize, f32) -> bool,
) {
    let check = |depth: &[f32], x: usize, y: usize| {
        let d = depth[y * width + x];
        d > 0.0 && valid(x, y, d)
    };
    let columns = width.div_ceil(MASK_BLOCK) + 1;
    let rows = height.div_ceil(MASK_BLOCK) + 1;
    let corner = |i: usize, extent: usize| (i * MASK_BLOCK).min(extent - 1);
    let corners: Vec<bool> = (0..rows * columns)
        .map(|k| {
            check(
                depth,
                corner(k % columns, width),
                corner(k / columns, height),
            )
        })
        .collect();
    for by in 0..rows - 1 {
        for bx in 0..columns - 1 {
            let block = [(bx, by), (bx + 1, by), (bx, by + 1), (bx + 1, by + 1)]
                .map(|(cx, cy)| corners[cy * columns + cx]);
            let whole = block.iter().all(|&v| v);
            let none = block.iter().all(|&v| !v);
            for y in by * MASK_BLOCK..((by + 1) * MASK_BLOCK).min(height) {
                for x in bx * MASK_BLOCK..((bx + 1) * MASK_BLOCK).min(width) {
                    let i = y * width + x;
                    if depth[i] > 0.0 && (none || !(whole || check(depth, x, y))) {
                        depth[i] = 0.0;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
