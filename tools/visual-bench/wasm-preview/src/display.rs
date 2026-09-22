use crate::{
    error::{PreviewError, render_error},
    model::{Camera, Pose},
    preview::Preview,
};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
impl Preview {
    /// Create a globe renderer that presents directly to a browser WebGPU canvas.
    pub async fn create_display(
        manifest_json: String,
        camera_json: String,
        read: js_sys::Function,
        canvas: web_sys::HtmlCanvasElement,
        format: String,
    ) -> Result<Preview, JsValue> {
        let format = match format.as_str() {
            "rgba8unorm" => wgpu::TextureFormat::Rgba8Unorm,
            "bgra8unorm" => wgpu::TextureFormat::Bgra8Unorm,
            _ => {
                return Err(PreviewError::Input {
                    reason: "unsupported canvas format".into(),
                }
                .into());
            }
        };
        Self::initialize(
            manifest_json,
            camera_json,
            read,
            true,
            Some((canvas, format)),
        )
        .await
        .map_err(Into::into)
    }
    /// Set display dimensions without changing an observation's calibration.
    pub fn resize_display(&mut self, camera_json: String) -> Result<(), JsValue> {
        let camera: Camera = serde_json::from_str(&camera_json).map_err(PreviewError::from)?;
        camera.validate()?;
        if self.surface.is_none() {
            return Err(PreviewError::Input {
                reason: "not a display renderer".into(),
            }
            .into());
        }
        self.camera = camera;
        self.map.resize(
            maplibre::window::PhysicalSize::new(camera.width, camera.height).ok_or_else(|| {
                PreviewError::Input {
                    reason: "invalid display dimensions".into(),
                }
            })?,
        );
        self.configure_surface();
        Ok(())
    }
    /// Submit a display frame without a GPU-to-CPU pixel copy.
    pub fn present(&mut self, pose_json: String) -> Result<(), JsValue> {
        let pose: Pose = serde_json::from_str(&pose_json).map_err(PreviewError::from)?;
        let surface = self.surface.as_ref().ok_or_else(|| PreviewError::Input {
            reason: "no canvas surface".into(),
        })?;
        let output = surface
            .get_current_texture()
            .map_err(|e| render_error("canvas acquisition", e))?;
        self.draw_to(pose.transform()?, Some(&output.texture), 1)?;
        output.present();
        Ok(())
    }
}
impl Preview {
    pub(crate) fn configure_surface(&mut self) {
        if let Some(surface) = &self.surface {
            surface.configure(
                self.map.device(),
                &wgpu::SurfaceConfiguration {
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    format: self
                        .map
                        .head_texture()
                        .map_or(wgpu::TextureFormat::Rgba8Unorm, |t| t.format()),
                    width: self.camera.width,
                    height: self.camera.height,
                    present_mode: wgpu::PresentMode::Fifo,
                    desired_maximum_frame_latency: 2,
                    alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                    view_formats: vec![],
                },
            );
        }
    }
}

pub(crate) fn surface(
    renderer: &maplibre::render::Renderer,
    canvas: Option<(web_sys::HtmlCanvasElement, wgpu::TextureFormat)>,
) -> Result<Option<wgpu::Surface<'static>>, PreviewError> {
    #[cfg(target_arch = "wasm32")]
    {
        canvas
            .map(|(canvas, _)| {
                renderer
                    .instance
                    .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            })
            .transpose()
            .map_err(|e| render_error("canvas surface", e))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = renderer;
        if canvas.is_some() {
            return Err(PreviewError::Input {
                reason: "browser canvas requires WASM".into(),
            });
        }
        Ok(None)
    }
}
