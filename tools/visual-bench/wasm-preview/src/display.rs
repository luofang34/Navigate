use crate::{
    error::{PreviewError, render_error},
    model::{Camera, Pose},
    preview::Preview,
};
use maplibre::render::resource::SurfaceAcquireError;
use wasm_bindgen::prelude::*;

#[derive(Debug, thiserror::Error)]
enum AcquireCanvasError {
    #[error("no canvas surface")]
    NoSurface,
    #[error(transparent)]
    Surface(#[from] SurfaceAcquireError),
}

#[wasm_bindgen]
impl Preview {
    /// Loaded terrain elevation in the package datum; missing data remains unknown.
    pub fn terrain_elevation_cached(&self, latitude: f64, longitude: f64) -> Option<f64> {
        if !latitude.is_finite()
            || !longitude.is_finite()
            || latitude.abs() > 85.0
            || longitude.abs() > 180.0
        {
            return None;
        }
        self.map
            .terrain_elevation_at(maplibre::coords::LatLon::new(latitude, longitude))
    }
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
        let output = self
            .acquire_canvas()
            .map_err(|e| render_error("canvas acquisition", e))?;
        self.draw_to(pose.transform()?, Some(&output.texture), 1)?;
        self.map.queue().present(output);
        Ok(())
    }
}
impl Preview {
    /// Retries once after reconfiguration, as the renderer does for window surfaces.
    fn acquire_canvas(&mut self) -> Result<wgpu::SurfaceTexture, AcquireCanvasError> {
        for attempt in 0..2 {
            let surface = self.surface.as_ref().ok_or(AcquireCanvasError::NoSurface)?;
            let failure = match surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(texture) => return Ok(texture),
                // Reconfiguration requires every outstanding surface texture to be dropped.
                wgpu::CurrentSurfaceTexture::Suboptimal(_)
                | wgpu::CurrentSurfaceTexture::Outdated => SurfaceAcquireError::Outdated,
                wgpu::CurrentSurfaceTexture::Timeout => SurfaceAcquireError::Timeout,
                wgpu::CurrentSurfaceTexture::Occluded => SurfaceAcquireError::Occluded,
                wgpu::CurrentSurfaceTexture::Lost => SurfaceAcquireError::Lost,
                wgpu::CurrentSurfaceTexture::Validation => SurfaceAcquireError::Validation,
            };
            if failure != SurfaceAcquireError::Outdated || attempt > 0 {
                return Err(failure.into());
            }
            self.configure_surface();
        }
        Err(SurfaceAcquireError::Outdated.into())
    }

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
                    color_space: Default::default(),
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
