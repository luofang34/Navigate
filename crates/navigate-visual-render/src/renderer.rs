use crate::{AtlasLayout, ImageAtlas, RenderError, TerrainAtlas};
use navigate_visual::{
    CameraModel, CameraPose, LocalFrame, MapRevision, ReferenceView, RendererIdentity,
};

mod device;
mod dispatch;
mod parameters;
#[cfg(not(target_arch = "wasm32"))]
use navigate_visual::ReferenceRenderer;

/// Bounded resource and ray-marching settings for a resident renderer.
#[derive(Clone, Copy, Debug)]
pub struct RenderLimits {
    /// Maximum texture bytes in the resident scene, excluding driver overhead.
    pub scene_bytes: usize,
    /// Maximum pixels in one view, at most 16,777,216. Output and readback each use eight bytes per pixel.
    pub pixels: usize,
    /// Maximum optical-axis depth, in metres.
    pub far_m: f32,
    /// Logarithmic depth samples from 10 metres through `far_m`.
    pub steps: u32,
    /// Bisections of the first observed above-to-below terrain interval.
    pub refinements: u32,
}

/// Immutable map data and geographic frame to upload once.
pub struct RenderScene {
    /// Map identity retained in each returned reference.
    pub map: MapRevision,
    /// Coordinate model shared by camera poses and terrain rays.
    pub frame: LocalFrame,
    /// Visible imagery.
    pub imagery: ImageAtlas,
    /// Terrain elevations.
    pub terrain: TerrainAtlas,
}

/// Hardware GPU renderer with resident textures and reusable output storage.
pub struct GpuRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    parameters: wgpu::Buffer,
    output: wgpu::Buffer,
    readback: wgpu::Buffer,
    camera: CameraModel,
    frame: LocalFrame,
    map: MapRevision,
    imagery: AtlasLayout,
    terrain: AtlasLayout,
    limits: RenderLimits,
    identity: String,
    style_sha256: String,
    resident_texture_bytes: usize,
}
impl GpuRenderer {
    /// Upload map textures and compile the ray kernel on a hardware adapter.
    ///
    /// # Errors
    /// Rejects invalid inputs, exceeded limits, unavailable hardware, and GPU failures.
    pub async fn new(
        scene: RenderScene,
        camera: CameraModel,
        limits: RenderLimits,
    ) -> Result<Self, RenderError> {
        let bytes = validate(&scene, camera, limits)?;
        let dimension = scene
            .imagery
            .layout
            .size
            .into_iter()
            .chain(scene.terrain.layout.size)
            .max()
            .ok_or(RenderError::Invalid("empty scene dimensions"))?;
        let (device, queue, identity) = device::create(dimension).await?;
        device::upload(scene, camera, limits, bytes, device, queue, identity).await
    }

    /// Bytes occupied by scene textures, excluding output buffers and driver overhead.
    pub fn resident_texture_bytes(&self) -> usize {
        self.resident_texture_bytes
    }

    /// Bytes copied from the GPU for one view, before host image/depth unpacking.
    pub fn readback_bytes_per_view(&self) -> u64 {
        self.output.size()
    }

    /// Render a view without disk I/O or scene uploads. Browser readback remains asynchronous.
    ///
    /// # Errors
    /// Rejects invalid or out-of-range poses and GPU failures.
    pub async fn render(&mut self, pose: CameraPose) -> Result<ReferenceView, RenderError> {
        pose.validate()?;
        if pose.position.norm() > 100_000.0 {
            return Err(RenderError::Invalid(
                "camera exceeds the 100 km local scene radius",
            ));
        }
        let parameters = parameters::make(self, pose);
        self.queue
            .write_buffer(&self.parameters, 0, bytemuck::cast_slice(&parameters));
        self.dispatch();
        let (image, depth_m) = self.read().await?;
        Ok(ReferenceView {
            map: self.map.clone(),
            frame: self.frame,
            pose,
            image,
            depth_m,
        })
    }

    /// Selected adapter and shader identity.
    pub fn backend_identity(&self) -> &str {
        &self.identity
    }

    /// Code and style fingerprint, available on both native and browser hosts.
    pub fn renderer_identity(&self) -> RendererIdentity {
        RendererIdentity {
            revision: self.identity.clone(),
            style_sha256: self.style_sha256.clone(),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl ReferenceRenderer for GpuRenderer {
    type Error = RenderError;
    fn identity(&self) -> RendererIdentity {
        self.renderer_identity()
    }
    fn render_blocking(&mut self, pose: CameraPose) -> Result<ReferenceView, RenderError> {
        pollster::block_on(self.render(pose))
    }
}

fn validate(
    scene: &RenderScene,
    camera: CameraModel,
    limits: RenderLimits,
) -> Result<usize, RenderError> {
    camera.validate()?;
    if matches!(scene.frame, LocalFrame::Wgs84Enu { .. }) {
        return Err(RenderError::Invalid(
            "WGS84 frames require ellipsoid terrain rays",
        ));
    }
    let [lat, lon] = scene.frame.anchor();
    scene.frame.local([lat, lon, 0.0])?;
    let image = scene.imagery.validate()?;
    let terrain = scene.terrain.validate()?;
    let bytes = image
        .saturating_mul(2)
        .saturating_add(terrain.saturating_mul(4));
    if bytes > limits.scene_bytes
        || limits.scene_bytes == 0
        || limits.pixels == 0
        || limits.pixels > 16_777_216
        || (camera.width as usize).saturating_mul(camera.height as usize) > limits.pixels
        || !limits.far_m.is_finite()
        || !(20.0..=100_000.0).contains(&limits.far_m)
        || !(16..=256).contains(&limits.steps)
        || !(8..=24).contains(&limits.refinements)
        || scene.map.release_id.is_empty()
        || scene.map.manifest_sha256.len() != 64
        || !scene
            .map
            .manifest_sha256
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(RenderError::Invalid(
            "scene, map identity, or resource limits",
        ));
    }
    for x in [0.0, f64::from(camera.width - 1)] {
        for y in [0.0, f64::from(camera.height - 1)] {
            let ray = nalgebra::Vector3::new(
                (x - camera.cx) / camera.fx,
                (camera.cy - y) / camera.fy,
                -1.0,
            );
            if ray.norm() * f64::from(limits.far_m) > 200_000.0 {
                return Err(RenderError::Invalid(
                    "ray exceeds the 200 km terrain extent",
                ));
            }
        }
    }
    Ok(bytes)
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
