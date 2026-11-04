use crate::{AtlasLayout, ImageAtlas, RenderError, TerrainAtlas};
use navigate_visual::{
    CameraModel, CameraPose, LocalFrame, MapRevision, ReferenceView, RendererIdentity,
};
use std::sync::Arc;

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

/// Shared GPU terrain scene owning the device, compute pipeline, and resident atlas textures.
///
/// Wrap one `GpuScene` in an [`Arc`] so multiple worker viewports share one set of GPU textures.
pub struct GpuScene {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_layout: wgpu::BindGroupLayout,
    image_texture: wgpu::Texture,
    height_texture: wgpu::Texture,
    frame: LocalFrame,
    map: MapRevision,
    imagery: AtlasLayout,
    terrain: AtlasLayout,
    identity: String,
    resident_texture_bytes: usize,
}

impl GpuScene {
    /// Upload map textures and compile the ray kernel once on a hardware GPU adapter.
    ///
    /// # Errors
    /// Rejects invalid scene data, exceeded `max_scene_bytes`, unavailable hardware, and GPU errors.
    pub async fn new(scene: RenderScene, max_scene_bytes: usize) -> Result<Self, RenderError> {
        let bytes = validate_scene(&scene, max_scene_bytes)?;
        let dimension = scene
            .imagery
            .layout
            .size
            .into_iter()
            .chain(scene.terrain.layout.size)
            .max()
            .ok_or(RenderError::Invalid("empty scene dimensions"))?;
        let (device, queue, identity) = device::create(dimension).await?;
        device::upload_scene(scene, bytes, device, queue, identity).await
    }

    /// Upload map textures and compile the ray kernel on a caller-supplied GPU device.
    ///
    /// # Errors
    /// Rejects invalid scene data, exceeded `max_scene_bytes`, and GPU validation errors.
    pub async fn with_device(
        scene: RenderScene,
        max_scene_bytes: usize,
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_label: impl Into<String>,
    ) -> Result<Self, RenderError> {
        let bytes = validate_scene(&scene, max_scene_bytes)?;
        let identity = format!("wgpu-terrain-v1/{}", adapter_label.into());
        device::upload_scene(scene, bytes, device, queue, identity).await
    }

    /// Borrow the underlying `wgpu::Device`.
    #[must_use]
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }

    /// Borrow the underlying `wgpu::Queue`.
    #[must_use]
    pub fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    /// Return the coordinate frame of the scene.
    #[must_use]
    pub fn frame(&self) -> LocalFrame {
        self.frame
    }

    /// Borrow the map revision of the scene.
    #[must_use]
    pub fn map(&self) -> &MapRevision {
        &self.map
    }

    /// Return the imagery atlas layout.
    #[must_use]
    pub fn imagery_layout(&self) -> AtlasLayout {
        self.imagery
    }

    /// Return the terrain atlas layout.
    #[must_use]
    pub fn terrain_layout(&self) -> AtlasLayout {
        self.terrain
    }

    /// Return the adapter and shader identity string.
    #[must_use]
    pub fn backend_identity(&self) -> &str {
        &self.identity
    }

    /// Return the byte count occupied by resident scene textures.
    #[must_use]
    pub fn resident_texture_bytes(&self) -> usize {
        self.resident_texture_bytes
    }
}

/// Alias for a per-worker GPU viewport bound to a [`GpuScene`].
pub type GpuViewport = GpuRenderer;

/// Hardware GPU renderer with a shared [`GpuScene`] and per-viewport output storage.
pub struct GpuRenderer {
    scene: Arc<GpuScene>,
    bindings: wgpu::BindGroup,
    parameters: wgpu::Buffer,
    output: wgpu::Buffer,
    readback: wgpu::Buffer,
    camera: CameraModel,
    limits: RenderLimits,
    identity: String,
    style_sha256: String,
}

#[cfg(not(target_arch = "wasm32"))]
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<GpuScene>();
};

impl GpuRenderer {
    /// Upload map textures and compile the ray kernel on a hardware adapter.
    ///
    /// # Errors
    /// Rejects invalid inputs, exceeded limits, unavailable hardware, and GPU failures.
    #[cfg_attr(target_arch = "wasm32", allow(clippy::arc_with_non_send_sync))]
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
        let gpu_scene = device::upload_scene(scene, bytes, device, queue, identity).await?;
        Self::from_scene(Arc::new(gpu_scene), camera, limits)
    }

    /// Upload map textures and compile the ray kernel on a caller-supplied GPU device.
    ///
    /// # Errors
    /// Rejects invalid inputs, scenes that exceed `device` limits, and GPU validation failures.
    #[cfg_attr(target_arch = "wasm32", allow(clippy::arc_with_non_send_sync))]
    pub async fn with_device(
        scene: RenderScene,
        camera: CameraModel,
        limits: RenderLimits,
        device: wgpu::Device,
        queue: wgpu::Queue,
        adapter_label: impl Into<String>,
    ) -> Result<Self, RenderError> {
        let bytes = validate(&scene, camera, limits)?;
        let identity = format!("wgpu-terrain-v1/{}", adapter_label.into());
        let gpu_scene = device::upload_scene(scene, bytes, device, queue, identity).await?;
        Self::from_scene(Arc::new(gpu_scene), camera, limits)
    }

    /// Create a lightweight per-worker viewport over an existing [`GpuScene`].
    ///
    /// This constructor allocates only parameter, output, and readback buffers for `camera`.
    /// It does not re-upload imagery or terrain textures.
    ///
    /// # Errors
    /// Rejects invalid intrinsics, out-of-range limits, or dimensions that exceed GPU limits.
    pub fn from_scene(
        scene: Arc<GpuScene>,
        camera: CameraModel,
        limits: RenderLimits,
    ) -> Result<Self, RenderError> {
        let identity = scene.identity.clone();
        device::create_viewport(scene, camera, limits, identity)
    }

    /// Create a per-worker viewport over `scene` with a custom backend identity label.
    ///
    /// # Errors
    /// Rejects invalid intrinsics, out-of-range limits, or dimensions that exceed GPU limits.
    pub fn from_scene_with_label(
        scene: Arc<GpuScene>,
        camera: CameraModel,
        limits: RenderLimits,
        role_label: impl Into<String>,
    ) -> Result<Self, RenderError> {
        let base = scene
            .identity
            .strip_prefix("wgpu-terrain-v1/")
            .unwrap_or(&scene.identity);
        let identity = format!("wgpu-terrain-v1/{}/{base}", role_label.into());
        device::create_viewport(scene, camera, limits, identity)
    }

    /// Borrow the shared [`GpuScene`] backing this viewport.
    #[must_use]
    pub fn scene(&self) -> &Arc<GpuScene> {
        &self.scene
    }

    /// Update camera intrinsics and resource limits without re-uploading resident textures.
    ///
    /// Reallocates output and readback buffers only when `(camera.width, camera.height)` changes.
    ///
    /// # Errors
    /// Rejects invalid intrinsics, out-of-range limits, or dimensions exceeding `device` limits.
    pub fn set_camera(
        &mut self,
        camera: CameraModel,
        limits: RenderLimits,
    ) -> Result<(), RenderError> {
        validate_camera_and_limits(camera, limits, self.scene.resident_texture_bytes)?;
        let output_bytes = check_viewport_limits(&self.scene.device, camera)?;
        if (camera.width, camera.height) != (self.camera.width, self.camera.height) {
            let (output, readback) = device::output_buffers(&self.scene.device, output_bytes);
            self.bindings = device::bindings(
                &self.scene.device,
                &self.scene.bind_layout,
                &self.parameters,
                &output,
                &self.scene.image_texture,
                &self.scene.height_texture,
            );
            self.output = output;
            self.readback = readback;
        }
        self.camera = camera;
        self.limits = limits;
        self.style_sha256 = device::compute_style_sha256(
            self.scene.frame,
            self.scene.imagery,
            self.scene.terrain,
            camera,
            limits,
        );
        Ok(())
    }

    /// Borrow the underlying `wgpu::Device` so another pipeline can share it.
    pub fn device(&self) -> &wgpu::Device {
        &self.scene.device
    }

    /// Borrow the underlying `wgpu::Queue` so another pipeline can share it.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.scene.queue
    }

    /// Bytes occupied by scene textures, excluding output buffers and driver overhead.
    pub fn resident_texture_bytes(&self) -> usize {
        self.scene.resident_texture_bytes
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
        self.scene
            .queue
            .write_buffer(&self.parameters, 0, bytemuck::cast_slice(&parameters));
        let submission = self.dispatch();
        let (image, depth_m) = self.read(submission).await?;
        Ok(ReferenceView {
            map: self.scene.map.clone(),
            frame: self.scene.frame,
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

pub(super) fn check_viewport_limits(
    device: &wgpu::Device,
    camera: CameraModel,
) -> Result<u64, RenderError> {
    let output_bytes = u64::from(camera.width)
        .checked_mul(u64::from(camera.height))
        .and_then(|n| n.checked_mul(8))
        .ok_or(RenderError::Invalid("output byte count overflow"))?;
    let maximum = device.limits();
    if output_bytes > maximum.max_storage_buffer_binding_size
        || camera.width.div_ceil(8) > maximum.max_compute_workgroups_per_dimension
        || camera.height.div_ceil(8) > maximum.max_compute_workgroups_per_dimension
    {
        return Err(RenderError::Invalid("scene exceeds selected GPU limits"));
    }
    Ok(output_bytes)
}

fn validate_scene(scene: &RenderScene, max_scene_bytes: usize) -> Result<usize, RenderError> {
    if max_scene_bytes == 0 {
        return Err(RenderError::Invalid(
            "scene, map identity, or resource limits",
        ));
    }
    crate::atlas::require_tile_frame(scene.frame)?;
    let [lat, lon] = scene.frame.anchor();
    scene.frame.local([lat, lon, 0.0])?;
    let image = scene.imagery.validate()?;
    let terrain = scene.terrain.validate()?;
    let bytes = image
        .saturating_mul(2)
        .saturating_add(terrain.saturating_mul(4));
    if bytes > max_scene_bytes
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
    Ok(bytes)
}

fn validate(
    scene: &RenderScene,
    camera: CameraModel,
    limits: RenderLimits,
) -> Result<usize, RenderError> {
    let bytes = validate_scene(scene, limits.scene_bytes)?;
    validate_camera_and_limits(camera, limits, bytes)?;
    Ok(bytes)
}

fn validate_camera_and_limits(
    camera: CameraModel,
    limits: RenderLimits,
    resident_bytes: usize,
) -> Result<(), RenderError> {
    camera.validate()?;
    if resident_bytes > limits.scene_bytes
        || limits.scene_bytes == 0
        || limits.pixels == 0
        || limits.pixels > 16_777_216
        || (camera.width as usize).saturating_mul(camera.height as usize) > limits.pixels
        || !limits.far_m.is_finite()
        || !(20.0..=100_000.0).contains(&limits.far_m)
        || !(16..=256).contains(&limits.steps)
        || !(8..=24).contains(&limits.refinements)
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
    Ok(())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests;
