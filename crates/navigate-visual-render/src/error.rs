use thiserror::Error;

/// A reference could not be rendered within the declared device and resource limits.
#[derive(Debug, Error)]
pub enum RenderError {
    /// Input dimensions, coverage, or resources are invalid.
    #[error("invalid render input: {0}")]
    Invalid(&'static str),
    /// No hardware GPU can execute this scene.
    #[error("no hardware GPU adapter is available")]
    Adapter,
    /// A GPU device could not be created.
    #[error("create rendering device: {0}")]
    Device(#[from] wgpu::RequestDeviceError),
    /// A shader, resource, or dispatch failed validation.
    #[error("GPU render validation: {0}")]
    Gpu(#[from] wgpu::Error),
    /// A readback failed.
    #[error("read GPU reference: {0}")]
    Mapping(#[from] wgpu::BufferAsyncError),
    /// The device dropped a pending readback callback.
    #[error("GPU readback callback disconnected")]
    Disconnected(#[from] futures_channel::oneshot::Canceled),
    /// The camera, pose, or coordinate frame is invalid.
    #[error("render geometry: {0}")]
    Geometry(#[from] navigate_visual::VisualError),
}
