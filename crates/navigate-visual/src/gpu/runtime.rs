//! Device and pipeline lifetime for GPU matching.

use crate::VisualError;
use thiserror::Error;

#[derive(Debug, Error)]
pub(super) enum GpuError {
    #[error("no hardware compute adapter is available")]
    Adapter,
    #[error("cannot create a compute device")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error("compute shader validation failed")]
    Validation(#[from] wgpu::Error),
    #[error("cannot map GPU match output")]
    Mapping(#[from] wgpu::BufferAsyncError),
    #[error("GPU readback callback disconnected")]
    Disconnected(#[from] std::sync::mpsc::RecvError),
}

pub(super) fn failure(source: GpuError) -> VisualError {
    VisualError::Backend {
        backend: "wgpu-pyramidal-lk-v1".into(),
        source: Box::new(source),
    }
}

pub(super) struct Compute {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub pipeline: wgpu::ComputePipeline,
}

impl Compute {
    pub async fn new() -> Result<(Self, String), GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or(GpuError::Adapter)?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(GpuError::Adapter);
        }
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("visual matching"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                },
                None,
            )
            .await?;
        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pyramidal patch matching"),
            source: wgpu::ShaderSource::Wgsl(include_str!("tracking.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("pyramidal patch matching"),
            layout: None,
            module: &module,
            entry_point: "match_points",
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(error) = device.pop_error_scope().await {
            return Err(error.into());
        }
        Ok((
            Self {
                device,
                queue,
                pipeline,
            },
            format!("{:?}/{}", info.backend, info.name),
        ))
    }
}
