//! Device and pipeline lifetime for GPU matching.

use crate::VisualError;
use thiserror::Error;

#[derive(Debug, Error)]
pub(super) enum GpuError {
    #[error("no hardware compute adapter is available")]
    Adapter,
    #[error("cannot request a compute adapter")]
    AdapterRequest(#[from] wgpu::RequestAdapterError),
    #[error("cannot create a compute device")]
    Device(#[from] wgpu::RequestDeviceError),
    #[error("compute shader validation failed")]
    Validation(#[from] wgpu::Error),
    #[error("cannot wait for submitted GPU work")]
    Poll(#[from] wgpu::PollError),
    #[error("cannot map GPU match output")]
    Mapping(#[from] wgpu::BufferAsyncError),
    #[error("cannot access mapped GPU match output")]
    MappedRange(#[from] wgpu::MapRangeError),
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
    pub refinement: wgpu::ComputePipeline,
    pub unpack: wgpu::ComputePipeline,
}

impl Compute {
    pub async fn new() -> Result<(Self, String), GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err(GpuError::Adapter);
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("visual matching"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                ..Default::default()
            })
            .await?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pyramidal patch matching"),
            source: wgpu::ShaderSource::Wgsl(include_str!("tracking.wgsl").into()),
        });
        let pipeline = pipeline(&device, &module, "match_points");
        let refinement = self::pipeline(&device, &module, "refine_points");
        let upload_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gray byte expansion"),
            source: wgpu::ShaderSource::Wgsl(include_str!("upload.wgsl").into()),
        });
        let unpack = self::pipeline(&device, &upload_module, "unpack_gray");
        if let Some(error) = scope.pop().await {
            return Err(error.into());
        }
        Ok((
            Self {
                device,
                queue,
                pipeline,
                refinement,
                unpack,
            },
            format!("{:?}/{}", info.backend, info.name),
        ))
    }
}

fn pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}
