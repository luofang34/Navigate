use super::*;
use sha2::{Digest, Sha256};
use wgpu::util::DeviceExt;

pub(super) async fn create() -> Result<(wgpu::Device, wgpu::Queue, String), RenderError> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .ok_or(RenderError::Adapter)?;
    let info = adapter.get_info();
    if info.device_type == wgpu::DeviceType::Cpu {
        return Err(RenderError::Adapter);
    }
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("terrain reference"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
            },
            None,
        )
        .await?;
    Ok((
        device,
        queue,
        format!("wgpu-terrain-v1/{:?}/{}", info.backend, info.name),
    ))
}

pub(super) async fn upload(
    scene: RenderScene,
    camera: CameraModel,
    limits: RenderLimits,
    bytes: usize,
    device: wgpu::Device,
    queue: wgpu::Queue,
    identity: String,
) -> Result<GpuRenderer, RenderError> {
    let output_bytes = u64::from(camera.width) * u64::from(camera.height) * 8;
    let maximum = device.limits();
    if [scene.imagery.layout, scene.terrain.layout]
        .iter()
        .any(|grid| {
            grid.size
                .iter()
                .any(|n| *n > maximum.max_texture_dimension_2d)
        })
        || output_bytes > u64::from(maximum.max_storage_buffer_binding_size)
    {
        return Err(RenderError::Invalid("scene exceeds selected GPU limits"));
    }
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let (image_texture, height_texture) = scene_textures(&device, &queue, &scene);
    let (parameters, output, readback) = buffers(&device, output_bytes);
    let layout = layout(&device);
    let bindings = bindings(
        &device,
        &layout,
        &parameters,
        &output,
        &image_texture,
        &height_texture,
    );
    let pipeline = pipeline(&device, &layout);
    if let Some(error) = device.pop_error_scope().await {
        return Err(error.into());
    }
    let style = format!(
        "{:?}/{:?}/{:?}/{:?}/{:?}",
        scene.frame, scene.imagery.layout, scene.terrain.layout, camera, limits
    );
    let mut hash = Sha256::new();
    hash.update(include_str!("terrain.wgsl"));
    hash.update(style.as_bytes());
    let style_sha256 = format!("{:x}", hash.finalize());
    Ok(GpuRenderer {
        device,
        queue,
        pipeline,
        bindings,
        parameters,
        output,
        readback,
        camera,
        frame: scene.frame,
        map: scene.map,
        imagery: scene.imagery.layout,
        terrain: scene.terrain.layout,
        limits,
        identity,
        style_sha256,
        resident_texture_bytes: bytes,
    })
}

fn pipeline(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("curved terrain rays"),
        source: wgpu::ShaderSource::Wgsl(include_str!("terrain.wgsl").into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("terrain ray marcher"),
        module: &module,
        entry_point: "render",
        layout: Some(
            &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[layout],
                push_constant_ranges: &[],
            }),
        ),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    grid: AtlasLayout,
    format: wgpu::TextureFormat,
    bytes: &[u8],
    stride: u32,
) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width: grid.size[0],
        height: grid.size[1],
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("resident map grid"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(grid.size[0] * stride),
            rows_per_image: Some(grid.size[1]),
        },
        size,
    );
    texture
}

fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let buffer = |read_only| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    let texture = wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: false },
        view_dimension: wgpu::TextureViewDimension::D2,
        multisampled: false,
    };
    let entries: Vec<_> = [buffer(true), texture, texture, buffer(false)]
        .into_iter()
        .enumerate()
        .map(|(binding, ty)| wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("terrain resources"),
        entries: &entries,
    })
}

fn scene_textures(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    scene: &RenderScene,
) -> (wgpu::Texture, wgpu::Texture) {
    let imagery: Vec<u8> = scene
        .imagery
        .pixels
        .iter()
        .zip(&scene.imagery.coverage)
        .flat_map(|(p, c)| [*p, *c])
        .collect();
    let terrain: Vec<[f32; 2]> = scene
        .terrain
        .heights_m
        .iter()
        .zip(&scene.terrain.coverage)
        .map(|(h, c)| if *c == 255 { [*h, 1.0] } else { [0.0, 0.0] })
        .collect();
    let image_texture = texture(
        device,
        queue,
        scene.imagery.layout,
        wgpu::TextureFormat::Rg8Unorm,
        &imagery,
        2,
    );
    let height_texture = texture(
        device,
        queue,
        scene.terrain.layout,
        wgpu::TextureFormat::Rg32Float,
        bytemuck::cast_slice(&terrain),
        8,
    );
    (image_texture, height_texture)
}

fn buffers(device: &wgpu::Device, output_bytes: u64) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
    let parameters = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("terrain parameters"),
        contents: &[0; 256],
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain image and depth"),
        size: output_bytes,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain readback"),
        size: output_bytes,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    });
    (parameters, output, readback)
}

fn bindings(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    parameters: &wgpu::Buffer,
    output: &wgpu::Buffer,
    image_texture: &wgpu::Texture,
    height_texture: &wgpu::Texture,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("resident terrain bindings"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: parameters.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(
                    &image_texture.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(
                    &height_texture.create_view(&Default::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output.as_entire_binding(),
            },
        ],
    })
}
