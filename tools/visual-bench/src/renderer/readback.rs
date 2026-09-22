//! Copy rendered color or depth to CPU memory.

use crate::BenchError;
use maplibre::headless::map::HeadlessMap;

pub(super) fn read_blocking(
    map: &HeadlessMap,
    texture: &wgpu::Texture,
    aspect: wgpu::TextureAspect,
) -> Result<Vec<u8>, BenchError> {
    let row = texture.width() * 4;
    let padded = row.div_ceil(256) * 256;
    let buffer = map.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("visual readback"),
        size: u64::from(padded) * u64::from(texture.height()),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = map.device().create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect,
        },
        wgpu::ImageCopyBuffer {
            buffer: &buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: None,
            },
        },
        texture.size(),
    );
    map.queue().submit([encoder.finish()]);
    let (send, receive) = std::sync::mpsc::sync_channel(1);
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            send.send(result).ok();
        });
    map.device().poll(wgpu::Maintain::Wait);
    receive
        .recv()
        .map_err(|source| BenchError::Readback {
            source: Box::new(source),
        })?
        .map_err(|source| BenchError::Readback {
            source: Box::new(source),
        })?;
    let mapped = buffer.slice(..).get_mapped_range();
    let bytes = mapped
        .chunks_exact(padded as usize)
        .flat_map(|row_bytes| row_bytes[..row as usize].iter().copied())
        .collect();
    drop(mapped);
    buffer.unmap();
    Ok(bytes)
}
