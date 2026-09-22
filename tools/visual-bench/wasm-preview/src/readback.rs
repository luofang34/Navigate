use crate::error::{PreviewError, render_error};
use maplibre::headless::map::HeadlessMap;
pub(crate) async fn read(map: &HeadlessMap) -> Result<Vec<u8>, PreviewError> {
    let texture = map.head_texture().ok_or_else(|| PreviewError::Input {
        reason: "renderer has no color target".into(),
    })?;
    read_texture(map, texture, wgpu::TextureAspect::All).await
}
pub(crate) async fn read_texture(
    map: &HeadlessMap,
    texture: &wgpu::Texture,
    aspect: wgpu::TextureAspect,
) -> Result<Vec<u8>, PreviewError> {
    let row = texture.width() * 4;
    let padded = row.div_ceil(256) * 256;
    let buffer = map.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("WASM preview readback"),
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
    let (send, receive) = futures_channel::oneshot::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            send.send(result).ok();
        });
    receive
        .await
        .map_err(|e| render_error("readback callback", e))?
        .map_err(|e| render_error("readback mapping", e))?;
    let mapped = buffer.slice(..).get_mapped_range();
    let bytes = mapped
        .chunks_exact(padded as usize)
        .flat_map(|r| r[..row as usize].iter().copied())
        .collect();
    drop(mapped);
    buffer.unmap();
    Ok(bytes)
}
