//! Upload prepared images, dispatch matching, and read correspondences.

use super::runtime::{Compute, GpuError};
use image::GrayImage;
use nalgebra::Vector2;
use wgpu::util::DeviceExt;

fn packed(levels: &[GrayImage]) -> Vec<f32> {
    levels
        .iter()
        .flat_map(|level| level.as_raw().iter().map(|v| f32::from(*v)))
        .collect()
}

fn parameters(levels: &[GrayImage], count: usize) -> [u32; 20] {
    let mut data = [0; 20];
    let mut offset = 0;
    for (index, level) in levels.iter().enumerate() {
        data[index * 4..index * 4 + 4].copy_from_slice(&[offset, level.width(), level.height(), 0]);
        offset += level.width() * level.height();
    }
    data[16] = count as u32;
    data[17] = levels.len() as u32;
    data
}

impl Compute {
    pub fn align_blocking(
        &self,
        source: &[GrayImage],
        target: &[GrayImage],
        points: &[Vector2<f64>],
    ) -> Result<Vec<[f32; 4]>, GpuError> {
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let buffers = self.inputs(source, target, points);
        let size = (points.len() * 16) as u64;
        let output = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matched points"),
            size,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matched point readback"),
            size,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        });
        self.dispatch(&buffers, &output, &readback, points.len() as u32);
        self.device.poll(wgpu::Maintain::Wait);
        if let Some(error) = pollster::block_on(self.device.pop_error_scope()) {
            return Err(error.into());
        }
        self.read_blocking(readback)
    }

    fn inputs(
        &self,
        source: &[GrayImage],
        target: &[GrayImage],
        points: &[Vector2<f64>],
    ) -> [wgpu::Buffer; 4] {
        let coords: Vec<[f32; 2]> = points.iter().map(|p| [p.x as f32, p.y as f32]).collect();
        let upload = |label, bytes, usage| {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage,
                })
        };
        [
            upload(
                "reference pyramid",
                bytemuck::cast_slice(&packed(source)),
                wgpu::BufferUsages::STORAGE,
            ),
            upload(
                "query pyramid",
                bytemuck::cast_slice(&packed(target)),
                wgpu::BufferUsages::STORAGE,
            ),
            upload(
                "reference corners",
                bytemuck::cast_slice(&coords),
                wgpu::BufferUsages::STORAGE,
            ),
            upload(
                "pyramid dimensions",
                bytemuck::cast_slice(&parameters(source, points.len())),
                wgpu::BufferUsages::UNIFORM,
            ),
        ]
    }

    fn dispatch(
        &self,
        inputs: &[wgpu::Buffer; 4],
        output: &wgpu::Buffer,
        readback: &wgpu::Buffer,
        count: u32,
    ) {
        let buffers: Vec<_> = inputs
            .iter()
            .chain([output])
            .enumerate()
            .map(|(index, buffer)| wgpu::BindGroupEntry {
                binding: index as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matching inputs"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &buffers,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("visual match"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(output, 0, readback, 0, u64::from(count) * 16);
        self.queue.submit([encoder.finish()]);
    }

    fn read_blocking(&self, buffer: wgpu::Buffer) -> Result<Vec<[f32; 4]>, GpuError> {
        let (send, receive) = std::sync::mpsc::sync_channel(1);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                send.send(result).ok();
            });
        self.device.poll(wgpu::Maintain::Wait);
        receive.recv()??;
        let mapped = buffer.slice(..).get_mapped_range();
        let points = mapped
            .as_chunks::<16>()
            .0
            .iter()
            .map(|bytes| {
                let value =
                    |i| f32::from_le_bytes([bytes[i], bytes[i + 1], bytes[i + 2], bytes[i + 3]]);
                [value(0), value(4), value(8), value(12)]
            })
            .collect();
        drop(mapped);
        buffer.unmap();
        Ok(points)
    }
}
