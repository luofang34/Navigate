//! Upload prepared images, dispatch matching, and read correspondences.

use super::{
    profile::Profile,
    runtime::{Compute, GpuError},
};
use image::GrayImage;
use nalgebra::Vector2;
use wgpu::util::DeviceExt;

pub(super) struct Aligned {
    pub points: Vec<[f32; 4]>,
    pub query: wgpu::Buffer,
}

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
        initial: Option<&[Vector2<f64>]>,
        resident: Option<wgpu::Buffer>,
        profile: &mut Profile,
    ) -> Result<Aligned, GpuError> {
        let total = profile.timer();
        self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let preparation = profile.timer();
        let buffers = self.inputs(source, target, points, initial, resident, profile);
        profile.record(preparation, "inputs_total", 0, 0);
        let (output, readback) = self.output_buffers(points.len(), profile);
        let pipeline = if initial.is_some() {
            &self.refinement
        } else {
            &self.pipeline
        };
        let groups = if initial.is_some() {
            points.len().div_ceil(64)
        } else {
            points.len()
        };
        let submission = profile.timer();
        self.dispatch(
            &buffers,
            &output,
            &readback,
            points.len() as u32,
            pipeline,
            groups as u32,
        );
        profile.record(submission, "encode_submit", 0, 0);
        let wait = profile.timer();
        self.device.poll(wgpu::Maintain::Wait);
        if let Some(error) = pollster::block_on(self.device.pop_error_scope()) {
            return Err(error.into());
        }
        profile.record(wait, "completion_wait", 0, 0);
        let result = self.read_blocking(readback, profile)?;
        profile.record(total, "align_total", 0, 0);
        let [_, query, _, _] = buffers;
        Ok(Aligned {
            points: result,
            query,
        })
    }

    fn output_buffers(&self, count: usize, profile: &mut Profile) -> (wgpu::Buffer, wgpu::Buffer) {
        let allocation = profile.timer();
        let size = (count * 16) as u64;
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
        profile.record(allocation, "output_buffer_allocation", size as usize * 2, 2);
        (output, readback)
    }

    fn inputs(
        &self,
        source: &[GrayImage],
        target: &[GrayImage],
        points: &[Vector2<f64>],
        initial: Option<&[Vector2<f64>]>,
        resident: Option<wgpu::Buffer>,
        profile: &mut Profile,
    ) -> [wgpu::Buffer; 4] {
        let coords: Vec<[f32; 2]> = points
            .iter()
            .chain(initial.into_iter().flatten())
            .map(|p| [p.x as f32, p.y as f32])
            .collect();
        let reference = if let Some(buffer) = resident {
            profile.record(profile.timer(), "reference_image_reuse", 0, 0);
            buffer
        } else {
            self.image_buffer(source, true, profile)
        };
        let query = self.image_buffer(target, false, profile);
        let upload = profile.timer();
        let corners = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reference corners"),
                contents: bytemuck::cast_slice(&coords),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let dimensions = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pyramid dimensions"),
                contents: bytemuck::cast_slice(&parameters(source, points.len())),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        profile.record(
            upload,
            "point_parameter_buffer_init",
            coords.len() * 8 + 80,
            2,
        );
        [reference, query, corners, dimensions]
    }

    fn image_buffer(
        &self,
        levels: &[GrayImage],
        reference: bool,
        profile: &mut Profile,
    ) -> wgpu::Buffer {
        let (label, pack_stage, init_stage) = if reference {
            (
                "reference pyramid",
                "reference_image_pack",
                "reference_image_buffer_init",
            )
        } else {
            (
                "query pyramid",
                "query_image_pack",
                "query_image_buffer_init",
            )
        };
        let packing = profile.timer();
        let pixels = packed(levels);
        let bytes = pixels.len() * 4;
        profile.record(packing, pack_stage, bytes, 0);
        let upload = profile.timer();
        let result = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(&pixels),
                usage: wgpu::BufferUsages::STORAGE,
            });
        profile.record(upload, init_stage, bytes, 1);
        result
    }

    fn dispatch(
        &self,
        inputs: &[wgpu::Buffer; 4],
        output: &wgpu::Buffer,
        readback: &wgpu::Buffer,
        count: u32,
        pipeline: &wgpu::ComputePipeline,
        groups: u32,
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
            layout: &pipeline.get_bind_group_layout(0),
            entries: &buffers,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("visual match"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        encoder.copy_buffer_to_buffer(output, 0, readback, 0, u64::from(count) * 16);
        self.queue.submit([encoder.finish()]);
    }

    fn read_blocking(
        &self,
        buffer: wgpu::Buffer,
        profile: &mut Profile,
    ) -> Result<Vec<[f32; 4]>, GpuError> {
        let wait = profile.timer();
        let (send, receive) = std::sync::mpsc::sync_channel(1);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                send.send(result).ok();
            });
        self.device.poll(wgpu::Maintain::Wait);
        receive.recv()??;
        profile.record(wait, "map_readback_wait", 0, 0);
        let decoding = profile.timer();
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
        profile.record(decoding, "readback_decode", buffer.size() as usize, 0);
        Ok(points)
    }
}
