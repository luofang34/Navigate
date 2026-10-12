//! Expand uploaded gray bytes without changing the matching image representation.

use super::{profile::Profile, runtime::Compute};
use image::GrayImage;
use wgpu::util::DeviceExt;

fn packed(levels: &[GrayImage]) -> Vec<u8> {
    let count: usize = levels.iter().map(|level| level.as_raw().len()).sum();
    let mut pixels = Vec::with_capacity(count.next_multiple_of(4));
    for level in levels {
        pixels.extend_from_slice(level.as_raw());
    }
    pixels.resize(count.next_multiple_of(4), 0);
    pixels
}

impl Compute {
    pub(super) fn image_buffer(
        &self,
        levels: &[GrayImage],
        reference: bool,
        profile: &mut Profile,
        encoder: &mut wgpu::CommandEncoder,
    ) -> wgpu::Buffer {
        let (pack_stage, init_stage) = if reference {
            ("reference_image_pack", "reference_image_buffer_init")
        } else {
            ("query_image_pack", "query_image_buffer_init")
        };
        let packing = profile.timer();
        let pixels = packed(levels);
        let count: usize = levels.iter().map(|level| level.as_raw().len()).sum();
        profile.record(packing, pack_stage, pixels.len(), 0);
        let upload = profile.timer();
        let bytes = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gray pyramid bytes"),
                contents: &pixels,
                usage: wgpu::BufferUsages::STORAGE,
            });
        profile.record(upload, init_stage, pixels.len(), 1);
        let allocation = profile.timer();
        let expanded = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("expanded pyramid"),
            size: (count * 4) as u64,
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        profile.record(allocation, "expanded_image_allocation", count * 4, 1);
        let conversion = profile.timer();
        self.encode_expansion(&bytes, &expanded, encoder);
        profile.record(conversion, "image_expand_encode", 0, 0);
        expanded
    }

    fn encode_expansion(
        &self,
        bytes: &wgpu::Buffer,
        expanded: &wgpu::Buffer,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gray expansion inputs"),
            layout: &self.unpack.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bytes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: expanded.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gray expansion"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.unpack);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((bytes.size() / 4).div_ceil(256) as u32, 1, 1);
    }
}

#[cfg(test)]
mod tests;
