use super::*;
use futures_channel::oneshot;
use image::GrayImage;

impl GpuRenderer {
    pub(super) fn dispatch(&self) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("terrain reference"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bindings, &[]);
            pass.dispatch_workgroups(
                self.camera.width.div_ceil(8),
                self.camera.height.div_ceil(8),
                1,
            );
        }
        encoder.copy_buffer_to_buffer(&self.output, 0, &self.readback, 0, self.output.size());
        self.queue.submit([encoder.finish()]);
    }

    pub(super) async fn read(&self) -> Result<(GrayImage, Vec<f32>), RenderError> {
        let (send, receive) = oneshot::channel();
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                send.send(result).ok();
            });
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::Maintain::Wait);
        receive.await??;
        let mapped = self.readback.slice(..).get_mapped_range();
        let count = (self.camera.width as usize) * (self.camera.height as usize);
        let mut pixels = Vec::with_capacity(count);
        let mut depth = Vec::with_capacity(count);
        for sample in mapped.as_chunks::<8>().0 {
            pixels.push(sample[0]);
            depth.push(f32::from_le_bytes([
                sample[4], sample[5], sample[6], sample[7],
            ]));
        }
        drop(mapped);
        self.readback.unmap();
        if depth
            .iter()
            .any(|d| !d.is_finite() || *d < 0.0 || *d > self.limits.far_m)
        {
            return Err(RenderError::Invalid("GPU returned invalid optical depth"));
        }
        let image = GrayImage::from_raw(self.camera.width, self.camera.height, pixels)
            .ok_or(RenderError::Invalid("GPU output dimensions"))?;
        Ok((image, depth))
    }
}
