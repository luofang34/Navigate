use super::*;

#[test]
fn expansion_shader_requires_no_optional_capabilities() {
    let module = wgpu::naga::front::wgsl::parse_str(include_str!("../upload.wgsl"))
        .expect("valid gray expansion shader");
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("core compute capabilities");
}

#[test]
#[ignore = "requires a hardware compute adapter; run explicitly on the target device"]
fn expansion_preserves_all_intensities_and_unaligned_levels() {
    let (compute, _) = pollster::block_on(Compute::new()).expect("GPU");
    for (width, height) in [(257, 3), (319, 239), (321, 241), (127, 95)] {
        let image = GrayImage::from_fn(width, height, |x, y| {
            image::Luma([((y * width + x) % 256) as u8])
        });
        let levels = crate::matching::pyramid::build(&image);
        let expected: Vec<_> = levels
            .iter()
            .flat_map(|image| image.as_raw().iter().map(|value| f32::from(*value)))
            .collect();
        let mut encoder = compute.device.create_command_encoder(&Default::default());
        let expanded = compute.image_buffer(&levels, false, &mut Profile::new(), &mut encoder);
        let readback = compute.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("expanded gray test readback"),
            size: expanded.size(),
            mapped_at_creation: false,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        });
        encoder.copy_buffer_to_buffer(&expanded, 0, &readback, 0, expanded.size());
        compute.queue.submit([encoder.finish()]);
        let (send, receive) = std::sync::mpsc::sync_channel(1);
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                send.send(result).ok();
            });
        compute.device.poll(wgpu::Maintain::Wait);
        receive.recv().expect("map event").expect("readback mapped");
        let mapped = readback.slice(..).get_mapped_range();
        let actual: Vec<_> = mapped
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect();
        assert_eq!(actual, expected);
        drop(mapped);
        readback.unmap();
    }
}
