#![allow(clippy::expect_used)]

use super::*;
use crate::PyramidalMatcher;

#[test]
fn compute_shader_validates_without_hardware() {
    let module = wgpu::naga::front::wgsl::parse_str(include_str!("tracking.wgsl"))
        .expect("valid matching shader syntax");
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .expect("matching shader needs no optional capabilities");
}

#[test]
#[ignore = "requires a hardware compute adapter; run explicitly on the target device"]
fn gpu_matches_translation_and_rejects_blank_frames() {
    let reference = GrayImage::from_fn(320, 240, |x, y| {
        let h = (x / 7).wrapping_mul(374761393) ^ (y / 9).wrapping_mul(668265263);
        image::Luma([40 + ((h ^ (h >> 13)).wrapping_mul(1274126177) % 160) as u8])
    });
    let query = GrayImage::from_fn(320, 240, |x, y| {
        if x >= 26 && y >= 11 {
            image::Luma([reference.get_pixel(x - 26, y - 11)[0] + 15])
        } else {
            image::Luma([0])
        }
    });
    let mut gpu = pollster::block_on(GpuPyramidalMatcher::new()).expect("hardware compute adapter");
    let matches = gpu
        .match_images_blocking(&reference, &query)
        .expect("GPU matching");
    let cpu = PyramidalMatcher
        .match_images_blocking(&reference, &query)
        .expect("CPU matching");
    let correct = matches
        .iter()
        .filter(|m| (m.query - m.reference - Vector2::new(26.0, 11.0)).norm() < 0.5)
        .count();
    assert!(
        correct > 40,
        "{correct} correct of {} GPU matches",
        matches.len()
    );
    assert!(correct * 10 >= matches.len() * 9);
    assert!(matches.len().abs_diff(cpu.len()) < 10);
    assert!(
        gpu.match_images_blocking(&reference, &GrayImage::new(320, 240))
            .expect("blank query")
            .is_empty()
    );
    assert!(gpu.identity().starts_with("wgpu-pyramidal-lk-v1/"));
}
