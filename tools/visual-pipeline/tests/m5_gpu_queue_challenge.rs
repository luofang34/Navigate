//! Empirical challenge tests for GPU queue isolation and device handles (Milestone 5, Feature 24).
//!
//! This suite follows ASD-STE100 Simplified Technical English.

use image::{Rgb, RgbImage};
use nalgebra::Matrix3;
use navigate_visual_pipeline::surface::{Surface, SurfaceGpuRenderer};

#[test]
fn challenge_gpu_renderer_handles() {
    let tile = RgbImage::from_pixel(256, 256, Rgb([140, 150, 130]));
    let surface = Surface::synthetic(39.75, -75.55, 80.0, tile);
    let size = [160_u32, 120_u32];
    let k = Matrix3::new(140.0, 0.0, 79.5, 0.0, 140.0, 59.5, 0.0, 0.0, 1.0);

    if let Some(renderer) = SurfaceGpuRenderer::try_new(&surface, &k, size) {
        assert!(
            !renderer.is_queue_isolated(),
            "Shared role must not be isolated"
        );
        let ident = renderer.queue_identity();
        assert!(
            ident.starts_with("shared:"),
            "Identity must start with 'shared:', got {ident}"
        );
        let role = renderer.role();
        assert!(
            !role.is_isolated(),
            "Role is_isolated() must match renderer.is_queue_isolated()"
        );
        assert_eq!(role.as_str(), "shared", "Role as_str() must be 'shared'");
        let _dev = renderer.device();
        let _q = renderer.queue();
    }
}
