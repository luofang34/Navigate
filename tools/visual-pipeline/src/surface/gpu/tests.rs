use super::*;
use image::{Rgb, RgbImage};

#[test]
fn gpu_depth_unprojection_matches_cpu_surface_raycast() {
    let tile = RgbImage::from_pixel(256, 256, Rgb([140, 150, 130]));
    let surface = Surface::synthetic(39.75, -75.55, 80.0, tile);
    let size = [160_u32, 120_u32];
    let k = Matrix3::new(140.0, 0.0, 79.5, 0.0, 140.0, 59.5, 0.0, 0.0, 1.0);
    let eye = surface.local(39.75, -75.55, 680.0);
    let rot = surface.camera_rotation(39.75, -75.55, 25.0, 55.0, 0.0);
    let (_, cpu_depth, _) = surface.render(eye, &rot, &k, size);

    let frame =
        LocalFrame::spherical_enu(surface.origin_lat, surface.origin_lon).expect("spherical frame");
    let pose = surface_pose_to_gpu(&surface, frame, eye, &rot).expect("gpu pose");
    let camera = camera_model(&k, size).expect("camera");
    let gpu_view = GpuSurfaceView {
        image: GrayImage::new(size[0], size[1]),
        depth_m: cpu_depth,
        camera,
        pose,
        frame,
    };

    let sample_px = [[80.0, 60.0], [45.25, 72.5], [112.75, 48.25]];
    let unproj = gpu_view.unproject_pixels(&surface, &sample_px);
    let cpu_hits = surface.raycast_pixels(eye, &rot, &k, &sample_px);
    for (u_opt, c_opt) in unproj.iter().zip(cpu_hits.iter()) {
        let u = u_opt.expect("gpu hit");
        let c = c_opt.expect("cpu hit").world_enu;
        let err_m = ((u[0] - c[0]).powi(2) + (u[1] - c[1]).powi(2) + (u[2] - c[2]).powi(2)).sqrt();
        assert!(err_m < 3.5, "expected < 3.5 m ENU delta, got {err_m:.3} m");
    }
}
