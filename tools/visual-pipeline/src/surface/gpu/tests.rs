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

    let frame = LocalFrame::wgs84_enu(surface.origin_lat, surface.origin_lon).expect("wgs84 frame");
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

#[test]
fn test_gpu_queue_role_isolation() {
    assert!(GpuQueueRole::ForegroundTracking.is_isolated());
    assert!(GpuQueueRole::BackgroundTileBuilder.is_isolated());
    assert!(GpuQueueRole::BackgroundAnchor.is_isolated());
    assert!(!GpuQueueRole::Shared.is_isolated());

    assert_eq!(
        GpuQueueRole::ForegroundTracking.as_str(),
        "foreground-tracking"
    );
    assert_eq!(
        GpuQueueRole::BackgroundTileBuilder.as_str(),
        "background-tile-builder"
    );
    assert_eq!(GpuQueueRole::BackgroundAnchor.as_str(), "background-anchor");
    assert_eq!(GpuQueueRole::Shared.as_str(), "shared");
}

#[test]
fn test_gpu_renderer_queue_roles_and_identities() {
    let tile = RgbImage::from_pixel(256, 256, Rgb([140, 150, 130]));
    let surface = Surface::synthetic(39.75, -75.55, 80.0, tile);
    let size = [160_u32, 120_u32];
    let k = Matrix3::new(140.0, 0.0, 79.5, 0.0, 140.0, 59.5, 0.0, 0.0, 1.0);

    let roles = [
        (
            GpuQueueRole::ForegroundTracking,
            true,
            "foreground-tracking",
        ),
        (
            GpuQueueRole::BackgroundTileBuilder,
            true,
            "background-tile-builder",
        ),
        (GpuQueueRole::BackgroundAnchor, true, "background-anchor"),
        (GpuQueueRole::Shared, false, "shared"),
    ];

    for (role, expected_isolated, expected_prefix) in roles {
        if let Some(renderer) = SurfaceGpuRenderer::try_new_with_role(&surface, &k, size, role) {
            assert_eq!(renderer.role(), role);
            assert_eq!(renderer.is_queue_isolated(), expected_isolated);
            let ident = renderer.queue_identity();
            assert!(
                ident.starts_with(expected_prefix),
                "expected prefix '{expected_prefix}', got '{ident}'"
            );
            assert!(
                ident.contains(renderer.backend_identity()),
                "identity must include backend identity"
            );
            let _device = renderer.device();
            let _queue = renderer.queue();
        }
    }
}

#[test]
fn large_z18_catalog_uses_bounded_subatlas_without_overflowing_16k_texture_limit() {
    let origin_lat = 39.75_f64;
    let origin_lon = -75.55_f64;
    let z_img = 18_u32;
    let z_dem = 14_u32;
    let (cx, cy, _, _) = mercator_tile_coords(origin_lat, origin_lon, z_img);
    let mut surface = Surface::synthetic(
        origin_lat,
        origin_lon,
        55.0,
        RgbImage::from_pixel(16, 16, Rgb([120, 140, 110])),
    );
    surface.catalog_sha256 = "f".repeat(64);
    surface.imagery_zoom = z_img;
    surface.dem_zoom = z_dem;
    surface.imagery_tiles.clear();
    surface.dem_tiles.clear();
    surface.dem_tile_size = 16;

    let small_rgb = RgbImage::from_pixel(16, 16, Rgb([135, 145, 125]));
    let small_dem = vec![55.0_f32; 16 * 16];
    let min_tx = cx - 40;
    let max_tx = cx + 40;
    let min_ty = cy - 40;
    let max_ty = cy + 40;
    // Populate corners and center across an 81 x 81 z=18 span (20,736 px > 16,384 px).
    for &(tx, ty) in &[
        (min_tx, min_ty),
        (max_tx, max_ty),
        (cx, cy),
        (cx + 1, cy),
        (max_tx - 1, max_ty - 1),
    ] {
        surface.imagery_tiles.insert((tx, ty), small_rgb.clone());
        surface
            .dem_tiles
            .insert((tx >> 4, ty >> 4), small_dem.clone());
    }

    assert!(
        build_image_atlas(&surface).is_none(),
        "full 81x81 tile span must be rejected before allocating >16k texture"
    );

    let mut template =
        SceneTemplate::from_surface(&surface).expect("sub-atlas SceneTemplate must succeed");
    let first_win = template
        .subatlas_window
        .expect("81x81 span must activate sub-atlas windowing");
    assert!(template.imagery_layout.size[0] <= 60 * 256);
    assert!(template.imagery_layout.size[1] <= 60 * 256);

    // Shift focus near `(max_tx, max_ty)` and confirm sub-atlas re-centers cleanly.
    let n = f64::from(1_u32 << z_img);
    let corner_lon = ((f64::from(max_tx) + 0.5) / n) * 360.0 - 180.0;
    let merc_y = std::f64::consts::PI * (1.0 - 2.0 * (f64::from(max_ty) + 0.5) / n);
    let corner_lat = merc_y.sinh().atan().to_degrees();
    assert!(template.update_subatlas_at(&surface, corner_lat, corner_lon, MAX_ATLAS_DIM_PX));
    let next_win = template.subatlas_window.expect("sub-atlas window");
    assert_ne!(first_win, next_win);
    assert!(next_win[1] >= max_tx && next_win[3] >= max_ty);
    assert!(template.imagery_layout.size[0] <= 60 * 256);
    assert!(template.imagery_layout.size[1] <= 60 * 256);

    let k = Matrix3::new(140.0, 0.0, 79.5, 0.0, 140.0, 59.5, 0.0, 0.0, 1.0);
    if let Some(mut gpu) = SurfaceGpuRenderer::try_new(&surface, &k, [160, 120]) {
        let eye_center = surface.local(origin_lat, origin_lon, 300.0);
        let rot_center = surface.camera_rotation(origin_lat, origin_lon, 0.0, 90.0, 0.0);
        let v0 = gpu
            .render_view(&surface, eye_center, &rot_center, &k, [160, 120])
            .expect("GPU render at center sub-atlas");
        assert_eq!(v0.image.dimensions(), (160, 120));

        let eye_corner = surface.local(corner_lat, corner_lon, 300.0);
        let rot_corner = surface.camera_rotation(corner_lat, corner_lon, 0.0, 90.0, 0.0);
        let v1 = gpu
            .render_view(&surface, eye_corner, &rot_corner, &k, [160, 120])
            .expect("GPU render after shifting sub-atlas to far corner");
        assert_eq!(v1.image.dimensions(), (160, 120));
    }
}
