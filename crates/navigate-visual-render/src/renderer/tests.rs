use super::*;
use nalgebra::{UnitQuaternion, Vector3};

fn camera() -> CameraModel {
    CameraModel {
        width: 48,
        height: 40,
        fx: 50.0,
        fy: 50.0,
        cx: 23.5,
        cy: 19.5,
    }
}

fn limits() -> RenderLimits {
    RenderLimits {
        scene_bytes: 3_000_000,
        pixels: 2048,
        far_m: 5000.0,
        steps: 100,
        refinements: 16,
    }
}

fn scene(frame: LocalFrame) -> RenderScene {
    let layout = AtlasLayout {
        zoom: 14,
        origin_tile: [8191, 8191],
        tile_pixels: 256,
        size: [512, 512],
    };
    RenderScene {
        map: MapRevision {
            release_id: "synthetic-plane".into(),
            manifest_sha256: "a".repeat(64),
        },
        frame,
        imagery: ImageAtlas {
            layout,
            pixels: vec![173; 512 * 512],
            coverage: vec![255; 512 * 512],
        },
        terrain: TerrainAtlas {
            layout,
            heights_m: vec![0.0; 512 * 512],
            coverage: vec![255; 512 * 512],
        },
    }
}

#[test]
fn resource_and_coverage_limits_are_checked_before_device_creation() {
    let mut scene = scene(LocalFrame::anchor_mercator(0.0, 0.0).unwrap());
    assert_eq!(validate(&scene, camera(), limits()).unwrap(), 1_572_864);
    assert!(
        validate(
            &scene,
            camera(),
            RenderLimits {
                scene_bytes: 1,
                ..limits()
            }
        )
        .is_err()
    );
    scene.terrain.heights_m[42] = f32::NAN;
    assert!(validate(&scene, camera(), limits()).is_err());
    scene.terrain.coverage[42] = 0;
    assert!(validate(&scene, camera(), limits()).is_ok());
    scene.imagery.layout.origin_tile[0] = 16_383;
    assert!(validate(&scene, camera(), limits()).is_err());
}

#[test]
fn oversized_views_and_unbounded_terrain_are_rejected_before_allocation() {
    let mut scene = scene(LocalFrame::anchor_mercator(0.0, 0.0).unwrap());
    let oversized = CameraModel {
        width: u32::MAX,
        height: u32::MAX,
        ..camera()
    };
    assert!(
        validate(
            &scene,
            oversized,
            RenderLimits {
                pixels: usize::MAX,
                ..limits()
            }
        )
        .is_err()
    );
    scene.terrain.heights_m[0] = 100_001.0;
    assert!(validate(&scene, camera(), limits()).is_err());
    scene.terrain.coverage[0] = 0;
    assert!(validate(&scene, camera(), limits()).is_ok());
}

#[test]
#[ignore = "requires a hardware GPU"]
fn hardware_packed_coverage_preserves_valid_edges_and_refuses_buried_entries() {
    let frame = LocalFrame::anchor_mercator(0.0, 0.0).unwrap();
    let mut half = scene(frame);
    for index in 0..half.terrain.heights_m.len() {
        if index % 512 >= 256 {
            half.terrain.coverage[index] = 0;
            half.terrain.heights_m[index] = f32::NAN;
        }
    }
    let pose = CameraPose {
        position: Vector3::new(0.0, 0.0, 1000.0),
        orientation: UnitQuaternion::identity(),
    };
    let mut renderer = pollster::block_on(GpuRenderer::new(half, camera(), limits())).unwrap();
    let view = renderer.render_blocking(pose).unwrap();
    for (index, depth) in view.depth_m.iter().enumerate() {
        let expected = if index % 48 < 24 { 1000.0 } else { 0.0 };
        assert!((*depth - expected).abs() < 0.003);
    }
    let mut buried = scene(frame);
    buried.terrain.heights_m.fill(1500.0);
    for (index, coverage) in buried.terrain.coverage.iter_mut().enumerate() {
        *coverage = if index % 512 < 256 { 0 } else { 255 };
    }
    let mut renderer = pollster::block_on(GpuRenderer::new(buried, camera(), limits())).unwrap();
    let view = renderer
        .render_blocking(CameraPose {
            position: Vector3::new(-500.0, 0.0, 1000.0),
            ..pose
        })
        .unwrap();
    assert!(view.depth_m.iter().all(|d| *d == 0.0));
}

#[test]
#[ignore = "requires a hardware GPU"]
fn hardware_plane_render_reuses_the_scene_and_preserves_optical_depth() {
    let mut renderer = pollster::block_on(GpuRenderer::new(
        scene(LocalFrame::anchor_mercator(0.0, 0.0).unwrap()),
        camera(),
        limits(),
    ))
    .unwrap();
    let pose = CameraPose {
        position: Vector3::new(0.0, 0.0, 1000.0),
        orientation: UnitQuaternion::identity(),
    };
    let first = renderer.render_blocking(pose).unwrap();
    assert!(first.image.as_raw().iter().all(|p| *p == 173));
    assert!(first.depth_m.iter().all(|d| (*d - 1000.0).abs() < 0.003));
    let next = renderer
        .render_blocking(CameraPose {
            position: Vector3::new(100.0, 150.0, 800.0),
            ..pose
        })
        .unwrap();
    assert!(next.depth_m.iter().all(|d| (*d - 800.0).abs() < 0.003));
    assert_eq!(renderer.resident_texture_bytes(), 1_572_864);
    assert_eq!(renderer.identity().style_sha256.len(), 64);
}

#[test]
#[ignore = "requires a hardware GPU"]
fn hardware_sphere_depth_agrees_with_an_analytic_ray_intersection() {
    let camera = camera();
    let mut renderer = pollster::block_on(GpuRenderer::new(
        scene(LocalFrame::spherical_enu(0.0, 0.0).unwrap()),
        camera,
        limits(),
    ))
    .unwrap();
    let pose = CameraPose {
        position: Vector3::new(700.0, 300.0, 1000.0),
        orientation: UnitQuaternion::identity(),
    };
    let view = renderer.render_blocking(pose).unwrap();
    let r = navigate_visual::MERCATOR_SPHERE_RADIUS_M;
    let origin = pose.position + Vector3::new(0.0, 0.0, r);
    for y in 0..camera.height {
        for x in 0..camera.width {
            let ray = Vector3::new(
                (f64::from(x) - camera.cx) / camera.fx,
                (camera.cy - f64::from(y)) / camera.fy,
                -1.0,
            );
            let b = origin.dot(&ray);
            let c = origin.norm_squared() - r * r;
            let expected = c / (-b + (b * b - ray.norm_squared() * c).sqrt());
            let actual = view.depth_m[(y * camera.width + x) as usize];
            assert!(
                (f64::from(actual) - expected).abs() < 0.005,
                "{actual} != {expected}"
            );
        }
    }
}

#[test]
#[ignore = "requires a hardware GPU"]
fn hardware_missing_coverage_cannot_create_a_surface() {
    let mut scene = scene(LocalFrame::anchor_mercator(0.0, 0.0).unwrap());
    scene.terrain.coverage.fill(0);
    let mut renderer = pollster::block_on(GpuRenderer::new(scene, camera(), limits())).unwrap();
    let pose = CameraPose {
        position: Vector3::new(0.0, 0.0, 1000.0),
        orientation: UnitQuaternion::identity(),
    };
    let view = renderer.render_blocking(pose).unwrap();
    assert!(view.depth_m.iter().all(|d| *d == 0.0));
    assert!(view.image.as_raw().iter().all(|p| *p == 0));
}

#[test]
#[ignore = "requires a hardware GPU"]
fn hardware_small_geographic_offsets_preserve_image_sample_positions() {
    let frame = LocalFrame::spherical_enu(41.075, -75.275).unwrap();
    let pose = CameraPose {
        position: frame.local([41.115, -75.275, 1000.0]).unwrap(),
        orientation: UnitQuaternion::identity(),
    };
    let xy = frame.mercator_xy(pose.position);
    let grid = |zoom| AtlasLayout {
        zoom,
        origin_tile: [
            (xy[0] * f64::from(1_u32 << zoom)) as u32 - 1,
            (xy[1] * f64::from(1_u32 << zoom)) as u32 - 1,
        ],
        tile_pixels: 256,
        size: [512, 512],
    };
    let mut scene = scene(frame);
    scene.imagery.layout = grid(18);
    scene.terrain.layout = grid(14);
    for (index, pixel) in scene.imagery.pixels.iter_mut().enumerate() {
        *pixel = ((index % 512 * 31 + index / 512 * 17) % 256) as u8;
    }
    let pixels = scene.imagery.pixels.clone();
    let layout = scene.imagery.layout;
    let camera = CameraModel {
        fx: 5000.0,
        fy: 5000.0,
        ..camera()
    };
    let mut renderer = pollster::block_on(GpuRenderer::new(scene, camera, limits())).unwrap();
    let view = renderer.render_blocking(pose).unwrap();
    for (index, actual) in view.image.as_raw().iter().enumerate() {
        let pixel = nalgebra::Vector2::new(
            (index % camera.width as usize) as f64,
            (index / camera.width as usize) as f64,
        );
        let point = camera.unproject(&pose, pixel, f64::from(view.depth_m[index]));
        let xy = frame.mercator_xy(point);
        let x = (xy[0] * f64::from(1_u32 << layout.zoom) - f64::from(layout.origin_tile[0]))
            * 256.0
            - 0.5;
        let y = (xy[1] * f64::from(1_u32 << layout.zoom) - f64::from(layout.origin_tile[1]))
            * 256.0
            - 0.5;
        let sample = |column: usize, row: usize| f64::from(pixels[row * 512 + column]);
        let a = x.floor() as usize;
        let b = y.floor() as usize;
        let top = sample(a, b) * (1.0 - x.fract()) + sample(a + 1, b) * x.fract();
        let bottom = sample(a, b + 1) * (1.0 - x.fract()) + sample(a + 1, b + 1) * x.fract();
        let expected = (top * (1.0 - y.fract()) + bottom * y.fract()).round();
        assert!(
            (f64::from(*actual) - expected).abs() <= 1.0,
            "GPU {actual}, f64 {expected}"
        );
    }
}

#[test]
fn wgs84_ellipsoid_frames_pass_validation_and_relative_f32_solver_matches_geodesy() {
    let frame = LocalFrame::wgs84_enu(47.0, 8.0).unwrap();
    let scene = scene(frame);
    assert!(validate(&scene, camera(), limits()).is_ok());

    let (s, c) = 47.0_f64.to_radians().sin_cos();
    let a = navigate_geodesy::wgs84::SEMI_MAJOR_AXIS_M;
    let e2 = navigate_geodesy::wgs84::FIRST_ECCENTRICITY_SQUARED;
    let d0_sqrt = (1.0 - e2 * s * s).sqrt();
    let n0 = (a / d0_sqrt) as f32;
    let s0 = s as f32;
    let c0 = c as f32;
    let e2_f = e2 as f32;
    let d0_f = d0_sqrt as f32;
    let a_1me2 = (a * (1.0 - e2)) as f32;

    for [east, north, up] in [
        [0.0_f32, 0.0, 1000.0],
        [4500.0, -3200.0, 1200.0],
        [-25_000.0, 30_000.0, 2500.0],
    ] {
        let ax = n0 * c0 - s0 * north + c0 * up;
        let h_cyl = (ax * ax + east * east).sqrt();
        let dlon = east.atan2(ax);
        let dp = -s0 * north + c0 * up + (east * east) / (h_cyl + ax);
        let dz = c0 * north + s0 * up;
        let mut dlat = 0.0_f32;
        let mut altitude = 0.0_f32;
        for _ in 0..3 {
            let sigma = dlat.sin();
            let half = (0.5 * dlat).sin();
            let gamma = 2.0 * half * half;
            let delta_s = c0 * sigma - s0 * gamma;
            let delta_c = -s0 * sigma - c0 * gamma;
            let s_lat = s0 + delta_s;
            let c_lat = c0 + delta_c;
            let delta_s2 = delta_s * (2.0 * s0 + delta_s);
            let d_val = (d0_f * d0_f - e2_f * delta_s2).max(1.0e-6);
            let d_sqrt = d_val.sqrt();
            let delta_n = n0 * (e2_f * delta_s2) / (d_sqrt * (d0_f + d_sqrt));
            let qp = dp - (delta_n * c_lat + n0 * delta_c);
            let qz = dz - (1.0 - e2_f) * (delta_n * s_lat + n0 * delta_s);
            altitude = qp * c_lat + qz * s_lat;
            let m_curv = a_1me2 / (d_val * d_sqrt);
            dlat += (-qp * s_lat + qz * c_lat) / (m_curv + altitude);
        }
        let exact = frame.geodetic(Vector3::new(
            f64::from(east),
            f64::from(north),
            f64::from(up),
        ));
        let exact_dlat = (exact[0] - 47.0).to_radians();
        let exact_dlon = (exact[1] - 8.0).to_radians();
        assert!((f64::from(dlat) - exact_dlat).abs() * a < 0.05);
        assert!((f64::from(dlon) - exact_dlon).abs() * a * c < 0.05);
        assert!((f64::from(altitude) - exact[2]).abs() < 0.05);
    }
}

#[test]
#[ignore = "requires a hardware GPU"]
fn hardware_wgs84_ellipsoid_depth_agrees_with_analytic_ray_intersection() {
    let camera = camera();
    let frame = LocalFrame::wgs84_enu(47.0, 8.0).unwrap();
    let xy = frame.mercator_xy(Vector3::zeros());
    let mut test_scene = scene(frame);
    let centered = AtlasLayout {
        zoom: 14,
        origin_tile: [
            (xy[0] * f64::from(1_u32 << 14)) as u32 - 1,
            (xy[1] * f64::from(1_u32 << 14)) as u32 - 1,
        ],
        tile_pixels: 256,
        size: [512, 512],
    };
    test_scene.imagery.layout = centered;
    test_scene.terrain.layout = centered;
    let mut renderer = pollster::block_on(GpuRenderer::new(test_scene, camera, limits())).unwrap();
    let pose = CameraPose {
        position: Vector3::new(700.0, 300.0, 1000.0),
        orientation: UnitQuaternion::identity(),
    };
    let view = renderer.render_blocking(pose).unwrap();
    for y in 0..camera.height {
        for x in 0..camera.width {
            let ray = Vector3::new(
                (f64::from(x) - camera.cx) / camera.fx,
                (camera.cy - f64::from(y)) / camera.fy,
                -1.0,
            );
            let actual = f64::from(view.depth_m[(y * camera.width + x) as usize]);
            assert!(actual > 900.0, "expected terrain hit at ({x}, {y})");
            let hit = pose.position + ray * actual;
            let [_, _, hit_alt_m] = frame.geodetic(hit);
            assert!(
                hit_alt_m.abs() < 0.02,
                "ellipsoid surface hit altitude {hit_alt_m} m at ({x}, {y})"
            );
        }
    }
}
