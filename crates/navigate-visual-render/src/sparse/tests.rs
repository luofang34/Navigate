use super::*;
use nalgebra::{UnitQuaternion, Vector2};
use navigate_visual::MapRevision;

fn centered_layout(frame: LocalFrame, zoom: u32) -> AtlasLayout {
    let xy = frame.mercator_xy(Vector3::zeros());
    let tiles = f64::from(1_u32 << zoom);
    AtlasLayout {
        zoom,
        origin_tile: [(xy[0] * tiles) as u32 - 1, (xy[1] * tiles) as u32 - 1],
        tile_pixels: 256,
        size: [512, 512],
    }
}

fn make_scene(frame: LocalFrame, height_m: f32) -> RenderScene {
    let layout = centered_layout(frame, 13);
    let mut pixels = vec![0_u8; 512 * 512];
    for (idx, p) in pixels.iter_mut().enumerate() {
        let col = idx % 512;
        let row = idx / 512;
        *p = ((col * 29 + row * 13 + 37) % 256) as u8;
    }
    RenderScene {
        map: MapRevision {
            release_id: "sparse-test-release".into(),
            manifest_sha256: "b".repeat(64),
        },
        frame,
        imagery: ImageAtlas {
            layout,
            pixels,
            coverage: vec![255; 512 * 512],
        },
        terrain: TerrainAtlas {
            layout,
            heights_m: vec![height_m; 512 * 512],
            coverage: vec![255; 512 * 512],
        },
    }
}

fn analytic_bilinear_luma(imagery: &ImageAtlas, frame: LocalFrame, world: Vector3<f64>) -> f64 {
    let xy = frame.mercator_xy(world);
    let tiles = f64::from(1_u32 << imagery.layout.zoom);
    let tp = f64::from(imagery.layout.tile_pixels);
    let x = (xy[0] * tiles - f64::from(imagery.layout.origin_tile[0])) * tp - 0.5;
    let y = (xy[1] * tiles - f64::from(imagery.layout.origin_tile[1])) * tp - 0.5;
    let w = imagery.layout.size[0] as usize;
    let a = x.floor() as usize;
    let b = y.floor() as usize;
    let fx = x.fract();
    let fy = y.fract();
    let at = |c: usize, r: usize| f64::from(imagery.pixels[r * w + c]);
    let top = at(a, b) * (1.0 - fx) + at(a + 1, b) * fx;
    let bot = at(a, b + 1) * (1.0 - fx) + at(a + 1, b + 1) * fx;
    (top * (1.0 - fy) + bot * fy).round()
}

#[test]
fn validation_checks_scene_parameters_and_coverage_edges() {
    let frame = LocalFrame::anchor_mercator(0.0, 0.0).unwrap();
    let mut scene = make_scene(frame, 0.0);
    assert!(SparseSceneSampler::new(&scene, 0.0, 64, 12).is_err());
    assert!(SparseSceneSampler::new(&scene, -20.0, 64, 12).is_err());
    assert!(SparseSceneSampler::new(&scene, f64::NAN, 64, 12).is_err());
    assert!(SparseSceneSampler::new(&scene, 5000.0, 0, 12).is_err());

    scene.terrain.heights_m[7] = f32::NAN;
    assert!(SparseSceneSampler::new(&scene, 5000.0, 64, 12).is_err());
    scene.terrain.coverage[7] = 0;
    let sampler = SparseSceneSampler::new(&scene, 5000.0, 100, 16).unwrap();

    let cam = CameraModel {
        width: 48,
        height: 40,
        fx: 50.0,
        fy: 50.0,
        cx: 23.5,
        cy: 19.5,
    };
    let pose = CameraPose {
        position: Vector3::new(0.0, 0.0, 1000.0),
        orientation: UnitQuaternion::identity(),
    };
    assert!(
        sampler
            .raycast_pixels(&cam, &pose, &[[f64::NAN, 10.0]])
            .is_err()
    );
    assert!(sampler.render_adaptive(&cam, &pose, 4, 8, -0.01).is_err());

    // Uncovered right half and buried side-entry refusal match GPU kernel rules.
    let mut half = make_scene(frame, 0.0);
    for idx in 0..half.terrain.heights_m.len() {
        if idx % 512 >= 256 {
            half.terrain.coverage[idx] = 0;
            half.terrain.heights_m[idx] = f32::NAN;
        }
    }
    let half_sampler = SparseSceneSampler::new(&half, 5000.0, 100, 16).unwrap();
    let hits = half_sampler
        .raycast_pixels(&cam, &pose, &[[10.0, 19.5], [36.0, 19.5]])
        .unwrap();
    assert!((hits[0].unwrap().depth_m - 1000.0).abs() < 0.005);
    assert!(hits[1].is_none());

    let mut buried = make_scene(frame, 1500.0);
    for (idx, cov) in buried.terrain.coverage.iter_mut().enumerate() {
        *cov = if idx % 512 < 256 { 0 } else { 255 };
    }
    let buried_sampler = SparseSceneSampler::new(&buried, 5000.0, 100, 16).unwrap();
    let buried_pose = CameraPose {
        position: Vector3::new(-500.0, 0.0, 1000.0),
        orientation: UnitQuaternion::identity(),
    };
    let (buried_view, _) = buried_sampler
        .render_adaptive(&cam, &buried_pose, 1, 0, 0.0)
        .unwrap();
    assert!(buried_view.depth_m.iter().all(|&d| d == 0.0));
}

#[test]
fn raycast_pixels_agrees_with_analytic_unproject_on_anchor_mercator() {
    let frame = LocalFrame::anchor_mercator(46.8, 8.2).unwrap();
    let mut scene = make_scene(frame, 180.0);
    // Add a smooth east-north planar slope to test non-flat terrain bisection.
    for row in 0..512 {
        for col in 0..512 {
            let h = 180.0 + 0.15 * (col as f32 - 256.0) - 0.1 * (row as f32 - 256.0);
            scene.terrain.heights_m[row * 512 + col] = h;
        }
    }
    let sampler = SparseSceneSampler::new(&scene, 8000.0, 128, 18).unwrap();
    let cam = CameraModel {
        width: 64,
        height: 48,
        fx: 72.0,
        fy: 72.0,
        cx: 31.5,
        cy: 23.5,
    };
    let pose = CameraPose {
        position: Vector3::new(120.0, -90.0, 1150.0),
        orientation: UnitQuaternion::from_scaled_axis(Vector3::new(0.12, -0.08, 0.25)),
    };
    let keypoints = [
        [8.25, 7.5],
        [19.75, 14.25],
        [31.5, 23.5],
        [44.125, 31.875],
        [55.625, 39.375],
    ];
    let hits = sampler.raycast_pixels(&cam, &pose, &keypoints).unwrap();
    for (&[u, v], maybe_hit) in keypoints.iter().zip(&hits) {
        let hit = maybe_hit.expect("expected terrain intersection");
        let world = Vector3::from(hit.world_local);
        let unprojected = cam.unproject(&pose, Vector2::new(u, v), f64::from(hit.depth_m));
        assert!(
            (world - unprojected).norm() < 0.005,
            "unproject mismatch: {world:?} vs {unprojected:?}"
        );
        let reprojected = cam.project(&pose, world).unwrap();
        assert!((reprojected.x - u).abs() < 1e-6);
        assert!((reprojected.y - v).abs() < 1e-6);

        let [_, _, alt_m] = frame.geodetic(world);
        let expected_h = sample_terrain(&scene.terrain, frame.mercator_xy(world)).unwrap();
        assert!(
            (alt_m - expected_h).abs() < 0.02,
            "terrain height mismatch: {alt_m} vs {expected_h}"
        );

        let expected_luma = analytic_bilinear_luma(&scene.imagery, frame, world);
        assert!(
            (f64::from(hit.luma) - expected_luma).abs() <= 1.0,
            "luma {} vs {expected_luma}",
            hit.luma
        );
    }
}

#[test]
fn raycast_pixels_agrees_with_analytic_geodesy_on_wgs84_enu() {
    let frame = LocalFrame::wgs84_enu(47.0, 8.0).unwrap();
    let elev_m = 325.0_f32;
    let scene = make_scene(frame, elev_m);
    let sampler = SparseSceneSampler::new(&scene, 9000.0, 128, 18).unwrap();
    let cam = CameraModel {
        width: 64,
        height: 48,
        fx: 68.0,
        fy: 68.0,
        cx: 31.5,
        cy: 23.5,
    };
    let pose = CameraPose {
        position: Vector3::new(650.0, -420.0, 1450.0),
        orientation: UnitQuaternion::from_scaled_axis(Vector3::new(-0.09, 0.11, -0.18)),
    };
    let keypoints = [
        [6.5, 8.25],
        [20.125, 15.75],
        [31.5, 23.5],
        [45.875, 30.25],
        [57.25, 41.5],
    ];
    let hits = sampler.raycast_pixels(&cam, &pose, &keypoints).unwrap();
    for (&[u, v], maybe_hit) in keypoints.iter().zip(&hits) {
        let hit = maybe_hit.expect("expected WGS84 terrain hit");
        let world = Vector3::from(hit.world_local);
        let unprojected = cam.unproject(&pose, Vector2::new(u, v), f64::from(hit.depth_m));
        assert!((world - unprojected).norm() < 0.005);

        let [lat, lon, alt_m] = frame.geodetic(world);
        assert!((lat - 47.0).abs() < 0.2 && (lon - 8.0).abs() < 0.2);
        assert!(
            (alt_m - f64::from(elev_m)).abs() < 0.02,
            "WGS84 ellipsoid altitude {alt_m} != {elev_m}"
        );
        let round_trip = frame.local([lat, lon, alt_m]).unwrap();
        assert!((round_trip - world).norm() < 1e-3);

        let expected_luma = analytic_bilinear_luma(&scene.imagery, frame, world);
        assert!((f64::from(hit.luma) - expected_luma).abs() <= 1.0);
    }
}

#[test]
fn render_adaptive_casts_under_quarter_rays_and_preserves_sharp_edges() {
    let frame = LocalFrame::wgs84_enu(46.95, 7.45).unwrap();
    let layout = centered_layout(frame, 14);
    let mut pixels = vec![0_u8; 512 * 512];
    let mut heights_m = vec![50.0_f32; 512 * 512];

    // Piecewise-smooth background with a sharp bright road/patch and a raised mesa.
    for row in 0..512 {
        for col in 0..512 {
            let idx = row * 512 + col;
            let smooth_bg = (48.0 + 0.12 * col as f64 + 0.10 * row as f64).round() as u8;
            let in_bright_patch = (232..268).contains(&col) && (228..264).contains(&row);
            let in_diagonal_band =
                (col as i32 - row as i32).abs() <= 6 && (210..295).contains(&col);
            pixels[idx] = if in_bright_patch {
                235
            } else if in_diagonal_band {
                205
            } else {
                smooth_bg
            };
            if (262..294).contains(&col) && (248..282).contains(&row) {
                heights_m[idx] = 190.0;
            }
        }
    }

    let scene = RenderScene {
        map: MapRevision {
            release_id: "adaptive-piecewise-smooth".into(),
            manifest_sha256: "c".repeat(64),
        },
        frame,
        imagery: ImageAtlas {
            layout,
            pixels,
            coverage: vec![255; 512 * 512],
        },
        terrain: TerrainAtlas {
            layout,
            heights_m,
            coverage: vec![255; 512 * 512],
        },
    };

    let sampler = SparseSceneSampler::new(&scene, 6000.0, 96, 16).unwrap();
    let cam = CameraModel {
        width: 128,
        height: 96,
        fx: 150.0,
        fy: 150.0,
        cx: 63.5,
        cy: 47.5,
    };
    let pose = CameraPose {
        position: Vector3::new(0.0, 0.0, 1200.0),
        orientation: UnitQuaternion::from_scaled_axis(Vector3::new(0.06, -0.04, 0.0)),
    };

    let (exact_view, exact_stats) = sampler.render_adaptive(&cam, &pose, 1, 0, 0.0).unwrap();
    assert_eq!(exact_stats.rays_cast, exact_stats.total_pixels);

    let (adaptive_view, adaptive_stats) =
        sampler.render_adaptive(&cam, &pose, 4, 4, 0.015).unwrap();
    adaptive_view.validate_camera(&cam).unwrap();

    assert_eq!(adaptive_stats.total_pixels, 128 * 96);
    assert!(
        adaptive_stats.rays_cast * 4 < adaptive_stats.total_pixels,
        "expected < 25% rays, cast {} of {}",
        adaptive_stats.rays_cast,
        adaptive_stats.total_pixels
    );

    let exact_raw = exact_view.image.as_raw();
    let adaptive_raw = adaptive_view.image.as_raw();
    let w = cam.width as usize;
    let h = cam.height as usize;
    let mut sharp_edge_pixels = 0_usize;

    for y in 1..(h - 1) {
        for x in 1..(w - 1) {
            let idx = y * w + x;
            let e = i16::from(exact_raw[idx]);
            let a = i16::from(adaptive_raw[idx]);
            let grad_x = (i16::from(exact_raw[idx + 1]) - i16::from(exact_raw[idx - 1])).abs();
            let grad_y = (i16::from(exact_raw[idx + w]) - i16::from(exact_raw[idx - w])).abs();
            if grad_x >= 40 || grad_y >= 40 {
                sharp_edge_pixels += 1;
                assert!(
                    (a - e).abs() <= 1,
                    "sharp edge pixel ({x}, {y}) differed: adaptive {a} vs exact {e}"
                );
            }
            assert!(
                (a - e).abs() <= 4,
                "pixel ({x}, {y}) differed: adaptive {a} vs exact {e}"
            );
        }
    }
    assert!(
        sharp_edge_pixels >= 80,
        "expected scene to exercise localized sharp edges, found only {sharp_edge_pixels}"
    );
}
