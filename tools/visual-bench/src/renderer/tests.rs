use super::*;
use crate::package::{DecodedTile, MapPackage};
use image::{Rgba, RgbaImage};
use nalgebra::{UnitQuaternion, Vector2, Vector3};

fn block_on<F: std::future::Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime")
        .block_on(future)
}

fn nonzero_terrain() -> MapPackage {
    let n = 65536.0;
    let lon = (32768.5 / n) * 360.0 - 180.0;
    let lat = (std::f64::consts::PI * (1.0 - 2.0 * 32768.5 / n))
        .sinh()
        .atan()
        .to_degrees();
    let manifest = serde_json::from_value(serde_json::json!({
        "schema_version": 1, "region_id": "test", "release_id": "nonzero-alpha-test",
        "anchor_lat_lon": [lat, lon], "elevation_datum": "test", "attribution": "test fixture",
        "files": [],
        "tiles": [
            {"xyz": [16,32768,32768], "imagery": {"chunk": "c".repeat(64), "offset": 0, "length": 1, "sha256": "a".repeat(64)}},
            {"xyz": [14,8192,8192], "elevation": {"chunk": "c".repeat(64), "offset": 1, "length": 1, "sha256": "b".repeat(64)}}
        ]
    }))
    .expect("test manifest");
    MapPackage {
        manifest,
        revision: MapRevision {
            release_id: "nonzero-alpha-test".into(),
            manifest_sha256: "c".repeat(64),
        },
        frame: navigate_visual::LocalFrame::anchor_mercator(lat, lon).expect("valid anchor"),
        tiles: vec![
            DecodedTile {
                xyz: [16, 32768, 32768],
                imagery: Some(RgbaImage::from_fn(512, 512, |x, _| {
                    Rgba([100, 150, 200, if x < 256 { 0 } else { 255 }])
                })),
                elevation: None,
            },
            DecodedTile {
                xyz: [14, 8192, 8192],
                imagery: None,
                elevation: Some(RgbaImage::from_pixel(256, 256, Rgba([128, 17, 0, 255]))),
            },
        ],
    }
}

#[test]
#[ignore = "requires a GPU adapter"]
fn terrain_height_and_missing_imagery_control_depth() {
    block_on(async {
        let camera = CameraModel {
            width: 64,
            height: 64,
            fx: 200.0,
            fy: 200.0,
            cx: 31.5,
            cy: 31.5,
        };
        let pose = CameraPose {
            position: Vector3::new(0.0, 0.0, 100.0),
            orientation: UnitQuaternion::identity(),
        };
        let mut renderer = ReferenceRenderer::new(nonzero_terrain(), camera)
            .await
            .expect("renderer");
        let reference = renderer.render_blocking(pose).expect("render real depth");
        let right = reference.depth_m[32 * 64 + 48];
        assert!(right > 0.0, "opaque imagery has usable depth");
        let world = camera.unproject(&pose, Vector2::new(48.0, 32.0), f64::from(right));
        assert!(
            (world.z - 17.0).abs() < 0.02,
            "DEM height must reach the mesh: {world:?}"
        );
        assert_eq!(
            reference.depth_m[32 * 64 + 16],
            0.0,
            "missing imagery cannot supply a correspondence"
        );
    });
}

#[test]
#[ignore = "requires a GPU adapter"]
fn loaded_dem_cannot_validate_a_coarser_fallback_surface() {
    block_on(async {
        let mut package = nonzero_terrain();
        package.manifest.tiles[1].xyz = navigate_imagery::Tile(22, 2097184, 2097184);
        package.tiles[1].xyz = [22, 2097184, 2097184];
        let camera = CameraModel {
            width: 64,
            height: 64,
            fx: 200.0,
            fy: 200.0,
            cx: 31.5,
            cy: 31.5,
        };
        let pose = CameraPose {
            position: Vector3::new(0.0, 0.0, 100.0),
            orientation: UnitQuaternion::identity(),
        };
        let mut renderer = ReferenceRenderer::new(package, camera)
            .await
            .expect("renderer");
        let reference = renderer.render_blocking(pose).expect("fallback reference");
        assert!(reference.depth_m.iter().all(|depth| *depth == 0.0));
    });
}

#[test]
#[ignore = "requires a GPU adapter"]
fn the_bench_renderer_serves_the_navigate_reference_port() -> Result<(), Box<dyn std::error::Error>>
{
    use navigate_visual::ReferenceRenderer as _;
    let camera = CameraModel {
        width: 64,
        height: 64,
        fx: 200.0,
        fy: 200.0,
        cx: 31.5,
        cy: 31.5,
    };
    let mut renderer = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?
        .block_on(ReferenceRenderer::new(nonzero_terrain(), camera))?;
    let identity = navigate_visual::ReferenceRenderer::identity(&renderer);
    assert_eq!(identity.revision.len(), 40);
    assert_eq!(identity.style_sha256.len(), 64);
    let poses = [100.0, 120.0].map(|z| CameraPose {
        position: Vector3::new(0.0, 0.0, z),
        orientation: UnitQuaternion::identity(),
    });
    let views = renderer.render_batch_blocking(&poses)?;
    assert_eq!(views.len(), 2);
    assert!(views.iter().all(|v| v.depth_m.iter().any(|d| *d > 0.0)));
    Ok(())
}

#[test]
#[ignore = "requires a GPU adapter"]
fn oblique_depth_unprojects_to_the_rendered_ground_plane() {
    block_on(async {
        let camera = CameraModel {
            width: 320,
            height: 180,
            fx: 230.0,
            fy: 235.0,
            cx: 145.5,
            cy: 83.5,
        };
        let mut package = nonzero_terrain();
        package.tiles[0].imagery = Some(RgbaImage::from_pixel(512, 512, Rgba([90, 150, 100, 255])));
        let mut renderer = ReferenceRenderer::new(package, camera)
            .await
            .expect("renderer");
        for orientation in [
            UnitQuaternion::identity(),
            UnitQuaternion::from_euler_angles(0.2, -0.1, 0.8),
            UnitQuaternion::from_euler_angles(0.45, 0.2, -1.4),
        ] {
            let pose = CameraPose {
                position: Vector3::new(0.0, 0.0, 100.0),
                orientation,
            };
            let view = renderer.render_blocking(pose).expect("reference render");
            let mut checked = 0_usize;
            for y in (20..160).step_by(30) {
                for x in (20..300).step_by(40) {
                    let depth = view.depth_m[y * camera.width as usize + x];
                    if depth <= 0.0 {
                        continue;
                    }
                    let world =
                        camera.unproject(&pose, Vector2::new(x as f64, y as f64), f64::from(depth));
                    assert!(
                        (world.z - 17.0).abs() < 0.1,
                        "optical depth and pose must describe the same surface: pixel=({x},{y}), pose={pose:?}, world={world:?}"
                    );
                    checked = checked.wrapping_add(1);
                }
            }
            assert!(checked >= 20, "test scene must cover the sampled image");
        }
    });
}
