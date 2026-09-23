#![allow(clippy::expect_used)]

use super::*;
use crate::package::{DecodedTile, MapPackage};
use image::{Rgba, RgbaImage};
use nalgebra::{UnitQuaternion, Vector2, Vector3};

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

#[tokio::test]
#[ignore = "requires a GPU adapter"]
async fn terrain_height_and_missing_imagery_control_depth() {
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
}

#[tokio::test]
#[ignore = "requires a GPU adapter"]
async fn loaded_dem_cannot_validate_a_coarser_fallback_surface() {
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
}

#[tokio::test]
#[ignore = "requires a GPU adapter"]
async fn the_bench_renderer_serves_the_navigate_reference_port() {
    use navigate_visual::ReferenceRenderer as _;
    let camera = CameraModel {
        width: 64,
        height: 64,
        fx: 200.0,
        fy: 200.0,
        cx: 31.5,
        cy: 31.5,
    };
    let mut renderer = ReferenceRenderer::new(nonzero_terrain(), camera)
        .await
        .expect("renderer");
    let identity = navigate_visual::ReferenceRenderer::identity(&renderer);
    assert_eq!(identity.revision.len(), 40);
    assert_eq!(identity.style_sha256.len(), 64);
    let poses = [100.0, 120.0].map(|z| CameraPose {
        position: Vector3::new(0.0, 0.0, z),
        orientation: UnitQuaternion::identity(),
    });
    let views = renderer.render_batch_blocking(&poses).expect("batch");
    assert_eq!(views.len(), 2);
    assert!(views.iter().all(|v| v.depth_m.iter().any(|d| *d > 0.0)));
}
