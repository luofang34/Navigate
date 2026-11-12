use navigate_visual::{LocalFrame, MapRevision};

use super::*;
use crate::{AtlasLayout, ImageAtlas, TerrainAtlas};

/// A scene of 1024 by 1024 imagery pixels at zoom 14 about its anchor,
/// with luma varying smoothly across it and a strip north to south
/// through the anchor that the package does not cover.
fn scene() -> RenderScene {
    let frame = LocalFrame::wgs84_enu(46.95, 7.45).unwrap();
    let xy = frame.mercator_xy(Vector3::zeros());
    let tiles = f64::from(1_u32 << 14);
    let layout = AtlasLayout {
        zoom: 14,
        origin_tile: [(xy[0] * tiles) as u32 - 2, (xy[1] * tiles) as u32 - 2],
        tile_pixels: 256,
        size: [1024, 1024],
    };
    // The strip is some 260 m wide about the anchor.
    let anchor_col = (xy[0] * tiles - f64::from(layout.origin_tile[0])) * 256.0;
    let mut pixels = vec![0_u8; 1024 * 1024];
    let mut coverage = vec![255_u8; 1024 * 1024];
    for (i, (p, c)) in pixels.iter_mut().zip(coverage.iter_mut()).enumerate() {
        let (col, row) = ((i % 1024) as f64, (i / 1024) as f64);
        *p =
            (128.0 + 60.0 * (col / 23.0).sin() + 50.0 * (row / 31.0).cos()).clamp(0.0, 255.0) as u8;
        if (anchor_col - 20.0..anchor_col + 20.0).contains(&col) {
            *c = 0;
        }
    }
    RenderScene {
        map: MapRevision {
            release_id: "map-raster".into(),
            manifest_sha256: "d".repeat(64),
        },
        frame,
        imagery: ImageAtlas {
            layout,
            pixels,
            coverage,
        },
        terrain: TerrainAtlas {
            layout,
            heights_m: vec![0.0; 1024 * 1024],
            coverage: vec![255; 1024 * 1024],
        },
    }
}

#[test]
fn a_cell_is_the_mean_imagery_of_its_ground_and_a_gap_is_unknown() {
    let scene = scene();
    let frame = scene.frame;
    let centre = frame.geodetic(Vector3::new(-300.0, 450.0, 0.0));
    let raster = map_raster(&scene, [centre[0], centre[1]], 0.0, (32.0, 40)).unwrap();
    let (mut known, mut unknown) = (0, 0);
    for north in -40_i64..=40 {
        for east in -40_i64..=40 {
            let mut sum = 0.0;
            let mut covered = true;
            for sy in 0..4 {
                for sx in 0..4 {
                    let x = -300.0 + (east as f64 - 0.5 + (f64::from(sx) + 0.5) / 4.0) * 32.0;
                    let y = 450.0 + (north as f64 - 0.5 + (f64::from(sy) + 0.5) / 4.0) * 32.0;
                    match imagery_luma(&scene.imagery, frame.mercator_xy(Vector3::new(x, y, 0.0))) {
                        Some(v) => sum += v,
                        None => covered = false,
                    }
                }
            }
            match (covered, raster.at(east, north)) {
                (true, Some(v)) => {
                    known += 1;
                    assert!((f64::from(v) - sum / 16.0).abs() < 0.5, "{east} {north}");
                }
                (false, None) => unknown += 1,
                other => panic!("{east} {north}: {other:?}"),
            }
        }
    }
    assert!(known > 4_000 && unknown > 100, "{known} {unknown}");
}
