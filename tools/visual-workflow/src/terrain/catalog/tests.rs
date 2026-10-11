use super::*;
use image::{Rgb, RgbImage};
use sha2::{Digest, Sha256};

fn tile(root: &Path, kind: &str, rgb: [u8; 3]) -> Tile {
    let path = PathBuf::from(format!("{kind}.png"));
    RgbImage::from_pixel(256, 256, Rgb(rgb))
        .save(root.join(&path))
        .unwrap();
    let bytes = std::fs::read(root.join(&path)).unwrap();
    Tile {
        kind: kind.into(),
        xyz: [14, 8192, 8192],
        path,
        sha256: format!("{:x}", Sha256::digest(&bytes)),
        bytes: bytes.len(),
    }
}

#[test]
fn verified_catalog_decodes_terrarium_and_rejects_changed_tiles() {
    let root = tempfile::tempdir().unwrap();
    let mut catalog = Catalog {
        format: "photo-reference-catalog".into(),
        version: 1,
        complete: true,
        configuration: Configuration {
            bbox: [-0.01, -0.01, 0.01, 0.01],
            dem_zoom: 14,
            imagery_zoom: 14,
        },
        tiles: vec![
            tile(root.path(), "imagery", [20; 3]),
            tile(root.path(), "dem", [128, 1, 128]),
        ],
    };
    let config = SceneConfig {
        catalog: "catalog.json".into(),
        catalog_sha256: "a".repeat(64),
        imagery_tile_pixels: 128,
        scene_bytes: 1_000_000,
        far_m: 5000.0,
        steps: 100,
        refinements: 12,
    };
    let map = || MapRevision {
        release_id: "test".into(),
        manifest_sha256: "a".repeat(64),
    };
    let frame = LocalFrame::spherical_enu(0.0, 0.0).unwrap();
    let scene = catalog
        .load_blocking(root.path(), &config, frame, map())
        .unwrap();
    assert_eq!(scene.imagery.layout.size, [128, 128]);
    assert!(scene.imagery.pixels.iter().all(|p| *p == 20));
    assert!(scene.terrain.heights_m.iter().all(|h| *h == 1.5));
    assert!(scene.terrain.coverage.iter().all(|c| *c == 255));
    catalog.tiles[0].sha256 = "b".repeat(64);
    assert!(
        catalog
            .load_blocking(root.path(), &config, frame, map())
            .is_err()
    );
}

#[test]
fn inconsistent_catalog_coordinates_fail_before_atlas_allocation() {
    let root = tempfile::tempdir().unwrap();
    let mut catalog = Catalog {
        format: "photo-reference-catalog".into(),
        version: 1,
        complete: true,
        configuration: Configuration {
            bbox: [-0.01, -0.01, 0.01, 0.01],
            dem_zoom: 14,
            imagery_zoom: 14,
        },
        tiles: vec![tile(root.path(), "imagery", [20; 3])],
    };
    assert_eq!(catalog.layout("imagery", 14, 128).unwrap().size, [128, 128]);
    catalog.tiles.push(tile(root.path(), "imagery", [20; 3]));
    assert!(catalog.layout("imagery", 14, 128).is_err());
    catalog.tiles.pop();
    catalog.tiles[0].xyz[0] = 15;
    assert!(catalog.layout("imagery", 14, 128).is_err());
}
