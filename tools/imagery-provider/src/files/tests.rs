#![allow(clippy::expect_used)]
use super::*;
use image::{ImageFormat, RgbaImage};

fn png(image: &RgbaImage) -> Vec<u8> {
    let mut bytes = std::io::Cursor::new(Vec::new());
    image
        .write_to(&mut bytes, ImageFormat::Png)
        .expect("encode");
    bytes.into_inner()
}

#[test]
fn an_imported_source_publishes_a_readable_package() {
    let source = tempfile::tempdir().expect("source");
    let state = tempfile::tempdir().expect("state");
    let imagery = png(&RgbaImage::from_pixel(512, 512, Rgba([90, 120, 150, 255])));
    let terrain = png(&RgbaImage::from_pixel(256, 256, Rgba([128, 0, 0, 255])));
    std::fs::write(source.path().join("a.png"), &imagery).expect("imagery");
    std::fs::write(source.path().join("a.dem.png"), &terrain).expect("terrain");
    let manifest = serde_json::json!({
        "schema_version": 1, "release_id": "source-release", "anchor_lat_lon": [40.5, -74.4],
        "elevation_datum": "unknown", "attribution": "test",
        "tiles": [{"xyz": [16, 19214, 24677],
            "imagery": {"path": "a.png", "sha256": digest(&imagery)},
            "elevation": {"path": "a.dem.png", "sha256": digest(&terrain)}}]
    });
    std::fs::write(source.path().join("map.json"), manifest.to_string()).expect("manifest");
    load_catalog_blocking(state.path(), None).expect("state folders");
    let region = import_region_blocking(source.path(), state.path(), "region-a", "Region A")
        .expect("import");
    let package = &region.manifest;
    package.validate_for_reading().expect("readable");
    assert_eq!(package.compute_pack_id().expect("digest"), region.pack_id);
    assert_eq!(package.region_id, "region-a");
    for chunk in &package.files {
        let bytes = std::fs::read(
            state
                .path()
                .join("chunks")
                .join(format!("{}.bin", chunk.sha256)),
        )
        .expect("chunk");
        assert_eq!(digest(&bytes), chunk.sha256);
    }
}

#[test]
fn a_source_asset_outside_the_folder_is_refused() {
    let source = tempfile::tempdir().expect("source");
    let state = tempfile::tempdir().expect("state");
    let manifest = serde_json::json!({
        "schema_version": 2, "release_id": "r", "anchor_lat_lon": [40.5, -74.4],
        "elevation_datum": "unknown", "attribution": "test",
        "tiles": [{"xyz": [16, 1, 1], "imagery": {"path": "../a.png", "sha256": "a".repeat(64)}}]
    });
    std::fs::write(source.path().join("map.json"), manifest.to_string()).expect("manifest");
    assert!(import_region_blocking(source.path(), state.path(), "r", "R").is_err());
}
