#![allow(clippy::expect_used)]
use super::*;
use crate::digest;
use std::collections::BTreeMap;

fn files() -> BTreeMap<String, Vec<u8>> {
    BTreeMap::from([
        ("a.png".to_owned(), vec![1; 40]),
        ("a.dem.png".to_owned(), vec![2; 30]),
    ])
}

fn source(schema_version: u32) -> SourceManifest {
    let files = files();
    let asset = |path: &str| SourceAsset {
        path: path.into(),
        sha256: digest(&files[path]),
    };
    SourceManifest {
        schema_version,
        release_id: "source-release".into(),
        anchor_lat_lon: [40.5, -74.4],
        elevation_datum: "unknown".into(),
        attribution: "test".into(),
        tiles: vec![SourceTile {
            xyz: Tile(16, 1, 2),
            imagery: Some(asset("a.png")),
            elevation: Some(asset("a.dem.png")),
        }],
        provenance: None,
    }
}

fn build(source: &SourceManifest, region: &str) -> Result<Package, ImageryError> {
    let files = files();
    source.build(
        region,
        |_, _, asset| Ok(files[&asset.path].clone()),
        |_, _| Ok::<(), ImageryError>(()),
    )
}

#[test]
fn one_source_builds_one_identity_per_region() {
    let first = build(&source(2), "region").expect("package");
    let again = build(&source(2), "region").expect("package");
    assert_eq!(first.pack_id, again.pack_id);
    assert_eq!(first.compute_pack_id().expect("digest"), first.pack_id);
    assert_ne!(
        build(&source(2), "other").expect("package").pack_id,
        first.pack_id
    );
    first.validate_for_reading().expect("readable");
}

#[test]
fn unsafe_paths_missing_roles_and_bad_digests_are_refused() {
    for path in ["../a.png", "/a.png", "a\\b.png", "a//b.png", ""] {
        let mut s = source(2);
        if let Some(asset) = s.tiles[0].imagery.as_mut() {
            asset.path = path.into();
        }
        assert!(s.validate().is_err(), "{path:?}");
    }
    let mut s = source(1);
    s.tiles[0].elevation = None;
    assert!(s.validate().is_err());
    let mut s = source(2);
    if let Some(asset) = s.tiles[0].imagery.as_mut() {
        asset.sha256 = "0".repeat(64);
    }
    assert!(build(&s, "region").is_err());
}
