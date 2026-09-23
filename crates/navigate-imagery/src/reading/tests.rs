#![allow(clippy::expect_used)]
use super::*;
use crate::PackageBuilder;

fn package() -> (Package, Vec<u8>) {
    let mut stored = Vec::new();
    let manifest = Package {
        schema_version: 1,
        region_id: "r".into(),
        release_id: "one".into(),
        anchor_lat_lon: [40.0, -74.0],
        elevation_datum: "unknown".into(),
        attribution: "test".into(),
        files: vec![],
        tiles: vec![],
        provenance: None,
        supersedes: None,
        pack_id: String::new(),
    };
    let mut builder = PackageBuilder::new(manifest, |_, data| {
        stored = data.to_vec();
        Ok(())
    })
    .expect("builder");
    let (a, b) = (vec![3; 100], vec![4; 50]);
    builder
        .add(Tile(16, 1, 2), false, &a, &digest(&a))
        .expect("imagery");
    builder
        .add(Tile(16, 1, 2), true, &b, &digest(&b))
        .expect("terrain");
    let pack = builder.finish().expect("pack");
    (pack, stored)
}

#[test]
fn a_built_package_passes_reading_checks() {
    let (pack, chunk) = package();
    pack.validate_for_reading().expect("valid");
    for asset in [&pack.tiles[0].imagery, &pack.tiles[0].elevation]
        .into_iter()
        .flatten()
    {
        verify_asset(asset, &chunk).expect("asset");
    }
}

#[test]
fn reading_checks_refuse_bad_identities_and_ranges() {
    let (pack, chunk) = package();
    let mut bad = pack.clone();
    bad.pack_id = "ABC".into();
    assert!(bad.validate_for_reading().is_err());
    let mut bad = pack.clone();
    bad.supersedes = Some("not-a-digest".into());
    assert!(bad.validate_for_reading().is_err());
    let mut asset = pack.tiles[0].imagery.clone().expect("imagery");
    asset.offset = chunk.len();
    assert!(verify_asset(&asset, &chunk).is_err());
    asset.offset = 1;
    assert!(verify_asset(&asset, &chunk).is_err());
}
