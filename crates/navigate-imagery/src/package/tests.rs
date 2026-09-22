#![allow(clippy::expect_used)]
use super::*;

fn empty() -> Package {
    Package {
        schema_version: 1,
        region_id: "test".into(),
        release_id: "source-1".into(),
        anchor_lat_lon: [40.5, -74.4],
        elevation_datum: "unknown".into(),
        attribution: "test".into(),
        files: vec![],
        tiles: vec![],
        provenance: Some(serde_json::json!({"registration_error":"unknown"})),
        pack_id: String::new(),
    }
}

#[test]
fn ranges_recover_original_bytes_and_bind_source_identity() {
    let mut chunks = std::collections::BTreeMap::new();
    let mut builder = PackageBuilder::new(empty(), |sha: &str, data: &[u8]| {
        chunks.insert(sha.to_owned(), data.to_vec());
        Ok(())
    })
    .expect("builder");
    let a = vec![17; 3 * 1024 * 1024];
    let b = vec![23; 2 * 1024 * 1024];
    builder
        .add(Tile(16, 19212, 24674), false, &a, &digest(&a))
        .expect("imagery");
    builder
        .add(Tile(14, 4803, 6168), true, &b, &digest(&b))
        .expect("terrain");
    let pack = builder.finish().expect("pack");
    assert_eq!(pack.files.len(), 2);
    for (asset, expected) in [
        (pack.tiles[1].imagery.as_ref().expect("imagery"), a),
        (pack.tiles[0].elevation.as_ref().expect("terrain"), b),
    ] {
        let chunk = &chunks[&asset.chunk];
        assert_eq!(chunk[asset.offset..asset.offset + asset.length], expected);
        assert_eq!(digest(chunk), asset.chunk);
    }
    assert_eq!(
        pack.provenance.expect("provenance")["registration_error"],
        "unknown"
    );
    assert_eq!(pack.pack_id.len(), 64);
}

#[test]
fn corruption_duplicate_roles_and_missing_geometry_are_rejected() {
    let mut builder = PackageBuilder::new(empty(), |_, _| Ok(())).expect("builder");
    assert!(
        builder
            .add(Tile(16, 1, 1), false, b"tile", "wrong")
            .is_err()
    );
    builder
        .add(Tile(16, 1, 1), false, b"tile", &digest(b"tile"))
        .expect("tile");
    assert!(
        builder
            .add(Tile(16, 1, 1), false, b"tile", &digest(b"tile"))
            .is_err()
    );
    assert!(builder.finish().is_err());
}
