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
        supersedes: None,
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

type Store = std::collections::BTreeMap<String, Vec<u8>>;

fn parent(chunks: &mut Store) -> Package {
    let mut builder = PackageBuilder::new(empty(), |sha: &str, data: &[u8]| {
        chunks.insert(sha.to_owned(), data.to_vec());
        Ok(())
    })
    .expect("builder");
    let imagery = vec![1; 1024];
    let terrain = vec![2; 1024];
    builder
        .add(Tile(16, 19212, 24674), false, &imagery, &digest(&imagery))
        .expect("imagery");
    builder
        .add(Tile(16, 19212, 24674), true, &terrain, &digest(&terrain))
        .expect("terrain");
    builder.finish().expect("parent")
}

#[test]
fn a_derived_package_replaces_one_asset_and_names_every_producer() {
    let mut chunks = Store::new();
    let parent = parent(&mut chunks);
    let refined = vec![9; 1024];
    let mut builder = PackageBuilder::from_parent(&parent, "flight-7".into(), |sha, data| {
        chunks.insert(sha.to_owned(), data.to_vec());
        Ok(())
    })
    .expect("derived builder");
    builder
        .add(Tile(16, 19212, 24674), false, &refined, &digest(&refined))
        .expect("replace imagery once");
    let child = builder.finish().expect("child");
    assert_eq!(child.supersedes.as_deref(), Some(parent.pack_id.as_str()));
    assert_eq!(child.region_id, parent.region_id);
    assert_eq!(child.release_id, "flight-7");
    assert_ne!(child.pack_id, parent.pack_id);
    assert_eq!(child.compute_pack_id().expect("digest"), child.pack_id);
    let tile = &child.tiles[0];
    let imagery = tile.imagery.as_ref().expect("imagery");
    let terrain = tile.elevation.as_ref().expect("terrain");
    assert_eq!(imagery.sha256, digest(&refined));
    assert_eq!(imagery.produced_by, None);
    assert_eq!(terrain.produced_by.as_deref(), Some("source-1"));
    let referenced: Vec<_> = [imagery, terrain].map(|a| a.chunk.clone()).into();
    assert!(child.files.iter().all(|f| referenced.contains(&f.sha256)));
}

#[test]
fn a_derived_package_refuses_a_second_replacement_and_a_tampered_parent() {
    let mut chunks = Store::new();
    let parent = parent(&mut chunks);
    let bytes = vec![5; 64];
    let mut builder =
        PackageBuilder::from_parent(&parent, "flight-8".into(), |_, _| Ok(())).expect("builder");
    let tile = Tile(16, 19212, 24674);
    builder
        .add(tile, true, &bytes, &digest(&bytes))
        .expect("first replacement");
    assert!(builder.add(tile, true, &bytes, &digest(&bytes)).is_err());
    let mut tampered = parent.clone();
    tampered.attribution = "changed".into();
    assert!(PackageBuilder::from_parent(&tampered, "flight-9".into(), |_, _| Ok(())).is_err());
    assert!(PackageBuilder::from_parent(&parent, "source-1".into(), |_, _| Ok(())).is_err());
}

#[test]
fn packages_without_lineage_keep_their_existing_pack_id() {
    let mut chunks = Store::new();
    let parent = parent(&mut chunks);
    let text = serde_json::to_string(&parent).expect("json");
    assert!(!text.contains("supersedes"));
    assert!(!text.contains("produced_by"));
}
