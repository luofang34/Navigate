use super::*;
use crate::manifest::tests::{ELEVATION, IMAGERY, elevation_spec, imagery_spec};
use crate::{MemoryFile, MemoryStore, WritePort};
use pollster::block_on;

#[cfg(test)]
fn addresses() -> Vec<TileAddress> {
    let mut out = Vec::new();
    for y in 1000..1004 {
        for x in 2000..2003 {
            out.push(TileAddress::new(12, x, y).unwrap());
        }
    }
    out
}

#[cfg(test)]
fn bytes_of(a: TileAddress) -> Vec<u8> {
    alloc::format!("tile {} {} {}", a.zoom, a.x, a.y).into_bytes()
}

/// Writes a package with the imagery tiles in `order`, the first address a
/// gap, and elevation at zoom 10.
#[cfg(test)]
fn build(store: &mut MemoryStore, order: &[TileAddress]) -> Sha256Digest {
    let mut writer = PackageWriter::new();
    writer.add_layer(elevation_spec()).unwrap();
    writer.add_layer(imagery_spec()).unwrap();
    for &a in order {
        if a == addresses()[0] {
            writer.add_gap(IMAGERY, a).unwrap();
        } else {
            block_on(writer.add_tile(store, IMAGERY, a, &bytes_of(a))).unwrap();
        }
    }
    let coarse = addresses()[0].ancestor(2);
    block_on(writer.add_tile(store, ELEVATION, coarse, &bytes_of(coarse))).unwrap();
    block_on(writer.seal(store, &Descriptor::default())).unwrap()
}

#[cfg(test)]
fn open(store: &MemoryStore, id: Sha256Digest) -> Result<Package<MemoryFile>, DataError> {
    block_on(Package::open(store, id))
}

#[test]
fn a_sealed_package_opens_and_reads_tile_for_tile() {
    let mut store = MemoryStore::default();
    let id = build(&mut store, &addresses());
    let package = open(&store, id).unwrap();
    assert_eq!(package.identity(), id);
    for &a in &addresses()[1..] {
        assert_eq!(
            block_on(package.tile(&store, IMAGERY, a)).unwrap(),
            TileRead::Present(bytes_of(a))
        );
    }
    assert_eq!(
        block_on(package.tile(&store, IMAGERY, addresses()[0])).unwrap(),
        TileRead::Gap
    );
    let outside = TileAddress::new(12, 1, 1).unwrap();
    assert_eq!(
        block_on(package.tile(&store, IMAGERY, outside)).unwrap(),
        TileRead::Outside
    );
    let other = LayerKey {
        kind: crate::LayerKind::Imagery,
        source: crate::SourceKey::Synthetic,
    };
    assert!(block_on(package.tile(&store, other, outside)).is_err());
    assert!(block_on(package.descriptor(&store)).unwrap().is_some());
    assert_eq!(
        block_on(package.verify_all(&store)).unwrap(),
        addresses().len()
    );
    let records = block_on(package.layer_entries(IMAGERY)).unwrap();
    assert_eq!(records.len(), addresses().len());
}

#[test]
fn identity_does_not_depend_on_the_order_of_the_tiles() {
    let mut forward = MemoryStore::default();
    let mut reverse = MemoryStore::default();
    let mut order = addresses();
    let a = build(&mut forward, &order);
    order.reverse();
    assert_eq!(a, build(&mut reverse, &order));
}

#[test]
fn overlapping_packages_share_blobs() {
    let mut store = MemoryStore::default();
    build(&mut store, &addresses());
    let before: Vec<String> = store.paths().map(String::from).collect();
    build(&mut store, &addresses()[..6]);
    let new: Vec<_> = store
        .paths()
        .filter(|p| !before.iter().any(|b| b == p))
        .collect();
    // The second package adds its imagery index, its manifest, and its
    // descriptor; every tile blob and the elevation index are shared.
    assert_eq!(new.len(), 3, "{new:?}");
}

#[test]
fn a_changed_blob_is_found_when_it_is_read() {
    let mut store = MemoryStore::default();
    let id = build(&mut store, &addresses());
    let victim = addresses()[4];
    let path = blob_path(Sha256Digest::of(&bytes_of(victim)));
    store.replace(&path, b"changed");
    let package = open(&store, id).unwrap();
    assert!(matches!(
        block_on(package.tile(&store, IMAGERY, victim)),
        Err(DataError::Integrity { path: p, .. }) if p == path
    ));
    assert_eq!(
        block_on(package.tile(&store, IMAGERY, addresses()[5])).unwrap(),
        TileRead::Present(bytes_of(addresses()[5]))
    );
    store.remove(&path);
    assert!(matches!(
        block_on(package.tile(&store, IMAGERY, victim)),
        Err(DataError::Missing { .. })
    ));
}

#[test]
fn a_changed_manifest_or_index_does_not_open() {
    let mut store = MemoryStore::default();
    let id = build(&mut store, &addresses());
    let package = open(&store, id).unwrap();
    let index = blob_path(package.manifest().layers[1].index.blob);
    let mut bytes = store.get(&index).unwrap().to_vec();
    if let Some(b) = bytes.last_mut() {
        *b ^= 1;
    }
    store.replace(&index, &bytes);
    assert!(matches!(open(&store, id), Err(DataError::Integrity { .. })));
    let manifest = manifest_path(id);
    store.replace(&manifest, b"{}");
    assert!(matches!(open(&store, id), Err(DataError::Integrity { .. })));
    assert!(matches!(
        open(&store, Sha256Digest::of(b"none")),
        Err(DataError::Missing { .. })
    ));
}

/// A store that refuses writes after a count, as an interrupted install.
#[cfg(test)]
struct Interrupted {
    inner: MemoryStore,
    writes_left: usize,
}

impl WritePort for Interrupted {
    async fn write(&mut self, path: &str, bytes: &[u8]) -> Result<(), DataError> {
        if self.writes_left == 0 {
            return Err(DataError::Store {
                path: path.into(),
                message: "interrupted".into(),
            });
        }
        self.writes_left -= 1;
        self.inner.write(path, bytes).await
    }

    async fn contains(&self, path: &str) -> Result<bool, DataError> {
        self.inner.contains(path).await
    }
}

#[test]
fn an_interrupted_write_leaves_no_manifest() {
    let mut store = Interrupted {
        inner: MemoryStore::default(),
        writes_left: 3,
    };
    let mut writer = PackageWriter::new();
    writer.add_layer(imagery_spec()).unwrap();
    let mut failed = false;
    for a in addresses() {
        failed |= block_on(writer.add_tile(&mut store, IMAGERY, a, &bytes_of(a))).is_err();
    }
    assert!(failed);
    assert!(matches!(
        block_on(writer.seal(&mut store, &Descriptor::default())),
        Err(DataError::Missing { .. })
    ));
    assert!(store.inner.paths().all(|p| p.starts_with("chunks/")));
}

#[test]
fn a_tile_cannot_be_declared_twice() {
    let mut store = MemoryStore::default();
    let mut writer = PackageWriter::new();
    writer.add_layer(imagery_spec()).unwrap();
    let a = addresses()[0];
    block_on(writer.add_tile(&mut store, IMAGERY, a, b"x")).unwrap();
    assert!(block_on(writer.add_tile(&mut store, IMAGERY, a, b"y")).is_err());
    assert!(writer.add_gap(IMAGERY, a).is_err());
    assert!(writer.add_layer(imagery_spec()).is_err());
    assert!(block_on(writer.add_tile(&mut store, IMAGERY, addresses()[1], b"")).is_err());
}

#[test]
fn correlation_keys_follow_the_source_not_the_package() {
    let mut store = MemoryStore::default();
    let id_a = build(&mut store, &addresses());
    let id_b = build(&mut store, &addresses()[..6]);
    let (a, b) = (open(&store, id_a).unwrap(), open(&store, id_b).unwrap());
    assert_ne!(a.identity(), b.identity());
    let point = [41.7, -1.0];
    let key_a = a
        .manifest()
        .layer(IMAGERY)
        .unwrap()
        .correlation_key(point)
        .unwrap();
    let key_b = b
        .manifest()
        .layer(IMAGERY)
        .unwrap()
        .correlation_key(point)
        .unwrap();
    assert_eq!(key_a, key_b);
    let coarse = a
        .manifest()
        .layer(ELEVATION)
        .unwrap()
        .correlation_key(point)
        .unwrap();
    assert_eq!(coarse.cell.0.zoom, 12);
}
