use super::*;
use crate::manifest::tests::{elevation_layer, imagery_layer};
use crate::{MemoryStore, WritePort, manifest_path};
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

/// Writes a package with the tiles in `order`, the first one a gap.
#[cfg(test)]
fn build(store: &mut MemoryStore, order: &[TileAddress], release: &str) -> Sha256Digest {
    let mut writer = PackageWriter::new(release);
    writer.add_layer(elevation_layer()).unwrap();
    writer.add_layer(imagery_layer()).unwrap();
    for (i, &a) in order.iter().enumerate() {
        if a == addresses()[0] {
            writer.add_gap("imagery", a).unwrap();
        } else {
            block_on(writer.add_tile(store, "imagery", a, &bytes_of(a))).unwrap();
        }
        if i % 2 == 0 {
            block_on(writer.add_tile(store, "elevation", a.ancestor(2), &bytes_of(a.ancestor(2))))
                .ok();
        }
    }
    block_on(writer.seal(store)).unwrap()
}

#[test]
fn a_sealed_package_opens_and_reads_tile_for_tile() {
    let mut store = MemoryStore::default();
    let id = build(&mut store, &addresses(), "r1");
    let package = block_on(Package::open(&store, id)).unwrap();
    assert_eq!(package.identity(), id);
    assert_eq!(package.release_id(), "r1");
    for &a in &addresses()[1..] {
        let read = block_on(package.tile(&store, "imagery", a)).unwrap();
        assert_eq!(read, TileRead::Present(bytes_of(a)));
    }
    assert_eq!(
        block_on(package.tile(&store, "imagery", addresses()[0])).unwrap(),
        TileRead::Gap
    );
    let outside = TileAddress::new(12, 1, 1).unwrap();
    assert_eq!(
        block_on(package.tile(&store, "imagery", outside)).unwrap(),
        TileRead::Outside
    );
    assert!(block_on(package.tile(&store, "lidar", outside)).is_err());
    assert_eq!(
        block_on(package.verify_all(&store)).unwrap(),
        addresses().len()
    );
}

#[test]
fn identity_does_not_depend_on_the_order_of_the_tiles() {
    let mut forward = MemoryStore::default();
    let mut reverse = MemoryStore::default();
    let mut order = addresses();
    let a = build(&mut forward, &order, "r1");
    order.reverse();
    let b = build(&mut reverse, &order, "r1");
    assert_eq!(a, b);
    let mut other = MemoryStore::default();
    assert_ne!(a, build(&mut other, &addresses(), "r2"));
}

#[test]
fn overlapping_packages_share_blobs() {
    let mut store = MemoryStore::default();
    build(&mut store, &addresses(), "r1");
    let after_one = store.files.len();
    build(&mut store, &addresses()[..6], "r2");
    assert_eq!(
        store.files.len(),
        after_one + 1,
        "only the second manifest is new"
    );
}

#[test]
fn a_changed_blob_is_found_when_it_is_read() {
    let mut store = MemoryStore::default();
    let id = build(&mut store, &addresses(), "r1");
    let package = block_on(Package::open(&store, id)).unwrap();
    let victim = addresses()[4];
    let path = blob_path(Sha256Digest::of(&bytes_of(victim)));
    store.files.insert(path.clone(), b"changed".to_vec());
    assert!(matches!(
        block_on(package.tile(&store, "imagery", victim)),
        Err(DataError::Integrity { path: p, .. }) if p == path
    ));
    assert_eq!(
        block_on(package.tile(&store, "imagery", addresses()[5])).unwrap(),
        TileRead::Present(bytes_of(addresses()[5]))
    );
    assert!(block_on(package.verify_all(&store)).is_err());
    store.files.remove(&path);
    assert!(matches!(
        block_on(package.tile(&store, "imagery", victim)),
        Err(DataError::Missing { .. })
    ));
}

#[test]
fn a_changed_manifest_does_not_open() {
    let mut store = MemoryStore::default();
    let id = build(&mut store, &addresses(), "r1");
    let path = manifest_path(id);
    let mut bytes = store.files.get(&path).cloned().unwrap();
    let at = bytes.iter().position(|&b| b == b'r').unwrap();
    if let Some(b) = bytes.get_mut(at) {
        *b = b's';
    }
    store.files.insert(path, bytes);
    assert!(matches!(
        block_on(Package::open(&store, id)),
        Err(DataError::Integrity { .. })
    ));
    let unknown = Sha256Digest::of(b"no such package");
    assert!(matches!(
        block_on(Package::open(&store, unknown)),
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
    let mut writer = PackageWriter::new("r1");
    writer.add_layer(imagery_layer()).unwrap();
    let mut failed = false;
    for a in addresses() {
        failed |= block_on(writer.add_tile(&mut store, "imagery", a, &bytes_of(a))).is_err();
    }
    assert!(failed);
    assert!(matches!(
        block_on(writer.seal(&mut store)),
        Err(DataError::Missing { .. })
    ));
    assert!(store.inner.files.keys().all(|p| p.starts_with("chunks/")));
}

#[test]
fn a_tile_cannot_be_declared_twice() {
    let mut store = MemoryStore::default();
    let mut writer = PackageWriter::new("r1");
    writer.add_layer(imagery_layer()).unwrap();
    let a = addresses()[0];
    block_on(writer.add_tile(&mut store, "imagery", a, b"x")).unwrap();
    assert!(block_on(writer.add_tile(&mut store, "imagery", a, b"y")).is_err());
    assert!(writer.add_gap("imagery", a).is_err());
    assert!(writer.add_layer(imagery_layer()).is_err());
}
