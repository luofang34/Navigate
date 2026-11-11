use alloc::string::String;
use alloc::vec::Vec;
use core::cell::Cell;
use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use pollster::block_on;

use super::*;
use crate::manifest::tests::{elevation_spec, imagery_spec};
use crate::{
    Descriptor, FetchPort, Fetched, LayerKind, MemoryFile, PlannedTiles, TileAddress, TileKey,
    WritePort, prepare,
};

/// A future that is not ready on its first poll: a read that completes
/// later, as a browser store's does.
struct Later<T>(Option<T>, bool);

impl<T: Unpin> Future for Later<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<T> {
        if self.1 {
            return Poll::Ready(self.0.take().unwrap());
        }
        self.1 = true;
        context.waker().wake_by_ref();
        Poll::Pending
    }
}

/// A memory store whose every open completes on a later poll.
struct LaterStore {
    inner: MemoryStore,
    opens: Cell<usize>,
}

impl ReadPort for LaterStore {
    type File = MemoryFile;

    fn open(&self, path: &str) -> impl Future<Output = Result<Option<MemoryFile>, DataError>> {
        self.opens.set(self.opens.get() + 1);
        let opened = block_on(self.inner.open(path));
        Later(Some(opened), false)
    }
}

struct Tiles;

impl FetchPort for Tiles {
    async fn fetch(&self, key: &TileKey) -> Result<Fetched, String> {
        if key.address.x.is_multiple_of(3) {
            return Ok(Fetched::Absent);
        }
        Ok(Fetched::Tile(alloc::format!("{key}").into_bytes()))
    }
}

/// A package of a few tiles of each layer, with a gap, sealed in memory
/// beside a file of no package.
fn sealed() -> (MemoryStore, Sha256Digest) {
    let mut tiles = PlannedTiles::new();
    for (layer, zoom) in [(LayerKind::Imagery, 15), (LayerKind::Elevation, 12)] {
        let addresses = (0..4)
            .map(|x| TileAddress::new(zoom, 1_000 + x, 2_000).unwrap())
            .collect();
        tiles.insert(layer, addresses);
    }
    let mut store = MemoryStore::default();
    let summary = block_on(prepare(
        &tiles,
        &[imagery_spec(), elevation_spec()],
        &Descriptor::default(),
        100,
        &Tiles,
        &mut store,
    ))
    .unwrap();
    block_on(store.write("chunks/unrelated", b"not of the package")).unwrap();
    (store, summary.identity)
}

#[test]
fn the_copy_holds_the_package_and_nothing_else() {
    let (source, identity) = sealed();
    let later = LaterStore {
        inner: source.clone(),
        opens: Cell::new(0),
    };
    let copy = block_on(resident_copy(&later, identity, None)).unwrap();
    assert!(later.opens.get() > 0);
    let source_paths: Vec<&str> = source
        .paths()
        .filter(|p| *p != "chunks/unrelated")
        .collect();
    assert_eq!(copy.store.paths().collect::<Vec<_>>(), source_paths);
    for path in copy.store.paths() {
        assert_eq!(copy.store.get(path), source.get(path), "{path}");
    }
    let bytes: u64 = source_paths
        .iter()
        .map(|p| source.get(p).unwrap().len() as u64)
        .sum();
    assert_eq!(copy.bytes, bytes);
    let opened = block_on(Package::open(&copy.store, identity)).unwrap();
    assert_eq!(block_on(opened.verify_all(&copy.store)).unwrap(), 6);
}

#[test]
fn a_tile_the_store_lacks_is_lacking_in_the_copy() {
    let (mut source, identity) = sealed();
    let package = block_on(Package::open(&source, identity)).unwrap();
    let blobs: Vec<Sha256Digest> = block_on(package.blobs()).unwrap().into_iter().collect();
    let lost = blob_path(blobs[0]);
    assert!(source.remove(&lost));
    let copy = block_on(resident_copy(&source, identity, None)).unwrap();
    assert!(copy.store.get(&lost).is_none());
    let opened = block_on(Package::open(&copy.store, identity)).unwrap();
    let on_copy = block_on(opened.verify_all(&copy.store)).unwrap_err();
    let on_source = block_on(package.verify_all(&source)).unwrap_err();
    assert_eq!(on_copy, on_source);
}

#[test]
fn a_package_that_does_not_open_is_refused() {
    let (source, identity) = sealed();
    let other = Sha256Digest::of(b"another package");
    assert!(matches!(
        block_on(resident_copy(&source, other, None)),
        Err(DataError::Missing { .. })
    ));
    let mut changed = source.clone();
    changed.replace(&manifest_path(identity), b"{}");
    assert!(matches!(
        block_on(resident_copy(&changed, identity, None)),
        Err(DataError::Integrity { .. })
    ));
    // A product list the store lacks is not copied, and the reader refuses
    // it there as it would on the source.
    let copy = block_on(resident_copy(&source, identity, Some(other))).unwrap();
    assert!(copy.store.get(&list_path(other)).is_none());
}
