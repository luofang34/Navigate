//! The fetch port. The host implements it; Navigate never fetches.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;
use core::future::Future;

use crate::{LayerKind, Revision, SourceKey, TileAddress};

/// The identity of one source tile: source, revision, layer, and address.
/// It names the content, not a URL.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileKey {
    /// Source.
    pub source: SourceKey,
    /// Source revision.
    pub revision: Revision,
    /// Layer kind.
    pub layer: LayerKind,
    /// Tile address.
    pub address: TileAddress,
}

impl fmt::Display for TileKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a = self.address;
        write!(
            f,
            "{:?}@{:?}/{:?}/{}/{}/{}",
            self.source, self.revision, self.layer, a.zoom, a.x, a.y
        )
    }
}

/// The answer of a source for one tile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fetched {
    /// The encoded tile.
    Tile(Vec<u8>),
    /// The source does not have this tile.
    Absent,
}

/// A source of tiles that the host implements, for example over HTTP with
/// a cache keyed by [`TileKey`]. Builders call it when a host prepares a
/// package.
pub trait FetchPort {
    /// Gets one tile. An error is a failure of the port; a tile that the
    /// source does not have is [`Fetched::Absent`].
    fn fetch(&self, key: &TileKey) -> impl Future<Output = Result<Fetched, String>>;
}
