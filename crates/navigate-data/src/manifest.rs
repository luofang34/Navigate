//! The package manifest and its canonical encoding.
//!
//! Canonical encoding: compact JSON with the object fields in the declared
//! order of each type and no unknown field; numbers are integers only
//! (millimetres, day numbers, counts, zooms); strings are only names from
//! closed vocabularies or 64-digit lower-case hexadecimal digests; each
//! collection is sorted in its stated order with no repeats; each optional
//! value is present, as `null` when it is absent. A reader decodes the
//! manifest, encodes it again, and refuses bytes that differ. The SHA-256
//! of the canonical bytes is the package identity.
//!
//! The manifest holds only content and its physical meaning. Labels,
//! attribution, licence text, and host notes are in the
//! [`Descriptor`](crate::Descriptor), outside the identity.

use alloc::string::String;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::{
    Accuracy, Body, DataError, EllipsoidId, Encoding, GridScheme, HorizontalDatum, LayerKind,
    Revision, Sha256Digest, SourceKey, VerticalDatum,
};

mod validate;

/// The `format` of a package manifest.
pub const PACKAGE_FORMAT: &str = "navigate.data.package";
/// The one `version` that this crate writes and reads.
pub const PACKAGE_FORMAT_VERSION: u32 = 1;

/// The manifest of one mission package.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// [`PACKAGE_FORMAT`].
    pub format: String,
    /// [`PACKAGE_FORMAT_VERSION`].
    pub version: u32,
    /// The body.
    pub body: Body,
    /// Its reference ellipsoid.
    pub ellipsoid: EllipsoidId,
    /// The tile scheme.
    pub grid: GridScheme,
    /// The layers, in [`LayerKey`] order.
    pub layers: Vec<Layer>,
}

/// The identity of a layer in a package: one layer for each kind and
/// source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerKey {
    /// What the layer holds.
    pub kind: LayerKind,
    /// Where the content comes from.
    pub source: SourceKey,
}

/// One layer of a package.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    /// Kind and source.
    pub key: LayerKey,
    /// The content version that the source states.
    pub revision: Revision,
    /// Horizontal datum of the tile positions.
    pub horizontal_datum: HorizontalDatum,
    /// Vertical datum of an elevation layer; `None` for imagery.
    pub vertical_datum: Option<VerticalDatum>,
    /// Encoding of each blob.
    pub encoding: Encoding,
    /// Pixels on one side of a tile.
    pub tile_pixels: u32,
    /// Declared accuracy.
    pub accuracy: Accuracy,
    /// Zoom of the correlation cells of this layer: map error is taken as
    /// shared inside one cell and independent between cells.
    pub correlation_zoom: u32,
    /// The tile index blob.
    pub index: IndexRef,
}

/// The tile index of a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexRef {
    /// Digest of the index blob.
    pub blob: Sha256Digest,
    /// Size of the index blob in bytes.
    pub bytes: u64,
    /// Records that are tiles.
    pub tiles: u64,
    /// Records that are gaps.
    pub gaps: u64,
}

impl IndexRef {
    /// Records in the index.
    #[must_use]
    pub const fn records(&self) -> u64 {
        self.tiles.saturating_add(self.gaps)
    }
}

impl Manifest {
    /// The canonical bytes of the manifest.
    ///
    /// # Errors
    ///
    /// Refuses a manifest that [`Self::validate`] refuses.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, DataError> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|error| DataError::Document {
            path: String::from("<manifest>"),
            detail: alloc::format!("{error}"),
        })
    }

    /// The package identity: the digest of the canonical bytes.
    ///
    /// # Errors
    ///
    /// As [`Self::canonical_bytes`].
    pub fn identity(&self) -> Result<Sha256Digest, DataError> {
        self.canonical_bytes().map(|bytes| Sha256Digest::of(&bytes))
    }

    /// Decodes canonical manifest bytes read from `path`.
    ///
    /// # Errors
    ///
    /// Refuses bytes that do not decode, a manifest of another version, a
    /// value that this crate does not support, a broken rule, and bytes
    /// that are not the canonical encoding of what they decode to.
    pub fn from_canonical(bytes: &[u8], path: &str) -> Result<Self, DataError> {
        let refuse = |detail: String| DataError::Document {
            path: String::from(path),
            detail,
        };
        let manifest: Self =
            serde_json::from_slice(bytes).map_err(|error| refuse(alloc::format!("{error}")))?;
        if manifest.canonical_bytes()? != bytes {
            return Err(refuse(String::from("the bytes are not in canonical form")));
        }
        Ok(manifest)
    }

    /// The layer with `key`.
    #[must_use]
    pub fn layer(&self, key: LayerKey) -> Option<&Layer> {
        self.layers
            .binary_search_by(|layer| layer.key.cmp(&key))
            .ok()
            .and_then(|i| self.layers.get(i))
    }

    /// The first layer of `kind`.
    #[must_use]
    pub fn layer_of_kind(&self, kind: LayerKind) -> Option<&Layer> {
        self.layers.iter().find(|layer| layer.key.kind == kind)
    }
}

#[cfg(test)]
pub(crate) mod tests;
