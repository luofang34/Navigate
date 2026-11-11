//! The package manifest and its canonical encoding.
//!
//! The canonical encoding is compact JSON with the fields in declaration
//! order, layers in identifier order, zoom levels in increasing order, and
//! tiles and gaps in row, then column order. A reader decodes the manifest,
//! encodes it again, and refuses it when the bytes differ. The SHA-256 of
//! the canonical bytes is the package identity.

use alloc::string::String;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::{
    Accuracy, DataError, Encoding, GridSpec, LayerKind, Sha256Digest, Sigma, SourceRef,
    TileAddress, VerticalDatum,
};

mod validate;

/// The `format` of a package manifest.
pub const PACKAGE_FORMAT: &str = "navigate.data.package";
/// The `version` that this crate writes and reads.
pub const PACKAGE_FORMAT_VERSION: u32 = 1;

/// The manifest of one mission package.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// [`PACKAGE_FORMAT`].
    pub format: String,
    /// [`PACKAGE_FORMAT_VERSION`].
    pub version: u32,
    /// Release identifier that the host gives the package.
    pub release_id: String,
    /// The fixed grid of correlation cells.
    pub grid: GridSpec,
    /// The layers, in identifier order.
    pub layers: Vec<Layer>,
}

/// One layer of a package.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    /// Identifier, unique in the package, for example `imagery`.
    pub id: String,
    /// What the layer holds.
    pub kind: LayerKind,
    /// Where the content comes from.
    pub source: SourceRef,
    /// Encoding of each blob.
    pub encoding: Encoding,
    /// Pixels on one side of a tile.
    pub tile_pixels: u32,
    /// Vertical datum of an elevation layer; `None` for imagery.
    pub vertical_datum: Option<VerticalDatum>,
    /// Declared accuracy.
    pub accuracy: Accuracy,
    /// The declared coverage at each zoom level.
    pub zooms: Vec<LayerZoom>,
}

/// The declared coverage of a layer at one zoom level: each address is a
/// tile or a gap.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerZoom {
    /// Zoom level.
    pub zoom: u32,
    /// Tiles that the package holds.
    pub tiles: Vec<TileRef>,
    /// Addresses in the declared coverage that the source did not have,
    /// `[x, y]`.
    pub gaps: Vec<[u32; 2]>,
}

/// A tile that the package holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TileRef {
    /// Column.
    pub x: u32,
    /// Row.
    pub y: u32,
    /// Digest of the blob.
    pub blob: Sha256Digest,
    /// Size of the blob in bytes.
    pub bytes: u64,
}

/// What a layer declares at one address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Declared<'a> {
    /// The package holds this tile.
    Tile(&'a TileRef),
    /// The address is in the coverage, but the source did not have it.
    Gap,
    /// The address is outside the declared coverage.
    Outside,
}

impl Manifest {
    /// The canonical bytes of the manifest.
    ///
    /// # Errors
    ///
    /// Refuses a manifest that breaks the ordering and value rules.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, DataError> {
        self.validate()?;
        serde_json::to_vec(self).map_err(|error| DataError::Manifest {
            path: String::from("<memory>"),
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
    /// Refuses bytes that do not decode, a manifest that breaks the rules,
    /// and bytes that are not the canonical encoding of what they decode
    /// to.
    pub fn from_canonical(bytes: &[u8], path: &str) -> Result<Self, DataError> {
        let refuse = |detail: String| DataError::Manifest {
            path: String::from(path),
            detail,
        };
        let manifest: Self =
            serde_json::from_slice(bytes).map_err(|error| refuse(alloc::format!("{error}")))?;
        let canonical = manifest
            .canonical_bytes()
            .map_err(|error| refuse(alloc::format!("{error}")))?;
        if canonical != bytes {
            return Err(refuse(String::from("the bytes are not in canonical form")));
        }
        Ok(manifest)
    }

    /// The layer with identifier `id`.
    #[must_use]
    pub fn layer(&self, id: &str) -> Option<&Layer> {
        self.layers
            .binary_search_by(|layer| layer.id.as_str().cmp(id))
            .ok()
            .and_then(|i| self.layers.get(i))
    }

    /// Every blob that the package refers to, with its size.
    pub fn blobs(&self) -> impl Iterator<Item = (Sha256Digest, u64)> + '_ {
        self.layers
            .iter()
            .flat_map(|layer| layer.zooms.iter())
            .flat_map(|zoom| zoom.tiles.iter())
            .map(|tile| (tile.blob, tile.bytes))
    }
}

impl Layer {
    /// What the layer declares at `address`.
    #[must_use]
    pub fn declared(&self, address: TileAddress) -> Declared<'_> {
        let Some(zoom) = self
            .zooms
            .binary_search_by_key(&address.zoom, |z| z.zoom)
            .ok()
            .and_then(|i| self.zooms.get(i))
        else {
            return Declared::Outside;
        };
        let key = (address.y, address.x);
        if let Some(tile) = zoom
            .tiles
            .binary_search_by_key(&key, |t| (t.y, t.x))
            .ok()
            .and_then(|i| zoom.tiles.get(i))
        {
            return Declared::Tile(tile);
        }
        if zoom
            .gaps
            .binary_search_by_key(&key, |&[x, y]| (y, x))
            .is_ok()
        {
            return Declared::Gap;
        }
        Declared::Outside
    }

    /// The horizontal sigma of the layer after `age_years`.
    #[must_use]
    pub fn horizontal_sigma_after(&self, age_years: f64) -> Sigma {
        self.accuracy.horizontal_after(age_years)
    }
}

#[cfg(test)]
pub(crate) mod tests;
