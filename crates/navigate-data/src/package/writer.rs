//! Writing a package: tile blobs, then tile indexes, then the manifest.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::index::{IndexEntry, IndexKey, encode_index, key};
use crate::{
    Accuracy, Body, DataError, Descriptor, EllipsoidId, Encoding, GridScheme, HorizontalDatum,
    IndexRef, Layer, LayerKey, Manifest, PACKAGE_FORMAT, PACKAGE_FORMAT_VERSION, Revision,
    Sha256Digest, TileAddress, VerticalDatum, WritePort, blob_path, descriptor_path, manifest_path,
};

/// How a host declares one layer before its tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerSpec {
    /// Kind and source.
    pub key: LayerKey,
    /// The content version that the source states.
    pub revision: Revision,
    /// Horizontal datum.
    pub horizontal_datum: HorizontalDatum,
    /// Vertical datum of an elevation layer.
    pub vertical_datum: Option<VerticalDatum>,
    /// Encoding of the blobs.
    pub encoding: Encoding,
    /// Pixels on one side of a tile.
    pub tile_pixels: u32,
    /// Declared accuracy.
    pub accuracy: Accuracy,
    /// Zoom of the correlation cells.
    pub correlation_zoom: u32,
}

/// Builds one package on the Web Mercator grid of the WGS84 Earth. Each
/// tile goes to the store at once, by its digest. [`Self::seal`] writes the
/// tile indexes, then the manifest, then the descriptor.
#[derive(Clone, Debug, Default)]
pub struct PackageWriter {
    layers: BTreeMap<LayerKey, (LayerSpec, BTreeMap<IndexKey, IndexEntry>)>,
}

impl PackageWriter {
    /// An empty writer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares a layer.
    ///
    /// # Errors
    ///
    /// Refuses a repeated key.
    pub fn add_layer(&mut self, spec: LayerSpec) -> Result<(), DataError> {
        if self.layers.contains_key(&spec.key) {
            return Err(DataError::Invalid {
                field: "layer",
                detail: alloc::format!("{:?} twice", spec.key),
            });
        }
        self.layers.insert(spec.key, (spec, BTreeMap::new()));
        Ok(())
    }

    fn declare(
        &mut self,
        layer: LayerKey,
        address: TileAddress,
        entry: IndexEntry,
    ) -> Result<(), DataError> {
        let (_, entries) = self
            .layers
            .get_mut(&layer)
            .ok_or_else(|| DataError::Invalid {
                field: "layer",
                detail: alloc::format!("{layer:?}"),
            })?;
        if entries.insert(key(address), entry).is_some() {
            return Err(DataError::Invalid {
                field: "tile",
                detail: alloc::format!("{layer:?} {address:?} twice"),
            });
        }
        Ok(())
    }

    /// Stores the blob of one tile and declares it.
    ///
    /// # Errors
    ///
    /// Refuses an empty blob, an unknown layer, an address declared before,
    /// and a failed write.
    pub async fn add_tile<W: WritePort + ?Sized>(
        &mut self,
        store: &mut W,
        layer: LayerKey,
        address: TileAddress,
        bytes: &[u8],
    ) -> Result<Sha256Digest, DataError> {
        if bytes.is_empty() {
            return Err(DataError::Invalid {
                field: "tile",
                detail: String::from("empty blob"),
            });
        }
        let blob = Sha256Digest::of(bytes);
        self.declare(
            layer,
            address,
            IndexEntry::Tile {
                blob,
                bytes: bytes.len() as u64,
            },
        )?;
        store.write(&blob_path(blob), bytes).await?;
        Ok(blob)
    }

    /// Declares an address in the coverage that the source did not have.
    ///
    /// # Errors
    ///
    /// Refuses an unknown layer and an address declared before.
    pub fn add_gap(&mut self, layer: LayerKey, address: TileAddress) -> Result<(), DataError> {
        self.declare(layer, address, IndexEntry::Gap)
    }

    /// The manifest and the index blobs that [`Self::seal`] would write.
    ///
    /// # Errors
    ///
    /// Refuses a manifest that breaks a rule.
    pub fn manifest(&self) -> Result<(Manifest, Vec<Vec<u8>>), DataError> {
        let mut layers = Vec::with_capacity(self.layers.len());
        let mut blobs = Vec::with_capacity(self.layers.len());
        for (spec, entries) in self.layers.values() {
            let index = encode_index(entries);
            let tiles = entries
                .values()
                .filter(|e| matches!(e, IndexEntry::Tile { .. }))
                .count() as u64;
            layers.push(Layer {
                key: spec.key,
                revision: spec.revision,
                horizontal_datum: spec.horizontal_datum,
                vertical_datum: spec.vertical_datum,
                encoding: spec.encoding,
                tile_pixels: spec.tile_pixels,
                accuracy: spec.accuracy,
                correlation_zoom: spec.correlation_zoom,
                index: IndexRef {
                    blob: Sha256Digest::of(&index),
                    bytes: index.len() as u64,
                    tiles,
                    gaps: entries.len() as u64 - tiles,
                },
            });
            blobs.push(index);
        }
        let manifest = Manifest {
            format: String::from(PACKAGE_FORMAT),
            version: PACKAGE_FORMAT_VERSION,
            body: Body::Earth,
            ellipsoid: EllipsoidId::Wgs84,
            grid: GridScheme::WebMercatorXyz,
            layers,
        };
        manifest.validate()?;
        Ok((manifest, blobs))
    }

    /// Checks that the store holds each tile blob, writes the tile indexes,
    /// the manifest, and the descriptor. Returns the package identity.
    ///
    /// # Errors
    ///
    /// Refuses a manifest that breaks a rule, a missing tile blob, and a
    /// failed write.
    pub async fn seal<W: WritePort + ?Sized>(
        &self,
        store: &mut W,
        descriptor: &Descriptor,
    ) -> Result<Sha256Digest, DataError> {
        let (manifest, indexes) = self.manifest()?;
        for (_, entries) in self.layers.values() {
            for entry in entries.values() {
                if let IndexEntry::Tile { blob, .. } = entry {
                    let path = blob_path(*blob);
                    if !store.contains(&path).await? {
                        return Err(DataError::Missing { path });
                    }
                }
            }
        }
        for index in &indexes {
            store
                .write(&blob_path(Sha256Digest::of(index)), index)
                .await?;
        }
        let bytes = manifest.canonical_bytes()?;
        let identity = Sha256Digest::of(&bytes);
        store.write(&manifest_path(identity), &bytes).await?;
        let descriptor =
            serde_json::to_vec_pretty(descriptor).map_err(|error| DataError::Document {
                path: descriptor_path(identity),
                detail: alloc::format!("{error}"),
            })?;
        store.write(&descriptor_path(identity), &descriptor).await?;
        Ok(identity)
    }
}
