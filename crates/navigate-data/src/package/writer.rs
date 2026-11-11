//! Writing a package: blobs first, the manifest last.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;

use crate::{
    DataError, GridSpec, Layer, LayerZoom, Manifest, PACKAGE_FORMAT, PACKAGE_FORMAT_VERSION,
    Sha256Digest, TileAddress, TileRef, WritePort, blob_path, manifest_path,
};

/// Tiles and gaps of one zoom level, keyed by `(y, x)`.
type ZoomCoverage = (BTreeMap<(u32, u32), TileRef>, BTreeSet<(u32, u32)>);
type Coverage = BTreeMap<u32, ZoomCoverage>;

/// Builds one package. Each tile goes to the store at once, by its digest.
/// [`Self::seal`] writes the manifest last, so a store without the manifest
/// holds no package, only blobs.
#[derive(Clone, Debug)]
pub struct PackageWriter {
    release_id: String,
    layers: BTreeMap<String, (Layer, Coverage)>,
}

impl PackageWriter {
    /// A writer for the release `release_id`.
    #[must_use]
    pub fn new(release_id: &str) -> Self {
        Self {
            release_id: String::from(release_id),
            layers: BTreeMap::new(),
        }
    }

    /// Declares a layer. Its `zooms` must be empty: the writer fills them.
    ///
    /// # Errors
    ///
    /// Refuses a repeated identifier, a layer with zooms, and a layer that
    /// [`Layer::validate`] refuses.
    pub fn add_layer(&mut self, layer: Layer) -> Result<(), DataError> {
        if !layer.zooms.is_empty() || self.layers.contains_key(&layer.id) {
            return Err(DataError::Invalid {
                field: "layer",
                detail: layer.id,
            });
        }
        layer.validate()?;
        self.layers
            .insert(layer.id.clone(), (layer, Coverage::new()));
        Ok(())
    }

    fn coverage(
        &mut self,
        layer_id: &str,
        address: TileAddress,
    ) -> Result<&mut ZoomCoverage, DataError> {
        let (_, coverage) = self
            .layers
            .get_mut(layer_id)
            .ok_or_else(|| DataError::Invalid {
                field: "layer",
                detail: String::from(layer_id),
            })?;
        let zoom = coverage.entry(address.zoom).or_default();
        if zoom.0.contains_key(&(address.y, address.x)) || zoom.1.contains(&(address.y, address.x))
        {
            return Err(DataError::Invalid {
                field: "tile",
                detail: alloc::format!("{layer_id} {address:?} declared twice"),
            });
        }
        Ok(zoom)
    }

    /// Stores the blob of one tile and declares it.
    ///
    /// # Errors
    ///
    /// Refuses an unknown layer, an address declared before, and a failed
    /// write.
    pub async fn add_tile<W: WritePort + ?Sized>(
        &mut self,
        store: &mut W,
        layer_id: &str,
        address: TileAddress,
        bytes: &[u8],
    ) -> Result<Sha256Digest, DataError> {
        let blob = Sha256Digest::of(bytes);
        let tile = TileRef {
            x: address.x,
            y: address.y,
            blob,
            bytes: bytes.len() as u64,
        };
        self.coverage(layer_id, address)?
            .0
            .insert((address.y, address.x), tile);
        store.write(&blob_path(blob), bytes).await?;
        Ok(blob)
    }

    /// Declares an address in the coverage that the source did not have.
    ///
    /// # Errors
    ///
    /// Refuses an unknown layer and an address declared before.
    pub fn add_gap(&mut self, layer_id: &str, address: TileAddress) -> Result<(), DataError> {
        self.coverage(layer_id, address)?
            .1
            .insert((address.y, address.x));
        Ok(())
    }

    /// The manifest that [`Self::seal`] would write.
    ///
    /// # Errors
    ///
    /// Refuses a manifest that breaks the rules.
    pub fn manifest(&self) -> Result<Manifest, DataError> {
        let layers = self
            .layers
            .values()
            .map(|(layer, coverage)| Layer {
                zooms: coverage
                    .iter()
                    .map(|(&zoom, (tiles, gaps))| LayerZoom {
                        zoom,
                        tiles: tiles.values().copied().collect(),
                        gaps: gaps.iter().map(|&(y, x)| [x, y]).collect::<Vec<_>>(),
                    })
                    .collect(),
                ..layer.clone()
            })
            .collect();
        let manifest = Manifest {
            format: String::from(PACKAGE_FORMAT),
            version: PACKAGE_FORMAT_VERSION,
            release_id: self.release_id.clone(),
            grid: GridSpec::default(),
            layers,
        };
        manifest.validate()?;
        Ok(manifest)
    }

    /// Checks that the store holds each blob, then writes the manifest.
    /// Returns the package identity.
    ///
    /// # Errors
    ///
    /// Refuses a manifest that breaks the rules, a blob that the store does
    /// not hold, and a failed write.
    pub async fn seal<W: WritePort + ?Sized>(
        &self,
        store: &mut W,
    ) -> Result<Sha256Digest, DataError> {
        let manifest = self.manifest()?;
        for (blob, _) in manifest.blobs() {
            let path = blob_path(blob);
            if !store.contains(&path).await? {
                return Err(DataError::Missing { path });
            }
        }
        let bytes = manifest.canonical_bytes()?;
        let identity = Sha256Digest::of(&bytes);
        store.write(&manifest_path(identity), &bytes).await?;
        Ok(identity)
    }
}
