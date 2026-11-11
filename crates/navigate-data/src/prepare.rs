//! Preparing a package from a resource plan through a host fetch port.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::corridor::corridor_tiles;
use crate::resource::{PlannedLeg, ResourcePlan, great_circle_m};
use crate::{
    Accuracy, DataError, Encoding, FetchPort, Fetched, Layer, LayerKind, PackageWriter,
    Sha256Digest, TileAddress, TileKey, VerticalDatum, WritePort,
};

/// How the host wants one layer of the package declared.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerTemplate {
    /// Layer identifier in the package.
    pub id: String,
    /// The layer kind; the fetch port names its source.
    pub kind: LayerKind,
    /// Encoding of the blobs.
    pub encoding: Encoding,
    /// Pixels on one side of a tile.
    pub tile_pixels: u32,
    /// Vertical datum of an elevation layer.
    pub vertical_datum: Option<VerticalDatum>,
    /// Declared accuracy.
    pub accuracy: Accuracy,
}

/// The tiles that a resource plan selects, by layer kind.
pub type PlannedTiles = BTreeMap<LayerKind, BTreeSet<TileAddress>>;

/// What a prepared package holds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrepareSummary {
    /// The package identity.
    pub identity: Sha256Digest,
    /// Tiles in the package.
    pub tiles: usize,
    /// Addresses that the source did not have.
    pub gaps: usize,
    /// Bytes of the tile blobs.
    pub bytes: u64,
}

/// The tiles that each band of each leg needs. The strip of a band runs
/// from the start of the leg to `far_m` past its end, because the camera
/// looks ahead of the vehicle.
///
/// # Errors
///
/// Refuses a plan that does not match the legs, and a corridor that
/// [`corridor_tiles`] refuses.
pub fn plan_tiles(legs: &[PlannedLeg], plan: &ResourcePlan) -> Result<PlannedTiles, DataError> {
    if legs.len() != plan.legs.len() {
        return Err(DataError::Invalid {
            field: "plan",
            detail: alloc::format!("{} legs, {} planned", legs.len(), plan.legs.len()),
        });
    }
    let mut tiles = PlannedTiles::new();
    for (leg, resources) in legs.iter().zip(&plan.legs) {
        let from = [leg.from[0], leg.from[1]];
        let to = [leg.to[0], leg.to[1]];
        let length = great_circle_m(from, to).max(1.0);
        for band in &resources.bands {
            let end = crate::resource::point_along(from, to, 1.0 + band.far_m / length);
            let set = tiles.entry(band.layer).or_default();
            corridor_tiles(&[from, to, end], band.half_width_m.max(1.0), band.zoom, set)?;
        }
    }
    Ok(tiles)
}

/// Fetches each planned tile once through the host port and writes a
/// sealed package. A tile that the source does not have becomes a gap.
///
/// # Errors
///
/// Refuses more than `max_tiles` tiles before the first fetch, a layer
/// kind without a template or without a source in the port, a failed
/// fetch, and a failed write.
pub async fn prepare<F, W>(
    release_id: &str,
    tiles: &PlannedTiles,
    templates: &[LayerTemplate],
    max_tiles: usize,
    fetch: &F,
    store: &mut W,
) -> Result<PrepareSummary, DataError>
where
    F: FetchPort + ?Sized,
    W: WritePort + ?Sized,
{
    let count: usize = tiles.values().map(BTreeSet::len).sum();
    if count > max_tiles {
        return Err(DataError::Invalid {
            field: "max_tiles",
            detail: alloc::format!("{count} tiles needed, limit {max_tiles}"),
        });
    }
    let mut writer = PackageWriter::new(release_id);
    let mut summary = PrepareSummary {
        identity: Sha256Digest([0; 32]),
        tiles: 0,
        gaps: 0,
        bytes: 0,
    };
    for (&kind, addresses) in tiles {
        let template =
            templates
                .iter()
                .find(|t| t.kind == kind)
                .ok_or_else(|| DataError::Invalid {
                    field: "templates",
                    detail: alloc::format!("no template for {kind:?}"),
                })?;
        let source = fetch.source(kind).ok_or_else(|| DataError::Invalid {
            field: "fetch",
            detail: alloc::format!("the port has no {kind:?} source"),
        })?;
        writer.add_layer(Layer {
            id: template.id.clone(),
            kind,
            source: source.clone(),
            encoding: template.encoding,
            tile_pixels: template.tile_pixels,
            vertical_datum: template.vertical_datum.clone(),
            accuracy: template.accuracy.clone(),
            zooms: Vec::new(),
        })?;
        for &address in addresses {
            let key = TileKey {
                source: source.id.clone(),
                revision: source.revision.clone(),
                layer: kind,
                address,
            };
            match fetch
                .fetch(&key)
                .await
                .map_err(|message| DataError::Fetch {
                    key: alloc::format!("{key}"),
                    message,
                })? {
                Fetched::Tile(bytes) => {
                    writer
                        .add_tile(store, &template.id, address, &bytes)
                        .await?;
                    summary.tiles = summary.tiles.wrapping_add(1);
                    summary.bytes = summary.bytes.saturating_add(bytes.len() as u64);
                }
                Fetched::Absent => {
                    writer.add_gap(&template.id, address)?;
                    summary.gaps = summary.gaps.wrapping_add(1);
                }
            }
        }
    }
    summary.identity = writer.seal(store).await?;
    Ok(summary)
}

#[cfg(test)]
mod tests;
