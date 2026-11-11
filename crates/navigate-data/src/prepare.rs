//! Preparing a package from a resource plan through a host fetch port.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::corridor::corridor_tiles;
use crate::resource::{PlannedLeg, ResourcePlan, great_circle_m, point_along};
use crate::{
    DataError, Descriptor, FetchPort, Fetched, LayerKind, LayerSpec, PackageWriter, Sha256Digest,
    TileAddress, TileKey, WritePort,
};

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
            let end = point_along(from, to, 1.0 + band.far_m / length);
            let set = tiles.entry(band.layer).or_default();
            corridor_tiles(&[from, to, end], band.half_width_m.max(1.0), band.zoom, set)?;
        }
    }
    Ok(tiles)
}

/// Fetches each planned tile once through the host port and writes a
/// sealed package. `layers` declares each layer kind of the plan with its
/// source and revision. A tile that the source does not have becomes a
/// gap.
///
/// # Errors
///
/// Refuses more than `max_tiles` tiles before the first fetch, a planned
/// layer kind without a spec, a failed fetch, and a failed write.
pub async fn prepare<F, W>(
    tiles: &PlannedTiles,
    layers: &[LayerSpec],
    descriptor: &Descriptor,
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
    let mut writer = PackageWriter::new();
    let mut specs = Vec::new();
    for &kind in tiles.keys() {
        let spec =
            layers
                .iter()
                .find(|s| s.key.kind == kind)
                .ok_or_else(|| DataError::Invalid {
                    field: "layers",
                    detail: alloc::format!("no layer spec for {kind:?}"),
                })?;
        writer.add_layer(*spec)?;
        specs.push(*spec);
    }
    let mut summary = PrepareSummary {
        identity: Sha256Digest([0; 32]),
        tiles: 0,
        gaps: 0,
        bytes: 0,
    };
    for spec in &specs {
        for &address in tiles.get(&spec.key.kind).into_iter().flatten() {
            let key = TileKey {
                source: spec.key.source,
                revision: spec.revision,
                layer: spec.key.kind,
                address,
            };
            let fetched = fetch
                .fetch(&key)
                .await
                .map_err(|message| DataError::Fetch {
                    key: alloc::format!("{key}"),
                    message,
                })?;
            match fetched {
                Fetched::Tile(bytes) => {
                    writer.add_tile(store, spec.key, address, &bytes).await?;
                    summary.tiles = summary.tiles.wrapping_add(1);
                    summary.bytes = summary.bytes.saturating_add(bytes.len() as u64);
                }
                Fetched::Absent => {
                    writer.add_gap(spec.key, address)?;
                    summary.gaps = summary.gaps.wrapping_add(1);
                }
            }
        }
    }
    summary.identity = writer.seal(store, descriptor).await?;
    Ok(summary)
}

#[cfg(test)]
mod tests;
