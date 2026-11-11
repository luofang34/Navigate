//! The size of a plan from its tile count.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use crate::LayerKind;
use crate::prepare::PlannedTiles;
use crate::resource::{LayerLimits, ResourceRequest};

/// The size of the planned tiles of one layer at one zoom.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerSize {
    /// The layer.
    pub layer: LayerKind,
    /// The zoom.
    pub zoom: u32,
    /// Planned tiles.
    pub tiles: u64,
    /// Tiles × the typical tile size, bytes.
    pub estimate_bytes: u64,
    /// Tiles × the upper tile size, bytes.
    pub upper_bytes: u64,
}

/// The size of a plan.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SizeEstimate {
    /// For each layer and zoom, in layer then zoom order.
    pub layers: Vec<LayerSize>,
    /// Sum of the estimates, bytes.
    pub estimate_bytes: u64,
    /// Sum of the upper bounds, bytes.
    pub upper_bytes: u64,
}

impl SizeEstimate {
    /// Whether the plan fits a storage budget: judged on the upper bound.
    #[must_use]
    pub const fn fits(&self, budget_bytes: u64) -> bool {
        self.upper_bytes <= budget_bytes
    }
}

/// The size of one layer at one zoom: `tiles` × the tile sizes of `limits`.
#[must_use]
pub fn layer_size(layer: LayerKind, zoom: u32, tiles: u64, limits: &LayerLimits) -> LayerSize {
    LayerSize {
        layer,
        zoom,
        tiles,
        estimate_bytes: tiles.saturating_mul(limits.typical_tile_bytes),
        upper_bytes: tiles.saturating_mul(limits.upper_tile_bytes),
    }
}

/// The size of the planned tiles of a request (from
/// [`plan_tiles`](crate::plan_tiles)), with the tile sizes of the request
/// layers.
#[must_use]
pub fn estimate_size(request: &ResourceRequest, tiles: &PlannedTiles) -> SizeEstimate {
    let mut counts: BTreeMap<(LayerKind, u32), u64> = BTreeMap::new();
    for (&layer, addresses) in tiles {
        for address in addresses {
            let count = counts.entry((layer, address.zoom)).or_default();
            *count = count.saturating_add(1);
        }
    }
    let layers: Vec<LayerSize> = counts
        .into_iter()
        .map(|((layer, zoom), count)| {
            let limits = match layer {
                LayerKind::Elevation => &request.elevation,
                _ => &request.imagery,
            };
            layer_size(layer, zoom, count, limits)
        })
        .collect();
    SizeEstimate {
        estimate_bytes: layers
            .iter()
            .map(|l| l.estimate_bytes)
            .fold(0, u64::saturating_add),
        upper_bytes: layers
            .iter()
            .map(|l| l.upper_bytes)
            .fold(0, u64::saturating_add),
        layers,
    }
}
