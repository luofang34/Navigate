//! Fusion admission labels for map anchors.
//!
//! The navigation filter treats each position fix as independent. Anchors
//! from one map cell share the map error, and tracking poses depend on
//! earlier anchors. A [`MapCellLedger`] labels each anchor so that a fusion
//! adapter can refuse correlated evidence.

use crate::session::FrameKey;
use navigate_data::{CorrelationKey, DataError, Layer};
use std::collections::BTreeMap;

/// A map area that shares one error bias.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MapCell {
    /// A cell of a declared source: the correlation key of ADR-0022. It
    /// names the source of the imagery, its revision, and the grid cell at
    /// the correlation zoom of the layer. It does not name the package, so
    /// two packages cut from the same imagery give the same cell.
    Source(CorrelationKey),
    /// A cell of the local frame of one map that has no correlation key: a
    /// map that declares no source, for example a tile catalog, or a frame
    /// that the grid scheme does not cover, for example at a pole or on
    /// another body. Fixes from this one map in this cell share error. The
    /// same imagery in another map is not found.
    Local {
        /// Manifest digest of the map revision.
        map_manifest_sha256: String,
        /// Cell column in the local frame.
        east: i64,
        /// Cell row in the local frame.
        north: i64,
    },
}

impl MapCell {
    /// The cell of a point, `[lat, lon]` in degrees, on a package layer.
    ///
    /// # Errors
    /// Refuses a point outside the grid scheme.
    pub fn of_layer(layer: &Layer, point: [f64; 2]) -> Result<Self, DataError> {
        layer.correlation_key(point).map(Self::Source)
    }

    /// The local cell of side `side_m` that holds the position
    /// (`east_m`, `north_m`) in the local frame of the map.
    #[must_use]
    pub fn local(map_manifest_sha256: &str, east_m: f64, north_m: f64, side_m: f64) -> Self {
        Self::Local {
            map_manifest_sha256: map_manifest_sha256.to_owned(),
            east: (east_m / side_m).floor() as i64,
            north: (north_m / side_m).floor() as i64,
        }
    }

    /// The column and row of a local cell.
    #[must_use]
    pub const fn local_index(&self) -> Option<[i64; 2]> {
        match self {
            Self::Local { east, north, .. } => Some([*east, *north]),
            Self::Source(_) => None,
        }
    }
}

impl From<CorrelationKey> for MapCell {
    fn from(key: CorrelationKey) -> Self {
        Self::Source(key)
    }
}

/// Whether an anchor may enter fusion as an independent position fix.
#[derive(Clone, Debug, PartialEq)]
pub enum FusionEligibility {
    /// The first anchor from this map cell. Its map error is not in an earlier fix.
    Independent,
    /// The ledger reached its memory bound. Independence is not known.
    Untracked,
    /// An earlier anchor already carries the map error of this cell.
    SharedMapError {
        /// The cell.
        cell: MapCell,
        /// Frame of the earlier anchor.
        earlier: FrameKey,
    },
}

/// The first admitted anchor of each map cell, up to a memory bound.
///
/// [`Self::label`] reads and [`Self::commit`] writes. A caller that offers a
/// fix to a filter commits only when the filter admits the fix. A refused
/// fix then leaves its cell open for the next frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MapCellLedger {
    first: BTreeMap<MapCell, FrameKey>,
    capacity: usize,
}

impl MapCellLedger {
    /// A ledger that tracks at most `capacity` cells.
    #[must_use]
    pub const fn new(capacity: usize) -> Self {
        Self {
            first: BTreeMap::new(),
            capacity,
        }
    }

    /// The label of an anchor of `frame` in `cell`. This does not change
    /// the ledger.
    #[must_use]
    pub fn label(&self, cell: &MapCell, frame: FrameKey) -> FusionEligibility {
        match self.first.get(cell) {
            Some(earlier) if *earlier != frame => FusionEligibility::SharedMapError {
                cell: cell.clone(),
                earlier: *earlier,
            },
            Some(_) => FusionEligibility::Independent,
            None if self.first.len() >= self.capacity => FusionEligibility::Untracked,
            None => FusionEligibility::Independent,
        }
    }

    /// Records `frame` as the first anchor of `cell`. A cell that has an
    /// anchor keeps it, and a full ledger takes no new cell.
    pub fn commit(&mut self, cell: MapCell, frame: FrameKey) {
        if self.first.len() < self.capacity || self.first.contains_key(&cell) {
            self.first.entry(cell).or_insert(frame);
        }
    }

    /// The count of cells with an anchor.
    #[must_use]
    pub fn len(&self) -> usize {
        self.first.len()
    }

    /// Whether no cell has an anchor.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.first.is_empty()
    }
}

#[cfg(test)]
mod tests;
