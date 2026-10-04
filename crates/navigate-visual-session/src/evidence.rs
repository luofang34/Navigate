//! Fusion admission labels for map anchors.
//!
//! The navigation filter treats each position fix as independent. Anchors
//! from one map cell share the map error, and tracking poses depend on
//! earlier anchors. The session labels each anchor so that a fusion adapter
//! can refuse correlated evidence.

use crate::FrameKey;
use std::collections::BTreeMap;

/// A map area that shares one error bias.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MapCell {
    /// Manifest digest of the map revision.
    pub map_manifest_sha256: String,
    /// Cell column in the session local frame.
    pub east: i64,
    /// Cell row in the session local frame.
    pub north: i64,
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

#[derive(Debug, Default)]
pub(crate) struct Ledger {
    first: BTreeMap<MapCell, FrameKey>,
}

impl Ledger {
    pub fn label(&mut self, cell: MapCell, frame: FrameKey, capacity: usize) -> FusionEligibility {
        match self.first.get(&cell) {
            Some(earlier) if *earlier != frame => FusionEligibility::SharedMapError {
                cell,
                earlier: *earlier,
            },
            Some(_) => FusionEligibility::Independent,
            None if self.first.len() >= capacity => FusionEligibility::Untracked,
            None => {
                self.first.insert(cell, frame);
                FusionEligibility::Independent
            }
        }
    }

    pub fn len(&self) -> usize {
        self.first.len()
    }
}
