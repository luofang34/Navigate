//! The fixed global grid.
//!
//! The grid is the XYZ tile pyramid itself, so it has no anchor and no
//! mission state. One level of the pyramid, [`CORRELATION_ZOOM`], defines
//! the correlation cells: map error is taken as shared inside one cell and
//! as independent between cells. Derived products tile on the same
//! pyramid, so two missions over the same ground name the same cells and
//! can share the same products.

use alloc::string::String;

use serde::{Deserialize, Serialize};

use crate::{DataError, TileAddress};

/// The zoom of the correlation cells. A cell is approximately 611 m on a
/// side at the equator and 460 m at 41° latitude.
pub const CORRELATION_ZOOM: u32 = 16;

/// The grid that a package and its products use.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GridSpec {
    /// Tile scheme: `web_mercator_xyz`.
    pub scheme: String,
    /// Zoom of the correlation cells.
    pub correlation_zoom: u32,
}

impl Default for GridSpec {
    fn default() -> Self {
        Self {
            scheme: String::from("web_mercator_xyz"),
            correlation_zoom: CORRELATION_ZOOM,
        }
    }
}

/// One correlation cell of the fixed grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GridCell(pub TileAddress);

impl GridCell {
    /// The cell that holds a point, `[lat, lon]` in degrees.
    ///
    /// # Errors
    ///
    /// Refuses a point that is not finite or that is past the Web Mercator
    /// latitude limit.
    pub fn containing(point: [f64; 2]) -> Result<Self, DataError> {
        TileAddress::containing(point, CORRELATION_ZOOM).map(Self)
    }

    /// The cell that holds a tile at the correlation zoom or finer. A
    /// coarser tile spans more than one cell, so the result is `None`.
    #[must_use]
    pub const fn of_tile(tile: TileAddress) -> Option<Self> {
        if tile.zoom < CORRELATION_ZOOM {
            return None;
        }
        Some(Self(tile.ancestor(tile.zoom - CORRELATION_ZOOM)))
    }

    /// The stable text identifier `z/x/y`.
    #[must_use]
    pub fn id(self) -> String {
        alloc::format!("{}/{}/{}", self.0.zoom, self.0.x, self.0.y)
    }
}

#[cfg(test)]
mod tests;
