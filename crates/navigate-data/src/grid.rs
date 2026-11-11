//! Grid schemes and grid cells.
//!
//! The grid is a named scheme with no anchor and no mission state. The one
//! scheme is the Web Mercator XYZ pyramid up to the latitude
//! [`MAX_TILE_LATITUDE_DEG`](crate::MAX_TILE_LATITUDE_DEG); a request past
//! it gets a typed refusal. A polar scheme and the schemes of other bodies
//! are reserved. A cell of the pyramid is not of constant size: a zoom 16
//! cell is approximately 611 m on a side at the equator, 460 m at 41°, and
//! 306 m at 60° latitude.

use serde::{Deserialize, Serialize};

use crate::{DataError, TileAddress, tile::ground_sample_distance_m};

/// The default zoom of the correlation cells of a layer.
pub const DEFAULT_CORRELATION_ZOOM: u32 = 16;

/// The tile scheme of a package.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GridScheme {
    /// The Web Mercator XYZ pyramid on the WGS84 Earth.
    WebMercatorXyz,
    /// A scheme that this crate does not know.
    #[serde(other)]
    Unsupported,
}

/// One cell of the pyramid at a stated zoom.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct GridCell(pub TileAddress);

impl GridCell {
    /// The cell at `zoom` that holds a point, `[lat, lon]` in degrees.
    ///
    /// # Errors
    ///
    /// Refuses a point that is not finite or that is past the latitude
    /// limit of the scheme.
    pub fn containing(point: [f64; 2], zoom: u32) -> Result<Self, DataError> {
        TileAddress::containing(point, zoom).map(Self)
    }

    /// The cell at `zoom` that holds a tile at that zoom or finer. A coarser
    /// tile spans more than one cell, so the result is `None`.
    #[must_use]
    pub const fn of_tile(tile: TileAddress, zoom: u32) -> Option<Self> {
        if tile.zoom < zoom {
            return None;
        }
        Some(Self(tile.ancestor(tile.zoom - zoom)))
    }

    /// The side of a cell at `zoom` and latitude `lat_deg`, metres.
    #[must_use]
    pub fn side_m(lat_deg: f64, zoom: u32) -> f64 {
        ground_sample_distance_m(lat_deg, zoom, 1)
    }
}

#[cfg(test)]
mod tests;
