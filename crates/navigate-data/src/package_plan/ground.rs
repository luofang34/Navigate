//! Ground heights for the resource model, from coarse Terrarium tiles read
//! before the plan: the plan cannot say how high above the ground a leg
//! flies until it knows the ground.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use navigate_geodesy::float_math::floor;

use crate::corridor::corridor_tiles;
use crate::tile::{decode_terrarium_m, tile_xy};
use crate::{DataError, GroundHeight, TileAddress};

/// The zoom of the planning elevation: pixels of about 120 m.
pub const GROUND_ZOOM: u32 = 10;
/// Half width of the strip of planning tiles about the route, metres.
pub const GROUND_CORRIDOR_M: f64 = 2_000.0;

/// The planning tiles about the route through `points`, `[lat, lon]` in
/// degrees, at [`GROUND_ZOOM`].
///
/// # Errors
///
/// Refuses an empty route, a value that is not finite, and a corridor past
/// the Web Mercator latitudes.
pub fn ground_tiles(points: &[[f64; 2]]) -> Result<BTreeSet<TileAddress>, DataError> {
    let mut tiles = BTreeSet::new();
    corridor_tiles(points, GROUND_CORRIDOR_M, GROUND_ZOOM, &mut tiles)?;
    Ok(tiles)
}

/// Decoded Terrarium tiles at [`GROUND_ZOOM`]: heights above mean sea
/// level. A point on no decoded tile has no known ground, which the
/// resource model refuses.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TerrariumGround {
    /// By column and row: the side of the tile in pixels and its heights,
    /// row by row.
    tiles: BTreeMap<(u32, u32), (u32, Vec<f32>)>,
}

impl TerrariumGround {
    /// Takes one square Terrarium tile at `address` as its RGB8 pixels,
    /// row by row, `side` pixels a row. A second tile at the same address
    /// replaces the first.
    ///
    /// # Errors
    ///
    /// Refuses a tile at another zoom than [`GROUND_ZOOM`] and pixels that
    /// do not fill a square of `side`.
    pub fn insert_rgb(
        &mut self,
        address: TileAddress,
        side: u32,
        rgb: &[u8],
    ) -> Result<(), DataError> {
        if address.zoom != GROUND_ZOOM {
            return Err(DataError::Invalid {
                field: "planning tile",
                detail: alloc::format!("zoom {}, not {GROUND_ZOOM}", address.zoom),
            });
        }
        let pixels = (side as usize).saturating_mul(side as usize);
        if side == 0 || rgb.len() != pixels.saturating_mul(3) {
            return Err(DataError::Invalid {
                field: "planning tile",
                detail: alloc::format!(
                    "{} bytes for a side of {side} pixels at {}/{}/{}",
                    rgb.len(),
                    address.zoom,
                    address.x,
                    address.y
                ),
            });
        }
        let heights = rgb
            .as_chunks::<3>()
            .0
            .iter()
            .map(|pixel| decode_terrarium_m(*pixel) as f32)
            .collect();
        self.tiles.insert((address.x, address.y), (side, heights));
        Ok(())
    }

    /// Tiles decoded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tiles.len()
    }

    /// Whether no tile is decoded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tiles.is_empty()
    }
}

impl GroundHeight for TerrariumGround {
    /// The height of the pixel holding `point`.
    fn ground_m(&self, point: [f64; 2]) -> Option<f64> {
        let [x, y] = tile_xy(point, GROUND_ZOOM);
        let (column, row) = (floor(x), floor(y));
        let (side, heights) = self.tiles.get(&(column as u32, row as u32))?;
        let last = side.saturating_sub(1) as usize;
        let px = (((x - column) * f64::from(*side)) as usize).min(last);
        let py = (((y - row) * f64::from(*side)) as usize).min(last);
        let index = py.saturating_mul(*side as usize).saturating_add(px);
        heights.get(index).map(|height| f64::from(*height))
    }
}
