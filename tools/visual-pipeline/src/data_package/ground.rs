//! Ground heights for the resource model, from coarse Terrarium tiles that
//! the tool fetches before it plans.

use crate::PipelineError;
use navigate_data::tile::{decode_terrarium_m, tile_xy};
use navigate_data::{FetchPort, Fetched, GroundHeight, LayerKind, TileAddress, TileKey};
use std::collections::{BTreeMap, BTreeSet};

/// The zoom of the planning elevation: approximately 120 m pixels.
pub const GROUND_ZOOM: u32 = 10;

/// Decoded Terrarium tiles at [`GROUND_ZOOM`].
#[derive(Clone, Debug, Default)]
pub struct TerrariumGround {
    tiles: BTreeMap<(u32, u32), (u32, Vec<f32>)>,
}

impl TerrariumGround {
    /// Fetches and decodes the planning tiles around `points`.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when the corridor is unusable, a fetch
    /// fails, or a tile does not decode.
    pub fn fetch_blocking<F: FetchPort>(
        fetch: &F,
        points: &[[f64; 2]],
    ) -> Result<Self, PipelineError> {
        let source = fetch
            .source(LayerKind::Elevation)
            .ok_or_else(|| PipelineError::Invalid("the port has no elevation source".into()))?;
        let mut wanted = BTreeSet::new();
        navigate_data::corridor::corridor_tiles(points, 2_000.0, GROUND_ZOOM, &mut wanted)?;
        let mut ground = Self::default();
        for address in wanted {
            let key = TileKey {
                source: source.id.clone(),
                revision: source.revision.clone(),
                layer: LayerKind::Elevation,
                address,
            };
            let fetched = pollster::block_on(fetch.fetch(&key)).map_err(PipelineError::Invalid)?;
            if let Fetched::Tile(bytes) = fetched {
                ground.insert(address, &bytes)?;
            }
        }
        Ok(ground)
    }

    /// Decodes one Terrarium PNG.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::Invalid`] for bytes that do not decode.
    pub fn insert(&mut self, address: TileAddress, bytes: &[u8]) -> Result<(), PipelineError> {
        let image = image::load_from_memory(bytes)
            .map_err(|e| PipelineError::Invalid(format!("elevation tile {address:?}: {e}")))?
            .to_rgb8();
        let heights = image
            .pixels()
            .map(|p| decode_terrarium_m(p.0) as f32)
            .collect();
        self.tiles
            .insert((address.x, address.y), (image.width(), heights));
        Ok(())
    }
}

impl GroundHeight for TerrariumGround {
    fn ground_m(&self, point: [f64; 2]) -> Option<f64> {
        let [x, y] = tile_xy(point, GROUND_ZOOM);
        let (tx, ty) = (x.floor() as u32, y.floor() as u32);
        let (side, heights) = self.tiles.get(&(tx, ty))?;
        let px = ((x - x.floor()) * f64::from(*side)) as usize;
        let py = ((y - y.floor()) * f64::from(*side)) as usize;
        let index = py.min(*side as usize - 1) * *side as usize + px.min(*side as usize - 1);
        heights.get(index).map(|h| f64::from(*h))
    }
}
