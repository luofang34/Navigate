//! Ground heights for the resource model: the tool fetches and decodes the
//! planning tiles that `navigate-data` names.

use crate::PipelineError;
use navigate_data::package_plan::{TerrariumGround, ground_tiles};
use navigate_data::{FetchPort, Fetched, LayerKind, LayerSpec, TileAddress, TileKey};

/// Fetches and decodes the planning tiles around `points`.
///
/// # Errors
///
/// Returns [`PipelineError`] when the corridor is unusable, a fetch fails,
/// or a tile does not decode.
pub fn fetch_ground_blocking<F: FetchPort>(
    fetch: &F,
    elevation: &LayerSpec,
    points: &[[f64; 2]],
) -> Result<TerrariumGround, PipelineError> {
    if elevation.key.kind != LayerKind::Elevation {
        return Err(PipelineError::Invalid(
            "the planning layer is not elevation".into(),
        ));
    }
    let mut ground = TerrariumGround::default();
    for address in ground_tiles(points)? {
        let key = TileKey {
            source: elevation.key.source,
            revision: elevation.revision,
            layer: LayerKind::Elevation,
            address,
        };
        let fetched = pollster::block_on(fetch.fetch(&key)).map_err(PipelineError::Invalid)?;
        if let Fetched::Tile(bytes) = fetched {
            insert_png(&mut ground, address, &bytes)?;
        }
    }
    Ok(ground)
}

/// Decodes one Terrarium PNG into `ground`.
///
/// # Errors
///
/// Returns [`PipelineError::Invalid`] for bytes that do not decode, and a
/// tile that is not square or not at the planning zoom.
pub fn insert_png(
    ground: &mut TerrariumGround,
    address: TileAddress,
    bytes: &[u8],
) -> Result<(), PipelineError> {
    let image = image::load_from_memory(bytes)
        .map_err(|e| PipelineError::Invalid(format!("elevation tile {address:?}: {e}")))?
        .to_rgb8();
    if image.width() != image.height() {
        return Err(PipelineError::Invalid(format!(
            "elevation tile {address:?}: {} by {} pixels",
            image.width(),
            image.height()
        )));
    }
    ground.insert_rgb(address, image.width(), image.as_raw())?;
    Ok(())
}
