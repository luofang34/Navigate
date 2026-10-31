//! Lookahead tile synthesis and preloading for corridor tracker.

use super::super::tile_builder::{
    BackgroundTileBuilder, TileSynthParams, synthesize_corridor_tiles,
};
use crate::{
    PipelineError,
    surface::{Surface, SurfaceGpuRenderer},
    xfeat_runtime::LoadedXFeat,
};
use navigate_match::{CorridorTileStore, HadamardProjector};
use std::{sync::Mutex, time::Instant};

/// Ensure corridor tiles are indexed or synthesized for lookahead position.
///
/// # Errors
/// Returns [`PipelineError`] if tile synthesis fails.
#[expect(
    clippy::too_many_arguments,
    reason = "lookahead tile preloading requires terrain, tile store, builders, and inference state"
)]
pub(crate) fn ensure_lookahead_tiles(
    surface: &Surface,
    store: &CorridorTileStore,
    tile_builder: &mut Option<BackgroundTileBuilder>,
    sync_gpu_renderer: &Mutex<Option<SurfaceGpuRenderer>>,
    xfeat: &LoadedXFeat,
    projector: &HadamardProjector,
    tile_params: TileSynthParams,
    lookahead: [f64; 2],
    tile_build_ms: &mut Vec<f64>,
    sync_tiles_built: &mut usize,
) -> Result<(), PipelineError> {
    if let Some(tb) = tile_builder.as_mut() {
        tb.poll(tile_build_ms)?;
        tb.request_neighborhood(surface, store, lookahead)?;
    } else {
        let ck = store.tile_key_at(lookahead[0], lookahead[1]);
        if !store.has_tile(ck) {
            let t_tb = Instant::now();
            let entries = synthesize_corridor_tiles(
                surface,
                Some(sync_gpu_renderer),
                xfeat,
                projector,
                ck,
                tile_params,
            )?;
            if !entries.is_empty() {
                for entry in entries {
                    store.insert_tile(entry, true)?;
                    *sync_tiles_built += 1;
                }
                tile_build_ms.push(t_tb.elapsed().as_secs_f64() * 1000.0);
            }
        }
    }
    Ok(())
}
