//! Re-exports of multi-tier global place fallback and physical plausibility checks.

pub use navigate_locate::corridor::plausibility;

/// Multi-tier search types and CLI place-shard loaders.
pub mod multitier {
    pub use navigate_locate::corridor::multitier::*;

    use crate::PipelineError;
    use burn::tensor::Device;
    use navigate_locate::place::Record;
    use navigate_match::CorridorTileStore;
    use std::{path::Path, sync::Arc};

    /// Load reference place [`Record`] entries from a directory, manifest JSON, or `.npls` shard.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when reading or decoding index files fails.
    pub fn load_place_records(place_index_path: &Path) -> Result<Vec<Record>, PipelineError> {
        Ok(navigate_visual_flight::load_place_records(
            place_index_path,
        )?)
    }

    /// Load an optional [`SearchTierMatcher`] from `--dinov2-weights` and `--place-index` paths.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError`] when configured weights or place-index shards fail to load.
    pub fn load_search_tier_matcher(
        dinov2_weights_path: Option<&Path>,
        place_index_path: Option<&Path>,
        store: Arc<CorridorTileStore>,
        device: &Device,
    ) -> Result<Option<Arc<SearchTierMatcher>>, PipelineError> {
        Ok(navigate_visual_flight::load_search_tier_matcher(
            dinov2_weights_path,
            place_index_path,
            store,
            device,
        )?)
    }
}
