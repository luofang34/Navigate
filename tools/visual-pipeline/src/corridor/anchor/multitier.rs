//! CLI loader and re-exports for `navigate_locate::corridor::multitier`.

pub use navigate_locate::corridor::multitier::{
    AnchorMatch, GlobalFallbackSeed, GlobalFallbackSeeds, MultiTierSearchConfig, SearchTierMatcher,
    SearchTierOutcome, build_search_prior_from_hint, descriptor_from_tokens_and_canvas,
    global_fallback_seeds_for_places, gray_mask_to_bools, ground_crop_to_canvas,
    load_candidate_tiles, load_candidate_tiles_geodetic, rank_candidate_regions,
    resolve_global_fallback_seeds, resolve_global_fallback_seeds_with_mask, resolve_search_tier,
    resolve_search_tier_from_prior,
};

use crate::PipelineError;
use burn::tensor::Device;
use navigate_match::{CorridorTileStore, Dinov2, Weights};
use navigate_place::{GRID, IndexManifest, Record, decode_shard};
use std::{fs, path::Path, sync::Arc};

const DINOV2_INPUT_PX: usize = GRID * 14;

/// Load reference place [`Record`] entries from a place-index directory, manifest JSON, or `.npls` shard.
///
/// # Errors
/// Returns [`PipelineError`] when reading or decoding index files fails.
pub fn load_place_records(place_index_path: &Path) -> Result<Vec<Record>, PipelineError> {
    if place_index_path.is_dir() {
        let manifest_path = if place_index_path.join("manifest.json").is_file() {
            place_index_path.join("manifest.json")
        } else {
            place_index_path.join("index.json")
        };
        return load_records_from_manifest(&manifest_path);
    }

    let bytes = fs::read(place_index_path).map_err(|source| PipelineError::Io {
        path: place_index_path.to_path_buf(),
        source,
    })?;
    if bytes.starts_with(b"NPLS") {
        let (_, _, records) = decode_shard(&place_index_path.display().to_string(), &bytes)
            .map_err(|e| PipelineError::Invalid(format!("place shard decode failed: {e}")))?;
        return Ok(records);
    }

    load_records_from_manifest(place_index_path)
}

fn load_records_from_manifest(manifest_path: &Path) -> Result<Vec<Record>, PipelineError> {
    let text = fs::read_to_string(manifest_path).map_err(|source| PipelineError::Io {
        path: manifest_path.to_path_buf(),
        source,
    })?;
    let manifest = IndexManifest::from_json(&text)
        .map_err(|e| PipelineError::Invalid(format!("place index manifest invalid: {e}")))?;
    let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let mut all = Vec::new();
    for scale in &manifest.scales {
        for shard in &scale.shards {
            let shard_path = root.join(&shard.path);
            let bytes = fs::read(&shard_path).map_err(|source| PipelineError::Io {
                path: shard_path.clone(),
                source,
            })?;
            let (_, _, mut records) = decode_shard(&shard.path, &bytes)
                .map_err(|e| PipelineError::Invalid(format!("place shard decode failed: {e}")))?;
            all.append(&mut records);
        }
    }
    Ok(all)
}

/// Load an optional [`SearchTierMatcher`] from `--dinov2-weights` and `--place-index` CLI paths.
///
/// # Errors
/// Returns [`PipelineError`] when configured weights or place-index shards fail to load.
pub fn load_search_tier_matcher(
    dinov2_weights_path: Option<&Path>,
    place_index_path: Option<&Path>,
    store: Arc<CorridorTileStore>,
    device: &Device,
) -> Result<Option<Arc<SearchTierMatcher>>, PipelineError> {
    if dinov2_weights_path.is_none() && place_index_path.is_none() {
        return Ok(None);
    }
    let mut matcher = SearchTierMatcher::new(MultiTierSearchConfig::default()).with_store(store);
    if let Some(w_path) = dinov2_weights_path {
        let bytes = std::fs::read(w_path).map_err(|source| PipelineError::Io {
            path: w_path.to_path_buf(),
            source,
        })?;
        let weights = Weights::from_safetensors(&bytes)?;
        let model = Dinov2::load(weights, DINOV2_INPUT_PX, device)?;
        matcher = matcher.with_dinov2(Arc::new(model));
    }
    if let Some(idx_path) = place_index_path {
        let records = load_place_records(idx_path)?;
        matcher = matcher.with_records(records);
    }
    Ok(Some(Arc::new(matcher)))
}
