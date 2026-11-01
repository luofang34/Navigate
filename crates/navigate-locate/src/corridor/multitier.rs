//! Multi-tier candidate retrieval and fail-closed `Dinov2` place recognition bridge.

mod seeds;
pub use seeds::{
    GlobalFallbackSeed, GlobalFallbackSeeds, build_search_prior_from_hint,
    global_fallback_seeds_for_places, resolve_global_fallback_seeds,
    resolve_global_fallback_seeds_with_mask,
};

use super::plausibility::CorridorTerrain;
use image::{DynamicImage, GrayImage};
use navigate_match::{CorridorTileKey, CorridorTileStore, DINOV2_WIDTH, Dinov2, MatchError};
use navigate_place::{
    CANVAS_PX, Canvas, Descriptor, Ranked, Record, Tokens, cell_weights, describe, rank,
};
use navigate_visual::{
    LocalFrame, SearchConfig, SearchPrior, SearchTier, holdout_pnp::PoseCandidate,
};
use std::sync::Arc;

/// Outcome of multi-tier anchor search evaluation.
#[derive(Clone, Debug, PartialEq)]
pub enum SearchTierOutcome {
    /// Solved at the local tier around the predicted pose.
    LocalSolved(PoseCandidate),
    /// Solved at the region tier across covariance grid candidates.
    RegionSolved(PoseCandidate),
    /// Solved at the global tier through `Dinov2` place recognition retrieval.
    GlobalSolved(PoseCandidate),
    /// Global retrieval fallback is required.
    GlobalFallbackNeeded {
        /// Rough prior center coordinate when available.
        rough_prior: Option<[f64; 3]>,
    },
}

/// Resolve the active [`SearchTier`] from an optional uncertainty radius in meters.
#[must_use]
pub fn resolve_search_tier(
    prior_radius_m: Option<f64>,
    local_limit_m: f64,
    region_limit_m: f64,
) -> SearchTier {
    match prior_radius_m {
        None => SearchTier::Global,
        Some(r) if r <= local_limit_m => SearchTier::Local,
        Some(r) if r <= region_limit_m => SearchTier::Region,
        Some(_) => SearchTier::Global,
    }
}

/// Resolve the active [`SearchTier`] directly from a [`SearchPrior`].
#[must_use]
pub fn resolve_search_tier_from_prior(prior: &SearchPrior, config: &SearchConfig) -> SearchTier {
    prior.tier(config)
}

/// Configuration parameters for multi-tier search coordination.
#[derive(Clone, Copy, Debug)]
pub struct MultiTierSearchConfig {
    /// Search configuration for covariance bounding.
    pub search_config: SearchConfig,
    /// Number of top candidate places to retrieve from global search.
    pub top_k_global: usize,
    /// Minimum spatial separation in meters between distinct candidate places.
    pub global_spacing_m: f64,
}

impl Default for MultiTierSearchConfig {
    fn default() -> Self {
        Self {
            search_config: SearchConfig::default(),
            top_k_global: 5,
            global_spacing_m: 100.0,
        }
    }
}

/// Convert a binary grayscale ground mask (`> 0` = valid ground) into a boolean vector.
#[must_use]
pub fn gray_mask_to_bools(mask: &GrayImage) -> Vec<bool> {
    mask.as_raw().iter().map(|&v| v > 0).collect()
}

/// Convert an image and optional ground validity mask into a normalized [`Canvas`].
#[must_use]
pub fn ground_crop_to_canvas(image: &DynamicImage, mask: Option<&[bool]>) -> Canvas {
    let rgb = image.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let mut canvas = Canvas::empty();
    if w == 0 || h == 0 {
        return canvas;
    }
    for cy in 0..CANVAS_PX {
        let sy = (cy * h) / CANVAS_PX;
        for cx in 0..CANVAS_PX {
            let sx = (cx * w) / CANVAS_PX;
            let src_idx = sy * w + sx;
            let valid = mask.is_none_or(|m| src_idx < m.len() && m[src_idx]);
            let pixel = rgb.get_pixel(sx as u32, sy as u32);
            let idx = cy * CANVAS_PX + cx;
            canvas.rgb[idx] = [pixel[0] as f32, pixel[1] as f32, pixel[2] as f32];
            canvas.valid[idx] = valid;
        }
    }
    canvas
}

/// Pool a 384-dimensional unit descriptor from patch tokens and a ground-masked canvas.
///
/// # Errors
/// Returns [`navigate_place::PlaceError`] when token dimensions do not match [`Canvas`].
pub fn descriptor_from_tokens_and_canvas(
    tokens: &Tokens,
    canvas: &Canvas,
) -> Result<Descriptor, navigate_place::PlaceError> {
    let weights = cell_weights(canvas);
    describe(tokens, &weights, None)
}

/// Rank top-K candidate geographic places by cosine similarity and minimum spacing.
#[must_use]
pub fn rank_candidate_regions(
    query_descriptor: &Descriptor,
    records: &[Record],
    top_k: usize,
    spacing_m: f64,
) -> Vec<Ranked> {
    rank(
        std::slice::from_ref(query_descriptor),
        records.iter(),
        |_| true,
        top_k,
        spacing_m,
    )
}

/// Resolve [`CorridorTileKey`] entries from `ENU` coordinates stored in `ranked_places`.
#[must_use]
pub fn load_candidate_tiles(
    store: &CorridorTileStore,
    ranked_places: &[Ranked],
) -> Vec<CorridorTileKey> {
    let mut keys = Vec::with_capacity(ranked_places.len());
    for place in ranked_places {
        let key = store.tile_key_at(place.centre[0], place.centre[1]);
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    keys
}

/// Project geodetic `[lat_deg, lon_deg]` place centers into local `ENU` and resolve tile keys.
#[must_use]
pub fn load_candidate_tiles_geodetic<S: CorridorTerrain + ?Sized>(
    store: &CorridorTileStore,
    terrain: &S,
    ranked_places: &[Ranked],
) -> Vec<CorridorTileKey> {
    let mut keys = Vec::with_capacity(ranked_places.len());
    for place in ranked_places {
        let enu = terrain.local(place.centre[0], place.centre[1], 0.0);
        let key = store.tile_key_at(enu[0], enu[1]);
        if !keys.contains(&key) {
            keys.push(key);
        }
    }
    keys
}

/// Match outcome and candidate resolution across search tiers.
#[derive(Clone, Debug, PartialEq)]
pub struct AnchorMatch {
    /// Active search tier determined from prior covariance.
    pub tier: SearchTier,
    /// Candidate tile keys resolved in [`CorridorTileStore`].
    pub candidate_tiles: Vec<CorridorTileKey>,
    /// Ranked candidate regions from place recognition.
    pub ranked_regions: Vec<Ranked>,
    /// Extracted place descriptor when global retrieval ran.
    pub query_descriptor: Option<Descriptor>,
}

impl AnchorMatch {
    fn empty_global() -> Self {
        Self {
            tier: SearchTier::Global,
            candidate_tiles: Vec::new(),
            ranked_regions: Vec::new(),
            query_descriptor: None,
        }
    }
}

impl PartialEq<SearchTier> for AnchorMatch {
    fn eq(&self, other: &SearchTier) -> bool {
        self.tier == *other
    }
}

impl PartialEq<AnchorMatch> for SearchTier {
    fn eq(&self, other: &AnchorMatch) -> bool {
        *self == other.tier
    }
}

/// Coordinator for multi-tier candidate search and fail-closed `Dinov2` place retrieval.
#[derive(Clone)]
pub struct SearchTierMatcher {
    /// Multi-tier search configuration.
    pub config: MultiTierSearchConfig,
    /// Reference place records (`[lat_deg, lon_deg]` centers) for global retrieval.
    pub records: Vec<Record>,
    /// Optional [`CorridorTileStore`] for resolving tile keys from ranked places.
    pub store: Option<Arc<CorridorTileStore>>,
    /// Optional `Dinov2` encoder for 384-D patch token extraction.
    pub dinov2: Option<Arc<Dinov2>>,
    /// Optional local frame used to project geodetic place centers into `ENU` meters.
    pub local_frame: Option<LocalFrame>,
}

impl std::fmt::Debug for SearchTierMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchTierMatcher")
            .field("config", &self.config)
            .field("records_len", &self.records.len())
            .field("has_store", &self.store.is_some())
            .field("has_dinov2", &self.dinov2.is_some())
            .field("has_local_frame", &self.local_frame.is_some())
            .finish()
    }
}

impl SearchTierMatcher {
    /// Create an unconfigured search-tier matcher.
    #[must_use]
    pub fn new(config: MultiTierSearchConfig) -> Self {
        Self {
            config,
            records: Vec::new(),
            store: None,
            dinov2: None,
            local_frame: None,
        }
    }

    /// Attach reference place records for global place recognition.
    #[must_use]
    pub fn with_records(mut self, records: Vec<Record>) -> Self {
        self.records = records;
        self
    }

    /// Attach a [`CorridorTileStore`] for candidate tile resolution.
    #[must_use]
    pub fn with_store(mut self, store: Arc<CorridorTileStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Attach a [`Dinov2`] model for query descriptor extraction.
    #[must_use]
    pub fn with_dinov2(mut self, dinov2: Arc<Dinov2>) -> Self {
        self.dinov2 = Some(dinov2);
        self
    }

    /// Attach a [`LocalFrame`] to project geodetic `[lat_deg, lon_deg]` centers into `ENU` meters.
    #[must_use]
    pub fn with_local_frame(mut self, local_frame: LocalFrame) -> Self {
        self.local_frame = Some(local_frame);
        self
    }

    /// Return `true` when both `Dinov2` weights and place records are configured.
    #[must_use]
    pub fn has_global_retrieval(&self) -> bool {
        self.dinov2.is_some() && !self.records.is_empty()
    }

    /// Determine search tier and resolve candidate tiles synchronously.
    ///
    /// # Errors
    /// Returns [`MatchError`] if `Dinov2` inference or descriptor pooling fails.
    pub fn execute(
        &self,
        prior: &SearchPrior,
        masked_crop: Option<&DynamicImage>,
    ) -> Result<AnchorMatch, MatchError> {
        self.execute_with_context(
            prior,
            masked_crop,
            self.store.as_deref(),
            Some(&self.records),
        )
    }

    /// Determine search tier and resolve candidates synchronously with explicit store and records.
    ///
    /// # Errors
    /// Returns [`MatchError`] if `Dinov2` inference or descriptor pooling fails.
    pub fn execute_with_context(
        &self,
        prior: &SearchPrior,
        masked_crop: Option<&DynamicImage>,
        store: Option<&CorridorTileStore>,
        records: Option<&[Record]>,
    ) -> Result<AnchorMatch, MatchError> {
        self.execute_with_mask(prior, masked_crop, None, store, records)
    }

    /// Determine search tier and resolve candidates synchronously with an optional ground mask.
    ///
    /// # Errors
    /// Returns [`MatchError`] if `Dinov2` inference or descriptor pooling fails.
    pub fn execute_with_mask(
        &self,
        prior: &SearchPrior,
        masked_crop: Option<&DynamicImage>,
        ground_mask: Option<&[bool]>,
        store: Option<&CorridorTileStore>,
        records: Option<&[Record]>,
    ) -> Result<AnchorMatch, MatchError> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            pollster::block_on(self.execute_with_mask_async(
                prior,
                masked_crop,
                ground_mask,
                store,
                records,
            ))
        }
        #[cfg(target_arch = "wasm32")]
        {
            use futures_util::FutureExt;
            self.execute_with_mask_async(prior, masked_crop, ground_mask, store, records)
                .now_or_never()
                .ok_or(MatchError::Output(
                    "async Dinov2 inference requires execute_with_mask_async on wasm32",
                ))?
        }
    }

    /// Determine search tier and resolve candidates asynchronously across Desktop, Linux, and Web.
    ///
    /// # Errors
    /// Returns [`MatchError`] if `Dinov2` inference or descriptor pooling fails.
    pub async fn execute_with_mask_async(
        &self,
        prior: &SearchPrior,
        masked_crop: Option<&DynamicImage>,
        ground_mask: Option<&[bool]>,
        store: Option<&CorridorTileStore>,
        records: Option<&[Record]>,
    ) -> Result<AnchorMatch, MatchError> {
        let tier = resolve_search_tier_from_prior(prior, &self.config.search_config);
        match tier {
            SearchTier::Local => {
                let mut candidate_tiles = Vec::new();
                if let Some(s) = store {
                    let pos = prior.center.position;
                    candidate_tiles.push(s.tile_key_at(pos.x, pos.y));
                }
                Ok(AnchorMatch {
                    tier: SearchTier::Local,
                    candidate_tiles,
                    ranked_regions: Vec::new(),
                    query_descriptor: None,
                })
            }
            SearchTier::Region => {
                let candidates = prior.candidates(&self.config.search_config);
                if !candidates.is_empty() {
                    let mut candidate_tiles = Vec::with_capacity(candidates.len());
                    if let Some(s) = store {
                        for c in &candidates {
                            let key = s.tile_key_at(c.position.x, c.position.y);
                            if !candidate_tiles.contains(&key) {
                                candidate_tiles.push(key);
                            }
                        }
                    }
                    Ok(AnchorMatch {
                        tier: SearchTier::Region,
                        candidate_tiles,
                        ranked_regions: Vec::new(),
                        query_descriptor: None,
                    })
                } else {
                    self.execute_global_retrieval_async(masked_crop, ground_mask, store, records)
                        .await
                }
            }
            SearchTier::Global | _ => {
                self.execute_global_retrieval_async(masked_crop, ground_mask, store, records)
                    .await
            }
        }
    }

    async fn execute_global_retrieval_async(
        &self,
        masked_crop: Option<&DynamicImage>,
        ground_mask: Option<&[bool]>,
        store: Option<&CorridorTileStore>,
        records: Option<&[Record]>,
    ) -> Result<AnchorMatch, MatchError> {
        let active_records: &[Record] = records.unwrap_or(&self.records);
        let (Some(crop), Some(dinov2)) = (masked_crop, &self.dinov2) else {
            return Ok(AnchorMatch::empty_global());
        };
        if active_records.is_empty() {
            return Ok(AnchorMatch::empty_global());
        }

        let canvas = ground_crop_to_canvas(crop, ground_mask);
        let input = canvas.model_input(224);
        let values = dinov2.tokens(&input, 1).await?;
        let tokens = Tokens {
            width: DINOV2_WIDTH,
            values,
        };
        let query_desc = descriptor_from_tokens_and_canvas(&tokens, &canvas)
            .map_err(|_| MatchError::Output("failed to pool Dinov2 descriptor from canvas"))?;
        let ranked = rank_candidate_regions(
            &query_desc,
            active_records,
            self.config.top_k_global,
            self.config.global_spacing_m,
        );
        let candidate_tiles = if let Some(s) = store {
            if let Some(frame) = self.local_frame {
                let mut keys = Vec::with_capacity(ranked.len());
                for place in &ranked {
                    if let Ok(enu) = frame.local([place.centre[0], place.centre[1], 0.0]) {
                        let key = s.tile_key_at(enu.x, enu.y);
                        if !keys.contains(&key) {
                            keys.push(key);
                        }
                    }
                }
                keys
            } else {
                load_candidate_tiles(s, &ranked)
            }
        } else {
            Vec::new()
        };
        Ok(AnchorMatch {
            tier: SearchTier::Global,
            candidate_tiles,
            ranked_regions: ranked,
            query_descriptor: Some(query_desc),
        })
    }
}

#[cfg(test)]
mod tests;
