//! Multi-tier candidate retrieval and DINOv2 place recognition bridge (Feature 14).

use crate::pose::PoseCandidate;
use image::DynamicImage;
use navigate_match::{CorridorTileKey, CorridorTileStore, DINOV2_WIDTH, Dinov2, MatchError};
use navigate_place::{
    CANVAS_PX, Canvas, Descriptor, GRID, Ranked, Record, Tokens, cell_weights, describe, rank,
};
use navigate_visual::{SearchConfig, SearchPrior, SearchTier};
use std::sync::Arc;

/// Outcome of multi-tier anchor search evaluation.
#[derive(Clone, Debug, PartialEq)]
pub enum SearchTierOutcome {
    /// Solved at local tier with predicted pose and attitude perturbation.
    LocalSolved(PoseCandidate),
    /// Solved at region tier across covariance grid candidates.
    RegionSolved(PoseCandidate),
    /// Solved at global tier through DINOv2 place recognition retrieval.
    GlobalSolved(PoseCandidate),
    /// Global retrieval fallback required.
    GlobalFallbackNeeded {
        /// Center coordinate of rough prior if available.
        rough_prior: Option<[f64; 3]>,
    },
}

/// Resolves search tier from uncertainty radius.
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

/// Resolves search tier directly from a [`SearchPrior`].
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

/// Converts a masked ground image into a normalized [`Canvas`].
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

/// Compute a pooled 384-dimensional unit descriptor from patch tokens and ground mask.
///
/// # Errors
/// Returns [`navigate_place::PlaceError`] if pooling fails or canvas has no valid cells.
pub fn descriptor_from_tokens_and_canvas(
    tokens: &Tokens,
    canvas: &Canvas,
) -> Result<Descriptor, navigate_place::PlaceError> {
    let weights = cell_weights(canvas);
    describe(tokens, &weights, None)
}

/// Retrieve top-K candidate geographic places using cosine similarity ranking.
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

/// Retrieve and load corresponding corridor tile keys in [`CorridorTileStore`].
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

/// Match outcome and candidate resolution across search tiers.
#[derive(Clone, Debug, PartialEq)]
pub struct AnchorMatch {
    /// Active search tier determined from prior covariance.
    pub tier: SearchTier,
    /// Candidate tile keys resolved in CorridorTileStore.
    pub candidate_tiles: Vec<CorridorTileKey>,
    /// Ranked candidate regions from place recognition.
    pub ranked_regions: Vec<Ranked>,
    /// Extracted place descriptor if global retrieval was executed.
    pub query_descriptor: Option<Descriptor>,
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

/// Coordinator for multi-tier candidate search and retrieval.
#[derive(Clone)]
pub struct SearchTierMatcher {
    /// Multi-tier search configuration.
    pub config: MultiTierSearchConfig,
    /// Reference place records for global visual place recognition.
    pub records: Vec<Record>,
    /// Optional CorridorTileStore for resolving tile keys from ranked places.
    pub store: Option<Arc<CorridorTileStore>>,
    /// Optional DINOv2 model for feature extraction.
    pub dinov2: Option<Arc<Dinov2>>,
}

impl std::fmt::Debug for SearchTierMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchTierMatcher")
            .field("config", &self.config)
            .field("records_len", &self.records.len())
            .field("has_store", &self.store.is_some())
            .field("has_dinov2", &self.dinov2.is_some())
            .finish()
    }
}

impl SearchTierMatcher {
    /// Construct a new search tier matcher with configuration.
    #[must_use]
    pub fn new(config: MultiTierSearchConfig) -> Self {
        Self {
            config,
            records: Vec::new(),
            store: None,
            dinov2: None,
        }
    }

    /// Attach reference records for candidate region matching.
    #[must_use]
    pub fn with_records(mut self, records: Vec<Record>) -> Self {
        self.records = records;
        self
    }

    /// Attach corridor tile store for tile resolution.
    #[must_use]
    pub fn with_store(mut self, store: Arc<CorridorTileStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Attach DINOv2 model for feature extraction.
    #[must_use]
    pub fn with_dinov2(mut self, dinov2: Arc<Dinov2>) -> Self {
        self.dinov2 = Some(dinov2);
        self
    }

    /// Determine search tier and resolve candidate tiles.
    ///
    /// # Errors
    /// Returns [`MatchError`] if resolution fails.
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

    /// Determine search tier and resolve candidates with explicit store and records context.
    ///
    /// # Errors
    /// Returns [`MatchError`] if resolution fails.
    pub fn execute_with_context(
        &self,
        prior: &SearchPrior,
        masked_crop: Option<&DynamicImage>,
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
                    self.execute_global_retrieval(prior, masked_crop, store, records)
                }
            }
            SearchTier::Global | _ => {
                self.execute_global_retrieval(prior, masked_crop, store, records)
            }
        }
    }

    fn execute_global_retrieval(
        &self,
        prior: &SearchPrior,
        masked_crop: Option<&DynamicImage>,
        store: Option<&CorridorTileStore>,
        records: Option<&[Record]>,
    ) -> Result<AnchorMatch, MatchError> {
        let Some(crop) = masked_crop else {
            return Ok(AnchorMatch {
                tier: SearchTier::Global,
                candidate_tiles: Vec::new(),
                ranked_regions: Vec::new(),
                query_descriptor: None,
            });
        };
        let canvas = ground_crop_to_canvas(crop, None);
        let query_desc = self.extract_query_descriptor(&canvas)?;
        let active_records: &[Record] = records.unwrap_or(&self.records);
        let ranked = if !active_records.is_empty() {
            rank_candidate_regions(
                &query_desc,
                active_records,
                self.config.top_k_global,
                self.config.global_spacing_m,
            )
        } else {
            Vec::new()
        };
        let candidate_tiles = if let Some(s) = store {
            if !ranked.is_empty() {
                load_candidate_tiles(s, &ranked)
            } else {
                let pos = prior.center.position;
                if pos.x.is_finite() && pos.y.is_finite() {
                    vec![s.tile_key_at(pos.x, pos.y)]
                } else {
                    Vec::new()
                }
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

    fn extract_query_descriptor(&self, canvas: &Canvas) -> Result<Descriptor, MatchError> {
        if let Some(dinov2) = &self.dinov2 {
            let input = canvas.model_input(224);
            let values = pollster::block_on(dinov2.tokens(&input, 1))?;
            let tokens = Tokens {
                width: DINOV2_WIDTH,
                values,
            };
            descriptor_from_tokens_and_canvas(&tokens, canvas)
                .map_err(|_| MatchError::Output("failed to pool descriptor from tokens and canvas"))
        } else {
            let weights = cell_weights(canvas);
            let mut values = vec![0.0f32; 384];
            let step = CANVAS_PX / GRID;
            for cy in 0..GRID {
                for cx in 0..GRID {
                    let idx = cy * GRID + cx;
                    if weights[idx] {
                        let px_idx = (cy * step) * CANVAS_PX + (cx * step);
                        let rgb = canvas.rgb[px_idx];
                        let slot = (idx * 3) % 384;
                        values[slot] += rgb[0] / 255.0;
                        values[(slot + 1) % 384] += rgb[1] / 255.0;
                        values[(slot + 2) % 384] += rgb[2] / 255.0;
                    }
                }
            }
            let norm: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for v in &mut values {
                    *v /= norm;
                }
            }
            Ok(Descriptor { values })
        }
    }
}

/// Construct search prior from seed hint.
#[must_use]
pub fn build_search_prior_from_hint(
    pred_pose: Option<([f64; 3], nalgebra::Matrix3<f64>)>,
    uncertainty_m: Option<f64>,
) -> SearchPrior {
    let (eye, rot) = pred_pose.unwrap_or(([0.0, 0.0, 500.0], nalgebra::Matrix3::identity()));
    let sigma_sq = match (pred_pose, uncertainty_m) {
        (Some(_), Some(u)) => u.clamp(1.0, 20_000.0).powi(2),
        (Some(_), None) => 2500.0,
        (None, _) => 100_000_000.0,
    };
    SearchPrior {
        center: navigate_visual::CameraPose {
            position: nalgebra::Vector3::new(eye[0], eye[1], eye[2]),
            orientation: nalgebra::UnitQuaternion::from_matrix(&rot.transpose()),
        },
        position_covariance_m2: nalgebra::Matrix3::from_diagonal(&nalgebra::Vector3::new(
            sigma_sq, sigma_sq, 100.0,
        )),
        attitude_sigma_rad: 0.1,
    }
}

/// Seed pose for fallback anchor acquisition.
pub type GlobalFallbackSeed = ([f64; 3], nalgebra::Matrix3<f64>);
/// Vector of global fallback seed poses.
pub type GlobalFallbackSeeds = Vec<GlobalFallbackSeed>;

/// Resolve global fallback seeds when local and regional searches fail.
///
/// # Errors
/// Returns [`MatchError`] if candidate retrieval fails.
pub fn resolve_global_fallback_seeds(
    matcher: &SearchTierMatcher,
    prior: &SearchPrior,
    query_image: &DynamicImage,
    surface: &crate::surface::Surface,
    store: &CorridorTileStore,
    nominal_agl_m: f64,
) -> Result<GlobalFallbackSeeds, MatchError> {
    let match_res = matcher.execute_with_context(prior, Some(query_image), Some(store), None)?;
    let tile_m = f64::from(store.tile_size_m.max(10.0));
    let mut seeds = Vec::with_capacity(match_res.candidate_tiles.len());
    for key in match_res.candidate_tiles {
        let ce = (f64::from(key.tile_x) + 0.5) * tile_m;
        let cn = (f64::from(key.tile_y) + 0.5) * tile_m;
        let [lat, lon, _] = surface.geodetic([ce, cn, 0.0]);
        let dem_msl = surface.sample_dem(lat, lon).unwrap_or(0.0);
        let eye_enu = [ce, cn, dem_msl + nominal_agl_m];
        let rot_nadir = nalgebra::Matrix3::new(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0);
        seeds.push((eye_enu, rot_nadir));
    }
    Ok(seeds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_search_tier_boundaries() {
        assert_eq!(
            resolve_search_tier(Some(20.0), 50.0, 2000.0),
            SearchTier::Local
        );
        assert_eq!(
            resolve_search_tier(Some(50.0), 50.0, 2000.0),
            SearchTier::Local
        );
        assert_eq!(
            resolve_search_tier(Some(50.1), 50.0, 2000.0),
            SearchTier::Region
        );
        assert_eq!(
            resolve_search_tier(Some(2000.0), 50.0, 2000.0),
            SearchTier::Region
        );
        assert_eq!(
            resolve_search_tier(Some(2000.1), 50.0, 2000.0),
            SearchTier::Global
        );
        assert_eq!(resolve_search_tier(None, 50.0, 2000.0), SearchTier::Global);
    }

    #[test]
    fn test_rank_candidate_regions_empty() {
        let query = Descriptor {
            values: vec![0.1; 384],
        };
        let ranked = rank_candidate_regions(&query, &[], 5, 50.0);
        assert!(ranked.is_empty());
    }

    #[test]
    fn test_ground_crop_to_canvas_basic() {
        let img = DynamicImage::new_rgb8(100, 100);
        let canvas = ground_crop_to_canvas(&img, None);
        assert_eq!(canvas.rgb.len(), CANVAS_PX * CANVAS_PX);
        assert_eq!(canvas.valid.len(), CANVAS_PX * CANVAS_PX);
    }

    #[test]
    fn test_anchor_match_partial_eq() {
        let m = AnchorMatch {
            tier: SearchTier::Local,
            candidate_tiles: Vec::new(),
            ranked_regions: Vec::new(),
            query_descriptor: None,
        };
        assert_eq!(m, SearchTier::Local);
        assert_eq!(SearchTier::Local, m);
        assert_ne!(m, SearchTier::Global);
    }
}
