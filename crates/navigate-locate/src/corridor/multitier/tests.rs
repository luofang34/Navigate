use super::*;
use nalgebra::Matrix3;

struct FlatTerrain;

impl CorridorTerrain for FlatTerrain {
    fn geodetic(&self, local_enu: [f64; 3]) -> [f64; 3] {
        [
            40.0 + local_enu[1] / 111_320.0,
            -75.0 + local_enu[0] / 85_000.0,
            local_enu[2],
        ]
    }

    fn local(&self, lat_deg: f64, lon_deg: f64, alt_m: f64) -> [f64; 3] {
        [
            (lon_deg + 75.0) * 85_000.0,
            (lat_deg - 40.0) * 111_320.0,
            alt_m,
        ]
    }

    fn sample_dem(&self, _lat_deg: f64, _lon_deg: f64) -> Option<f64> {
        Some(120.0)
    }

    fn camera_rotation(
        &self,
        _lat_deg: f64,
        _lon_deg: f64,
        heading_deg: f64,
        down_deg: f64,
        roll_deg: f64,
    ) -> Matrix3<f64> {
        super::super::plausibility::local_camera_rotation(heading_deg, down_deg, roll_deg)
    }
}

#[test]
fn resolve_search_tier_boundaries() {
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
fn unconfigured_global_retrieval_fails_closed_without_fake_descriptor() {
    let matcher = SearchTierMatcher::new(MultiTierSearchConfig::default());
    let prior = build_search_prior_from_hint(None, None);
    let img = DynamicImage::new_rgb8(64, 64);
    let res = matcher.execute(&prior, Some(&img));
    assert!(res.is_ok());
    if let Ok(m) = res {
        assert_eq!(m.tier, SearchTier::Global);
        assert!(m.candidate_tiles.is_empty());
        assert!(m.ranked_regions.is_empty());
        assert!(m.query_descriptor.is_none());
    }
}

#[test]
fn global_fallback_seeds_expand_headings_and_pitches_at_ranked_places() {
    let ranked = vec![Ranked {
        centre: [40.01, -74.99],
        score: 0.92,
        query: 0,
    }];
    let seeds = global_fallback_seeds_for_places(&ranked, &FlatTerrain, 400.0, Some((14.0, 2.0)));
    // 1 live attitude + 3 distinct default pitches = 4 pitches * 6 headings = 24 seeds
    assert_eq!(seeds.len(), 24);
    assert!((seeds[0].0[2] - 520.0).abs() < 1e-6);
}
