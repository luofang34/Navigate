//! Multi-heading and multi-pitch global fallback seed generation from ranked places.

use super::SearchTierMatcher;
use crate::corridor::plausibility::CorridorTerrain;
use image::DynamicImage;
use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use navigate_match::{CorridorTileStore, MatchError};
use navigate_place::Ranked;
use navigate_visual::{CameraPose, SearchPrior};

const GLOBAL_HEADINGS_DEG: [f64; 6] = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0];

/// Build a [`SearchPrior`] from an optional predicted pose and horizontal uncertainty in meters.
#[must_use]
pub fn build_search_prior_from_hint(
    pred_pose: Option<([f64; 3], Matrix3<f64>)>,
    uncertainty_m: Option<f64>,
) -> SearchPrior {
    let (eye, rot) = pred_pose.unwrap_or(([0.0, 0.0, 500.0], Matrix3::identity()));
    let sigma_sq = match (pred_pose, uncertainty_m) {
        (Some(_), Some(u)) => u.clamp(1.0, 20_000.0).powi(2),
        (Some(_), None) => 2500.0,
        (None, _) => 100_000_000.0,
    };
    SearchPrior {
        center: CameraPose {
            position: Vector3::new(eye[0], eye[1], eye[2]),
            orientation: UnitQuaternion::from_matrix(&rot.transpose()),
        },
        position_covariance_m2: Matrix3::from_diagonal(&Vector3::new(sigma_sq, sigma_sq, 100.0)),
        attitude_sigma_rad: 0.1,
    }
}

/// Seed pose `(eye_enu, world_to_cam)` for global fallback anchor acquisition.
pub type GlobalFallbackSeed = ([f64; 3], Matrix3<f64>);
/// Ordered list of global fallback seed poses.
pub type GlobalFallbackSeeds = Vec<GlobalFallbackSeed>;

/// Expand top-ranked geographic places into multi-heading and multi-pitch candidate seeds.
#[must_use]
pub fn global_fallback_seeds_for_places<S: CorridorTerrain + ?Sized>(
    ranked_places: &[Ranked],
    surface: &S,
    nominal_agl_m: f64,
    live_attitude: Option<(f64, f64)>,
) -> GlobalFallbackSeeds {
    if ranked_places.is_empty() {
        return Vec::new();
    }
    let agl = nominal_agl_m.max(25.0);
    let mut pitches: Vec<(f64, f64)> = Vec::with_capacity(4);
    if let Some(att) = live_attitude {
        pitches.push(att);
    }
    let defaults: [(f64, f64); 3] = if agl <= 250.0 {
        [(88.0, 0.0), (50.0, 0.0), (22.0, 0.0)]
    } else {
        [(22.0, 0.0), (50.0, 0.0), (88.0, 0.0)]
    };
    for cand_p in defaults {
        if !pitches
            .iter()
            .any(|&(d, r)| (d - cand_p.0).abs() < 4.0 && (r - cand_p.1).abs() < 4.0)
        {
            pitches.push(cand_p);
        }
    }

    let mut seeds =
        Vec::with_capacity(ranked_places.len() * GLOBAL_HEADINGS_DEG.len() * pitches.len());
    for place in ranked_places {
        let [lat_deg, lon_deg] = place.centre;
        let dem_msl = surface.sample_dem(lat_deg, lon_deg).unwrap_or(0.0);
        let eye_enu = surface.local(lat_deg, lon_deg, dem_msl + agl);
        for &(down_deg, roll_deg) in &pitches {
            for &hdg_deg in &GLOBAL_HEADINGS_DEG {
                let rot = surface.camera_rotation(lat_deg, lon_deg, hdg_deg, down_deg, roll_deg);
                seeds.push((eye_enu, rot));
            }
        }
    }
    seeds
}

/// Resolve global fallback seeds without an explicit ground mask or attitude hint.
///
/// # Errors
/// Returns [`MatchError`] if `Dinov2` retrieval fails.
pub fn resolve_global_fallback_seeds<S: CorridorTerrain + ?Sized>(
    matcher: &SearchTierMatcher,
    prior: &SearchPrior,
    query_image: &DynamicImage,
    surface: &S,
    store: &CorridorTileStore,
    nominal_agl_m: f64,
) -> Result<GlobalFallbackSeeds, MatchError> {
    resolve_global_fallback_seeds_with_mask(
        matcher,
        prior,
        query_image,
        None,
        surface,
        store,
        nominal_agl_m,
        None,
    )
}

/// Resolve multi-heading and multi-pitch global fallback seeds using an active ground mask.
///
/// # Errors
/// Returns [`MatchError`] if `Dinov2` retrieval fails.
#[expect(
    clippy::too_many_arguments,
    reason = "global fallback coordinates matcher, search prior, masked query image, terrain DEM, tile store, AGL, and horizon attitude"
)]
pub fn resolve_global_fallback_seeds_with_mask<S: CorridorTerrain + ?Sized>(
    matcher: &SearchTierMatcher,
    prior: &SearchPrior,
    query_image: &DynamicImage,
    ground_mask: Option<&[bool]>,
    surface: &S,
    store: &CorridorTileStore,
    nominal_agl_m: f64,
    live_attitude: Option<(f64, f64)>,
) -> Result<GlobalFallbackSeeds, MatchError> {
    if !matcher.has_global_retrieval() {
        return Ok(Vec::new());
    }
    let global_prior = build_search_prior_from_hint(None, None);
    let _ = prior;
    let match_res = matcher.execute_with_mask(
        &global_prior,
        Some(query_image),
        ground_mask,
        Some(store),
        None,
    )?;
    Ok(global_fallback_seeds_for_places(
        &match_res.ranked_regions,
        surface,
        nominal_agl_m,
        live_attitude,
    ))
}
