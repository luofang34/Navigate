//! Seed camera pose generation across compass headings, pitches, and altitudes.

use super::MatchStageOptions;
use crate::{surface::Surface, testset::PhotoCase};
use nalgebra::Matrix3;

/// Shallow grazing pitch angle seeds in degrees.
pub const GRAZING_PITCH_SEEDS_DEG: [f64; 6] = [-10.0, 8.0, 22.0, 38.0, 58.0, 88.0];

/// Banked roll angle seeds in degrees.
pub const BANKED_ROLL_SEEDS_DEG: [f64; 3] = [0.0, 25.0, -25.0];

/// Generate seed camera poses across compass headings, pitches, and altitudes.
pub fn generate_seeds(
    surface: &Surface,
    case: &PhotoCase,
    opts: &MatchStageOptions,
    attitude: Option<(f64, f64)>,
) -> Vec<([f64; 3], Matrix3<f64>)> {
    let (lat, lon, base_alt) = case
        .prior
        .as_ref()
        .and_then(|p| p.position.as_ref())
        .map(|pos| {
            let dem = surface
                .sample_dem(pos.latitude, pos.longitude)
                .unwrap_or(100.0);
            let alt = match pos.altitude_m {
                Some(a) => a.max(dem + 25.0),
                None => (dem + 800.0).max(dem + 150.0),
            };
            (pos.latitude, pos.longitude, alt)
        })
        .unwrap_or((
            surface.origin_lat,
            surface.origin_lon,
            surface.min_elev_m + 900.0,
        ));
    let pitches: Vec<(f64, f64)> = match attitude {
        Some(att) => vec![att],
        None => {
            let mut list =
                Vec::with_capacity(GRAZING_PITCH_SEEDS_DEG.len() * BANKED_ROLL_SEEDS_DEG.len());
            for &p in &GRAZING_PITCH_SEEDS_DEG {
                list.push((p, 0.0));
            }
            for &r in &[25.0, -25.0] {
                for &p in &GRAZING_PITCH_SEEDS_DEG {
                    list.push((p, r));
                }
            }
            list
        }
    };
    let headings: Vec<f64> = if let Some(dir) = case.prior.as_ref().and_then(|p| p.direction_deg)
        && opts.heading_step_deg <= 0.0
    {
        [
            0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0, 180.0,
        ]
        .into_iter()
        .map(|d| (dir + d).rem_euclid(360.0))
        .collect()
    } else {
        let step = opts.heading_step_deg.max(30.0);
        let count = (360.0 / step).round().max(1.0) as usize;
        (0..count).map(|i| (i as f64) * step).collect()
    };

    let dem_h = surface.sample_dem(lat, lon).unwrap_or(surface.min_elev_m);
    let agl = (base_alt - dem_h).max(0.0);
    let prioritize_nadir = attitude.is_none() && agl <= 250.0;
    let mut out = Vec::new();
    for &hf in &opts.height_factors {
        let eye = surface.local(lat, lon, dem_h + agl.max(25.0) * hf.max(0.25));
        if prioritize_nadir {
            for &hdg in &headings {
                for &down_deg in &[88.0, 75.0] {
                    if out.len() >= opts.attempts.max(1) {
                        return out;
                    }
                    out.push((eye, surface.camera_rotation(lat, lon, hdg, down_deg, 0.0)));
                }
            }
            for &(down_deg, roll_deg) in &pitches {
                if (down_deg - 88.0).abs() < 1e-3 && roll_deg.abs() < 1e-3 {
                    continue;
                }
                for &hdg in &headings {
                    if out.len() >= opts.attempts.max(1) {
                        return out;
                    }
                    out.push((
                        eye,
                        surface.camera_rotation(lat, lon, hdg, down_deg, roll_deg),
                    ));
                }
            }
        } else {
            for &(down_deg, roll_deg) in &pitches {
                for &hdg in &headings {
                    if out.len() >= opts.attempts.max(1) {
                        return out;
                    }
                    out.push((
                        eye,
                        surface.camera_rotation(lat, lon, hdg, down_deg, roll_deg),
                    ));
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::{PhotoClaims, PositionClaim};
    use image::{Rgb, RgbImage};
    use std::path::PathBuf;

    fn dummy_opts(attempts: usize) -> MatchStageOptions {
        MatchStageOptions {
            dataset_dir: PathBuf::new(),
            profile_path: PathBuf::new(),
            catalog_path: PathBuf::new(),
            context_catalog: None,
            output_dir: PathBuf::new(),
            attempts,
            steps: 1,
            heading_step_deg: 30.0,
            height_factors: vec![1.0],
            only_cases: None,
        }
    }

    #[test]
    fn generate_seeds_supports_low_agl_and_nadir_fallback_pitch() {
        let img = RgbImage::from_pixel(256, 256, Rgb([100, 120, 90]));
        let surface = Surface::synthetic(41.0, -75.0, 100.0, img);
        let case = PhotoCase {
            id: "drone-low".into(),
            image: "img.png".into(),
            image_sha256: "0".repeat(64),
            size: [800, 600],
            valid_mask: String::new(),
            mask_sha256: String::new(),
            mask_status: String::new(),
            semantic_labels: None,
            semantic_sha256: None,
            prior: Some(PhotoClaims {
                position: Some(PositionClaim {
                    latitude: 41.0,
                    longitude: -75.0,
                    reported_horizontal_error_m: None,
                    altitude_m: Some(135.0),
                    unsigned_altitude_m: Some(135.0),
                    altitude_reference_text: None,
                    altitude_datum: "msl".into(),
                }),
                ..PhotoClaims::default()
            }),
            sequence: None,
            time_s: None,
        };

        let seeds = generate_seeds(&surface, &case, &dummy_opts(72), None);
        assert_eq!(seeds.len(), 72);
        let [_, _, alt_m] = surface.geodetic(seeds[0].0);
        assert!(
            (alt_m - 135.0).abs() < 1.0,
            "expected low-AGL altitude near 135 m, got {alt_m}"
        );
        // Nadir pitch 88.0 deg is prioritized first at index 0 -> -rot[(2, 2)] == sin(88 deg) > 0.999
        let nadir_rot = seeds[0].1;
        assert!(
            -nadir_rot[(2, 2)] > 0.998,
            "expected 88 deg nadir fallback pitch, got -r22={}",
            -nadir_rot[(2, 2)]
        );
        let nadir75_rot = seeds[1].1;
        assert!(
            -nadir75_rot[(2, 2)] > 0.96,
            "expected 75 deg nadir pitch, got -r22={}",
            -nadir75_rot[(2, 2)]
        );
    }
}
