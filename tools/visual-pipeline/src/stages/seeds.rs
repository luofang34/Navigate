//! Seed camera pose generation across compass headings, pitches, and altitudes.

use super::MatchStageOptions;
use crate::{surface::Surface, testset::PhotoCase};
use nalgebra::Matrix3;

pub(crate) fn generate_seeds(
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
        None => vec![
            (35.0, 0.0),
            (55.0, 0.0),
            (75.0, 0.0),
            (20.0, 0.0),
            (88.0, 0.0),
        ],
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
    let agl = (base_alt - dem_h).max(25.0);
    let mut out = Vec::new();
    for &hf in &opts.height_factors {
        let eye = surface.local(lat, lon, dem_h + agl * hf.max(0.25));
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
        };

        let seeds = generate_seeds(&surface, &case, &dummy_opts(60), None);
        assert_eq!(seeds.len(), 60);
        let [_, _, alt_m] = surface.geodetic(seeds[0].0);
        assert!(
            (alt_m - 135.0).abs() < 1.0,
            "expected low-AGL altitude near 135 m, got {alt_m}"
        );
        // Pitch #5 (index 48..60) is 88.0 deg down -> -rot[(2, 2)] == sin(88 deg) > 0.999
        let nadir_rot = seeds[48].1;
        assert!(
            -nadir_rot[(2, 2)] > 0.998,
            "expected 88 deg nadir fallback pitch, got -r22={}",
            -nadir_rot[(2, 2)]
        );
    }
}
