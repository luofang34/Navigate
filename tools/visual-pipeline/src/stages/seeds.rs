//! Seed camera pose generation across compass headings, pitches, and altitudes.

use super::MatchStageOptions;
use crate::{surface::Surface, testset::PhotoCase};
use nalgebra::{Matrix3, Vector3};

/// Shallow grazing pitch angle seeds in degrees.
pub const GRAZING_PITCH_SEEDS_DEG: [f64; 6] = [-10.0, 8.0, 22.0, 38.0, 58.0, 88.0];

/// Banked roll angle seeds in degrees.
pub const BANKED_ROLL_SEEDS_DEG: [f64; 3] = [0.0, 25.0, -25.0];

type SeedPose = ([f64; 3], Matrix3<f64>);

fn resolve_prior_origin(surface: &Surface, case: &PhotoCase) -> (f64, f64, f64, f64) {
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
    let dem_h = surface.sample_dem(lat, lon).unwrap_or(surface.min_elev_m);
    let agl = (base_alt - dem_h).max(0.0);
    (lat, lon, dem_h, agl)
}

fn resolve_pitches(attitude: Option<(f64, f64)>) -> Vec<(f64, f64)> {
    match attitude {
        Some((down_deg, roll_deg)) => vec![
            (down_deg, roll_deg),
            ((down_deg + 10.0).clamp(-30.0, 88.0), roll_deg),
            ((down_deg - 10.0).clamp(-30.0, 88.0), roll_deg),
            ((down_deg + 22.0).clamp(-30.0, 88.0), 0.0),
        ],
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
    }
}

fn resolve_headings(case: &PhotoCase, heading_step_deg: f64) -> Vec<f64> {
    if let Some(dir) = case.prior.as_ref().and_then(|p| p.direction_deg)
        && heading_step_deg <= 0.0
    {
        [
            0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0, 120.0, -120.0, 150.0, -150.0, 180.0,
        ]
        .into_iter()
        .map(|d| (dir + d).rem_euclid(360.0))
        .collect()
    } else {
        let step = heading_step_deg.max(30.0);
        let count = (360.0 / step).round().max(1.0) as usize;
        (0..count).map(|i| (i as f64) * step).collect()
    }
}

/// Generate seed camera poses across compass headings, pitches, and altitudes.
pub fn generate_seeds(
    surface: &Surface,
    case: &PhotoCase,
    opts: &MatchStageOptions,
    attitude: Option<(f64, f64)>,
) -> Vec<SeedPose> {
    let (lat, lon, dem_h, agl) = resolve_prior_origin(surface, case);
    let pitches = resolve_pitches(attitude);
    let headings = resolve_headings(case, opts.heading_step_deg);
    let prioritize_nadir = attitude.is_none() && agl <= 250.0;
    let height_factors: &[f64] = if opts.height_factors.is_empty() {
        &[1.0]
    } else {
        &opts.height_factors
    };
    let cap = opts.attempts.max(1);
    let mut out = Vec::with_capacity(cap);

    if prioritize_nadir {
        for &hf in height_factors {
            let eye = surface.local(lat, lon, dem_h + agl.max(25.0) * hf.max(0.25));
            for &down_deg in &[88.0, 75.0] {
                for &hdg in &headings {
                    if out.len() >= cap {
                        return out;
                    }
                    out.push((eye, surface.camera_rotation(lat, lon, hdg, down_deg, 0.0)));
                }
            }
        }
        for &hf in height_factors {
            let eye = surface.local(lat, lon, dem_h + agl.max(25.0) * hf.max(0.25));
            for &(down_deg, roll_deg) in &pitches {
                if (down_deg - 88.0).abs() < 1e-3 && roll_deg.abs() < 1e-3 {
                    continue;
                }
                for &hdg in &headings {
                    if out.len() >= cap {
                        return out;
                    }
                    out.push((
                        eye,
                        surface.camera_rotation(lat, lon, hdg, down_deg, roll_deg),
                    ));
                }
            }
        }
    } else if attitude.is_none() && headings.len() > 2 {
        let n_pitches = GRAZING_PITCH_SEEDS_DEG.len();
        for &roll_deg in &BANKED_ROLL_SEEDS_DEG {
            for pass in 0..n_pitches {
                for &hf in height_factors {
                    let eye = surface.local(lat, lon, dem_h + agl.max(25.0) * hf.max(0.25));
                    for (h_idx, &hdg) in headings.iter().enumerate() {
                        if out.len() >= cap {
                            return out;
                        }
                        let down_deg = GRAZING_PITCH_SEEDS_DEG[(h_idx + pass) % n_pitches];
                        out.push((
                            eye,
                            surface.camera_rotation(lat, lon, hdg, down_deg, roll_deg),
                        ));
                    }
                }
            }
        }
    } else {
        for &(down_deg, roll_deg) in &pitches {
            for &hf in height_factors {
                let eye = surface.local(lat, lon, dem_h + agl.max(25.0) * hf.max(0.25));
                for &hdg in &headings {
                    if out.len() >= cap {
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

fn same_seed_pose(a: &SeedPose, b: &SeedPose) -> bool {
    let dist = (Vector3::from(a.0) - Vector3::from(b.0)).norm();
    let cos_ang = (((a.1 * b.1.transpose()).trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
    dist <= 8.0 && cos_ang >= 0.995
}

fn push_unique_seed(out: &mut Vec<SeedPose>, cand: SeedPose, cap: usize) {
    if out.len() < cap && !out.iter().any(|prev| same_seed_pose(prev, &cand)) {
        out.push(cand);
    }
}

/// Build a stratified coarse seed slice covering headings, altitudes, pitches, and ring offsets.
pub(crate) fn build_stratified_coarse_seeds(
    surface: &Surface,
    case: &PhotoCase,
    opts: &MatchStageOptions,
    attitude: Option<(f64, f64)>,
    pred_pose: Option<SeedPose>,
    coarse_cap: usize,
) -> (Vec<SeedPose>, Vec<SeedPose>) {
    let full_seeds: Vec<SeedPose> = pred_pose
        .into_iter()
        .chain(
            generate_seeds(surface, case, opts, attitude)
                .into_iter()
                .take(opts.attempts.max(1)),
        )
        .collect();
    let cap = coarse_cap.max(4);
    let mut coarse = Vec::with_capacity(cap);
    let (lat, lon, dem_h, agl) = resolve_prior_origin(surface, case);
    let pitches = resolve_pitches(attitude);
    let headings = resolve_headings(case, opts.heading_step_deg);
    let heading_unknown =
        opts.heading_step_deg > 0.0 || case.prior.as_ref().and_then(|p| p.direction_deg).is_none();
    let want_probes = !headings.is_empty() && !pitches.is_empty() && cap >= 6;
    let reserve = if !want_probes {
        0
    } else if cap >= 12 {
        6
    } else {
        4
    };
    let center_cap = cap.saturating_sub(reserve).clamp(2, 12);
    let lead = usize::from(pred_pose.is_some());
    if lead == 1
        && let Some(&p0) = full_seeds.first()
    {
        push_unique_seed(&mut coarse, p0, center_cap);
    }
    if heading_unknown && full_seeds.len() >= lead + 12 && center_cap <= 6 {
        let offsets: &[usize] = if attitude.is_none() && agl > 250.0 {
            &[0, 2, 4, 7, 9, 11]
        } else {
            &[0, 2, 4, 6, 8, 10]
        };
        for &off in offsets {
            push_unique_seed(&mut coarse, full_seeds[lead + off], center_cap);
        }
    }
    for &seed in &full_seeds {
        if coarse.len() >= center_cap {
            break;
        }
        push_unique_seed(&mut coarse, seed, center_cap);
    }

    let alt_factors: &[f64] = if opts.height_factors.len() > 1 {
        &opts.height_factors
    } else {
        &[1.0, 0.80, 1.25]
    };

    if want_probes {
        let (down0, roll0) = if attitude.is_none() && agl <= 250.0 {
            (88.0, 0.0)
        } else {
            pitches[0]
        };
        let ring_limit = if cap >= 12 { 4 } else { 2 };
        let alt_cap = cap.saturating_sub(ring_limit);
        let hdg_per_alt = if cap >= 16 { 2 } else { 1 };
        for (probe_i, &hf) in alt_factors.iter().skip(1).take(2).enumerate() {
            let eye = surface.local(lat, lon, dem_h + agl.max(25.0) * hf.max(0.25));
            for h_slot in 0..hdg_per_alt {
                let hdg = if heading_unknown {
                    headings[(probe_i * 6 + h_slot * 3) % headings.len()]
                } else {
                    headings[h_slot % headings.len()]
                };
                let rot = surface.camera_rotation(lat, lon, hdg, down0, roll0);
                push_unique_seed(&mut coarse, (eye, rot), alt_cap);
            }
        }

        let center_xy = surface.local(lat, lon, 0.0);
        let ring_step_m = (agl * 0.45).clamp(65.0, 420.0);
        for (k, (dx, dy)) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
            .into_iter()
            .take(ring_limit)
            .enumerate()
        {
            let [rlat, rlon, _] = surface.geodetic([
                center_xy[0] + dx * ring_step_m,
                center_xy[1] + dy * ring_step_m,
                0.0,
            ]);
            let rdem = surface.sample_dem(rlat, rlon).unwrap_or(dem_h);
            let reye = surface.local(rlat, rlon, rdem + agl.max(25.0));
            let hdg = if heading_unknown {
                headings[(k * 3) % headings.len()]
            } else {
                headings[0]
            };
            let rrot = surface.camera_rotation(rlat, rlon, hdg, down0, roll0);
            push_unique_seed(&mut coarse, (reye, rrot), cap);
        }
    }

    for &seed in &full_seeds {
        push_unique_seed(&mut coarse, seed, cap);
    }
    let remaining: Vec<SeedPose> = full_seeds
        .into_iter()
        .filter(|s| !coarse.iter().any(|c| same_seed_pose(c, s)))
        .collect();
    (coarse, remaining)
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
        let nadir_rot = seeds[0].1;
        assert!(
            -nadir_rot[(2, 2)] > 0.998,
            "expected 88 deg nadir fallback pitch, got -r22={}",
            -nadir_rot[(2, 2)]
        );
        let nadir75_rot = seeds[12].1;
        assert!(
            -nadir75_rot[(2, 2)] > 0.96 && -nadir75_rot[(2, 2)] < 0.98,
            "expected 75 deg nadir pitch after 12 compass headings, got -r22={}",
            -nadir75_rot[(2, 2)]
        );
    }

    #[test]
    fn generate_seeds_high_agl_interleaves_pitches_across_first_twelve_attempts() {
        let img = RgbImage::from_pixel(256, 256, Rgb([100, 120, 90]));
        let surface = Surface::synthetic(41.0, -75.0, 100.0, img);
        let case = PhotoCase {
            id: "high-agl".into(),
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
                    altitude_m: Some(1_100.0),
                    unsigned_altitude_m: Some(1_100.0),
                    altitude_reference_text: None,
                    altitude_datum: "msl".into(),
                }),
                ..PhotoClaims::default()
            }),
            sequence: None,
            time_s: None,
        };

        let seeds = generate_seeds(&surface, &case, &dummy_opts(12), None);
        assert_eq!(seeds.len(), 12);
        let mut distinct_sin_pitches = Vec::new();
        for &(_, rot) in &seeds {
            let sin_down = (-rot[(2, 2)]).clamp(-1.0, 1.0);
            if !distinct_sin_pitches
                .iter()
                .any(|&prev: &f64| (prev - sin_down).abs() < 0.02)
            {
                distinct_sin_pitches.push(sin_down);
            }
        }
        assert_eq!(
            distinct_sin_pitches.len(),
            GRAZING_PITCH_SEEDS_DEG.len(),
            "first 12 seeds must cover all 6 grazing pitches rather than repeating -10 deg"
        );
    }

    #[test]
    fn stratified_coarse_seeds_sample_altitudes_and_spatial_ring_offsets() {
        let img = RgbImage::from_pixel(256, 256, Rgb([100, 120, 90]));
        let surface = Surface::synthetic(41.0, -75.0, 100.0, img);
        let case = PhotoCase {
            id: "mojave-uncal".into(),
            image: "img.png".into(),
            image_sha256: "0".repeat(64),
            size: [1920, 1080],
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
                    altitude_m: Some(700.0),
                    unsigned_altitude_m: Some(700.0),
                    altitude_reference_text: None,
                    altitude_datum: "msl".into(),
                }),
                direction_deg: Some(90.0),
                ..PhotoClaims::default()
            }),
            sequence: None,
            time_s: None,
        };
        let mut opts = dummy_opts(36);
        opts.heading_step_deg = 0.0;
        for (cap_in, hdg_step) in [
            (12_usize, 0.0_f64),
            (12_usize, 30.0_f64),
            (20_usize, 0.0_f64),
        ] {
            opts.heading_step_deg = hdg_step;
            let (coarse, remaining) = build_stratified_coarse_seeds(
                &surface,
                &case,
                &opts,
                Some((18.0, 0.0)),
                None,
                cap_in,
            );
            assert_eq!(coarse.len(), cap_in);
            assert!(!remaining.is_empty());

            let has_alt_variation = coarse.iter().any(|&(eye, _)| {
                let [_, _, alt] = surface.geodetic(eye);
                (alt - 700.0).abs() > 80.0
            });
            let has_xy_ring_offset = coarse.iter().any(|&(eye, _)| eye[0].hypot(eye[1]) > 150.0);
            assert!(
                has_alt_variation,
                "cap={cap_in} step={hdg_step}: must sample off-nominal altitudes"
            );
            assert!(
                has_xy_ring_offset,
                "cap={cap_in} step={hdg_step}: must sample spatial ring offsets"
            );
        }
    }
}
