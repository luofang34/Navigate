//! Seed camera pose generation, stratified coarse slices, and focal-sweep estimation.

use crate::surface::Surface;
use nalgebra::{Matrix3, Vector3};

/// Candidate seed camera pose `(eye_enu_m, world_to_cam_rotation)`.
pub type SeedPose = ([f64; 3], Matrix3<f64>);

/// Shallow grazing and steep nadir pitch seeds in degrees.
pub const GRAZING_PITCH_SEEDS_DEG: [f64; 6] = [-10.0, 8.0, 22.0, 38.0, 58.0, 88.0];

/// Banked roll seeds in degrees.
pub const BANKED_ROLL_SEEDS_DEG: [f64; 3] = [0.0, 25.0, -25.0];

/// Default 35mm-equivalent focal length in millimeters (`26.0` mm).
pub const DEFAULT_FOCAL_MM: f64 = 26.0;

/// Standard focal-multiplier sweep values for oblique or high-AGL uncalibrated cameras.
pub const OBLIQUE_FOCAL_MULTIPLIERS: [f64; 4] = [1.0, 0.55, 1.45, 2.0];

/// Dense focal-multiplier sweep values for low-AGL (`<= 250 m`) nadir uncalibrated cameras.
pub const LOW_AGL_FOCAL_MULTIPLIERS: [f64; 4] = [0.88, 1.0, 0.76, 1.22];

/// Geodetic position, altitude, and compass heading prior for cold anchor acquisition.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AnchorPrior {
    /// Prior latitude in degrees.
    pub lat_deg: Option<f64>,
    /// Prior longitude in degrees.
    pub lon_deg: Option<f64>,
    /// Prior altitude above mean sea level in meters.
    pub alt_msl_m: Option<f64>,
    /// Optional compass heading in degrees clockwise from North (`0..360`).
    pub heading_deg: Option<f64>,
}

impl AnchorPrior {
    /// Return `(lat_deg, lon_deg)` when both coordinates are finite.
    #[must_use]
    pub fn lat_lon(self) -> Option<(f64, f64)> {
        let (lat, lon) = (self.lat_deg?, self.lon_deg?);
        (lat.is_finite() && lon.is_finite()).then_some((lat, lon))
    }

    /// Return `heading_deg` normalized into `[0, 360)` when finite.
    #[must_use]
    pub fn finite_heading_deg(self) -> Option<f64> {
        self.heading_deg
            .filter(|d| d.is_finite())
            .map(|d| d.rem_euclid(360.0))
    }
}

/// Configuration parameters for a corridor map-anchor search.
#[derive(Clone, Debug)]
pub struct AnchorSearchOpts {
    /// Maximum seed poses to evaluate per focal hypothesis.
    pub attempts: usize,
    /// Compass heading sweep step in degrees (`<= 0.0` means heading prior is trusted).
    pub heading_step_deg: f64,
    /// Altitude scale factors around the prior height above ground level.
    pub height_factors: Vec<f64>,
    /// Maximum query keypoints used during matching.
    pub query_keypoints: usize,
    /// Enable GPU or sparse terrain polish on top seed candidates.
    pub polish_anchors: bool,
    /// Enable tight real-time latency budgets (`ExecutionPolicy::LiveFlight`).
    pub pace_realtime: bool,
    /// Optional one-sigma horizontal uncertainty in meters for multi-tier place search.
    pub anchor_uncertainty_m: Option<f64>,
}

impl Default for AnchorSearchOpts {
    fn default() -> Self {
        Self {
            attempts: 36,
            heading_step_deg: 0.0,
            height_factors: vec![1.0],
            query_keypoints: 1024,
            polish_anchors: true,
            pace_realtime: false,
            anchor_uncertainty_m: None,
        }
    }
}

/// Return true when `prior` carries a trusted heading (`heading_step_deg <= 0.0`).
#[must_use]
pub fn is_heading_known(prior: AnchorPrior, opts: &AnchorSearchOpts) -> bool {
    opts.heading_step_deg <= 0.0 && prior.finite_heading_deg().is_some()
}

/// Estimator for uncalibrated camera focal length across cold anchor sweeps.
#[derive(Clone, Debug, PartialEq)]
pub struct FocalSweepEstimator {
    /// Nominal 35mm-equivalent focal length in millimeters.
    pub default_focal_mm: f64,
    /// Active 4-element multiplier array for the sweep.
    pub sweep_multipliers: [f64; 4],
    /// Locked refined focal length in full-frame pixels once estimated.
    pub locked_focal_px: Option<f64>,
}

impl Default for FocalSweepEstimator {
    fn default() -> Self {
        Self {
            default_focal_mm: DEFAULT_FOCAL_MM,
            sweep_multipliers: OBLIQUE_FOCAL_MULTIPLIERS,
            locked_focal_px: None,
        }
    }
}

impl FocalSweepEstimator {
    /// Create a focal estimator tuned for low-AGL (`<= 250 m`) nadir cameras when `low_agl_nadir` is true.
    #[must_use]
    pub fn for_regime(low_agl_nadir: bool) -> Self {
        Self {
            default_focal_mm: DEFAULT_FOCAL_MM,
            sweep_multipliers: if low_agl_nadir {
                LOW_AGL_FOCAL_MULTIPLIERS
            } else {
                OBLIQUE_FOCAL_MULTIPLIERS
            },
            locked_focal_px: None,
        }
    }

    /// Compute the baseline pixel focal length for `image_width_px` assuming a `36 mm` sensor width.
    #[must_use]
    pub fn baseline_focal_px(&self, image_width_px: u32) -> f64 {
        (self.default_focal_mm / 36.0) * f64::from(image_width_px.max(1))
    }

    /// Select and lock the focal length with minimum normalized angular reprojection score.
    pub fn select_multiplier(&mut self, scores: &[f64; 4], baseline_px: f64) -> f64 {
        let mut best_idx = 0_usize;
        let mut min_score = f64::INFINITY;
        for (i, &score) in scores.iter().enumerate() {
            if score < min_score {
                min_score = score;
                best_idx = i;
            }
        }
        let chosen = baseline_px * self.sweep_multipliers[best_idx];
        self.locked_focal_px = Some(chosen);
        chosen
    }
}

/// Resolve `(lat_deg, lon_deg, dem_m, agl_m)` from `prior` or `surface` origin.
#[must_use]
pub fn resolve_prior_origin(surface: &Surface, prior: AnchorPrior) -> (f64, f64, f64, f64) {
    let (lat, lon, base_alt) = if let Some((lat, lon)) = prior.lat_lon() {
        let dem = surface.sample_dem(lat, lon).unwrap_or(100.0);
        let alt = match prior.alt_msl_m.filter(|a| a.is_finite()) {
            Some(a) => a.max(dem + 25.0),
            None => (dem + 800.0).max(dem + 150.0),
        };
        (lat, lon, alt)
    } else {
        (
            surface.origin_lat,
            surface.origin_lon,
            surface.min_elev_m + 900.0,
        )
    };
    let dem_h = surface.sample_dem(lat, lon).unwrap_or(surface.min_elev_m);
    let agl = (base_alt - dem_h).max(0.0);
    (lat, lon, dem_h, agl)
}

fn resolve_pitches(attitude: Option<(f64, f64)>) -> Vec<(f64, f64)> {
    match attitude {
        Some((down_deg, roll_deg)) => vec![
            (down_deg, roll_deg),
            ((down_deg + 14.0).clamp(-30.0, 88.0), roll_deg),
            ((down_deg - 10.0).clamp(-30.0, 88.0), roll_deg),
            (down_deg, (roll_deg + 25.0).clamp(-65.0, 65.0)),
            (down_deg, (roll_deg - 25.0).clamp(-65.0, 65.0)),
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

fn resolve_headings(prior: AnchorPrior, heading_step_deg: f64) -> Vec<f64> {
    if let Some(dir) = prior.finite_heading_deg()
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

/// Generate candidate seed poses across compass headings, pitches, and altitude factors.
#[must_use]
pub fn generate_anchor_seeds(
    surface: &Surface,
    prior: AnchorPrior,
    opts: &AnchorSearchOpts,
    attitude: Option<(f64, f64)>,
) -> Vec<SeedPose> {
    let (lat, lon, dem_h, agl) = resolve_prior_origin(surface, prior);
    let pitches = resolve_pitches(attitude);
    let headings = resolve_headings(prior, opts.heading_step_deg);
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
        if is_heading_known(prior, opts) {
            let lead_hdg = headings[0];
            let eye0 = surface.local(
                lat,
                lon,
                dem_h + agl.max(25.0) * height_factors[0].max(0.25),
            );
            for &down_deg in &[28.0, 8.0, 48.0, -10.0] {
                if out.len() >= cap {
                    return out;
                }
                out.push((
                    eye0,
                    surface.camera_rotation(lat, lon, lead_hdg, down_deg, 0.0),
                ));
            }
        }
        let n_pitches = GRAZING_PITCH_SEEDS_DEG.len();
        for &roll_deg in &BANKED_ROLL_SEEDS_DEG {
            for &hf in height_factors {
                let eye = surface.local(lat, lon, dem_h + agl.max(25.0) * hf.max(0.25));
                for pass in 0..n_pitches {
                    for (h_idx, &hdg) in headings.iter().enumerate() {
                        if out.len() >= cap {
                            return out;
                        }
                        let down_deg = GRAZING_PITCH_SEEDS_DEG[(h_idx + pass) % n_pitches];
                        let cand = (
                            eye,
                            surface.camera_rotation(lat, lon, hdg, down_deg, roll_deg),
                        );
                        if !out.iter().any(|prev| same_seed_pose(prev, &cand)) {
                            out.push(cand);
                        }
                    }
                }
            }
        }
    } else {
        for &hf in height_factors {
            let eye = surface.local(lat, lon, dem_h + agl.max(25.0) * hf.max(0.25));
            for &(down_deg, roll_deg) in &pitches {
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

/// Prepend `pred_pose` (when supplied) before [`generate_anchor_seeds`].
#[must_use]
pub fn build_anchor_seeds(
    surface: &Surface,
    prior: AnchorPrior,
    opts: &AnchorSearchOpts,
    attitude: Option<(f64, f64)>,
    pred_pose: Option<SeedPose>,
) -> Vec<SeedPose> {
    pred_pose
        .into_iter()
        .chain(
            generate_anchor_seeds(surface, prior, opts, attitude)
                .into_iter()
                .take(opts.attempts.max(1)),
        )
        .collect()
}

pub(crate) fn same_seed_pose(a: &SeedPose, b: &SeedPose) -> bool {
    let dist = (Vector3::from(a.0) - Vector3::from(b.0)).norm();
    let cos_ang = (((a.1 * b.1.transpose()).trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
    dist <= 8.0 && cos_ang >= 0.995
}

pub(crate) fn push_unique_seed(out: &mut Vec<SeedPose>, cand: SeedPose, cap: usize) {
    if out.len() < cap && !out.iter().any(|prev| same_seed_pose(prev, &cand)) {
        out.push(cand);
    }
}

/// Build a stratified `(coarse_seeds, remaining_seeds)` split covering headings, altitudes, and ring offsets.
#[must_use]
pub fn build_stratified_coarse_anchor_seeds(
    surface: &Surface,
    prior: AnchorPrior,
    opts: &AnchorSearchOpts,
    attitude: Option<(f64, f64)>,
    pred_pose: Option<SeedPose>,
    coarse_cap: usize,
) -> (Vec<SeedPose>, Vec<SeedPose>) {
    let full_seeds = build_anchor_seeds(surface, prior, opts, attitude, pred_pose);
    let cap = coarse_cap.max(4);
    let mut coarse = Vec::with_capacity(cap);
    let (lat, lon, dem_h, agl) = resolve_prior_origin(surface, prior);
    let pitches = resolve_pitches(attitude);
    let headings = resolve_headings(prior, opts.heading_step_deg);
    let heading_unknown = !is_heading_known(prior, opts);
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
        if attitude.is_none() && agl > 250.0 {
            let eye0 = surface.local(lat, lon, dem_h + agl.max(25.0));
            for &hdg in &[0.0, 60.0, 120.0, 180.0, 240.0, 300.0] {
                let rot = surface.camera_rotation(lat, lon, hdg, 18.0, 0.0);
                push_unique_seed(&mut coarse, (eye0, rot), center_cap);
            }
        } else {
            for &off in &[0, 2, 4, 6, 8, 10] {
                push_unique_seed(&mut coarse, full_seeds[lead + off], center_cap);
            }
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
        } else if attitude.is_none() && agl > 250.0 {
            (22.0, 0.0)
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

/// Build eight compass oblique fallback seeds at `down_deg` when `agl > 250 m`.
#[must_use]
pub(crate) fn cardinal_oblique_fallbacks(
    surface: &Surface,
    prior: AnchorPrior,
    down_deg: f64,
) -> Vec<SeedPose> {
    let (lat, lon, dem_h, agl) = resolve_prior_origin(surface, prior);
    let eye = surface.local(lat, lon, dem_h + agl.max(25.0));
    [0.0, 180.0, 90.0, 270.0, 300.0, 60.0, 120.0, 240.0]
        .into_iter()
        .map(|hdg| (eye, surface.camera_rotation(lat, lon, hdg, down_deg, 0.0)))
        .collect()
}
