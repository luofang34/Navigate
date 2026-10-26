//! The search plan: which crop sizes, view directions and index cells a query uses.
//!
//! The plan reads the priors that the caller has (all optional): heading, height
//! above the ground (for example from a barometer or photo metadata) and the camera
//! angle below the horizon. From each pair of height and angle it estimates the
//! ground that the camera sees, and it selects the index crop size closest to that
//! ground. A prior that is not known gives more hypotheses, not a failure.
use crate::{Cell, Region, cells};
use serde::{Deserialize, Serialize};

/// Heights above the ground tried when the height is not known, metres.
const HEIGHTS_M: [f64; 4] = [100.0, 300.0, 700.0, 1_200.0];
/// Camera angles below the horizon tried when the angle is not known, degrees.
const PITCHES_DEG: [f64; 3] = [25.0, 55.0, 85.0];
/// Heading steps when the heading is not known.
const HEADINGS: usize = 8;
/// Farthest ground that a view keeps, metres, and in heights above the ground.
const RANGE_M: f64 = 6_000.0;
const RANGE_HEIGHTS: f64 = 10.0;
/// Largest region for the dense candidate grid of the locate pipeline, square km.
const DENSE_KM2: f64 = 4.0;

/// A heading prior: compass direction of the view, degrees clockwise from north.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeadingPrior {
    /// Most likely heading, degrees.
    pub deg: f64,
    /// One standard deviation, degrees.
    pub sigma_deg: f64,
}

/// A height prior: camera height above the ground, metres.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeightPrior {
    /// Most likely height, metres.
    pub m: f64,
    /// One standard deviation, metres.
    pub sigma_m: f64,
}

/// A camera angle prior: angle of the optical axis below the horizon, degrees.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PitchPrior {
    /// Most likely angle, degrees.
    pub deg: f64,
    /// One standard deviation, degrees.
    pub sigma_deg: f64,
}

/// What the caller knows about one photo or camera frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchRequest {
    /// Where the camera can be.
    pub region: Region,
    /// Image width and height, pixels.
    pub image_px: [u32; 2],
    /// Focal length, pixels of that image.
    pub focal_px: f64,
    /// Heading prior, if known.
    pub heading: Option<HeadingPrior>,
    /// Height prior, if known.
    pub height: Option<HeightPrior>,
    /// Camera angle prior, if known.
    pub pitch: Option<PitchPrior>,
}

/// The ground that one view sees on flat ground, metres along the heading from the
/// point below the camera.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Footprint {
    /// Nearest ground in view; negative when it is behind the point below the camera.
    pub near_m: f64,
    /// Farthest ground in view, limited to the range that a projection keeps.
    pub far_m: f64,
    /// Width of the view at the middle of the footprint.
    pub width_m: f64,
}

impl Footprint {
    /// The view of a camera at `height_m` with its axis `pitch_deg` below the horizon.
    pub fn of(image_px: [u32; 2], focal_px: f64, height_m: f64, pitch_deg: f64) -> Self {
        let half_v = (f64::from(image_px[1]) / 2.0 / focal_px)
            .atan()
            .to_degrees();
        let half_h = (f64::from(image_px[0]) / 2.0 / focal_px).atan();
        let range = RANGE_M.min(RANGE_HEIGHTS * height_m);
        let ground = |depression_deg: f64| {
            if depression_deg <= 0.5 {
                range
            } else {
                (height_m / depression_deg.to_radians().tan()).min(range)
            }
        };
        let near_m = ground((pitch_deg + half_v).min(179.0));
        let far_m = ground(pitch_deg - half_v);
        let slant = height_m / pitch_deg.clamp(1.0, 90.0).to_radians().sin();
        Self {
            near_m,
            far_m,
            width_m: 2.0 * slant.min(range) * half_h.tan(),
        }
    }

    /// The larger of the depth and the width: the crop side that holds the view.
    pub fn extent_m(&self) -> f64 {
        (self.far_m - self.near_m).max(self.width_m)
    }
}

/// The queries for one index crop size.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScalePlan {
    /// Crop side, metres.
    pub side_m: f64,
    /// Heights and camera angles to project the photo with: `(height_m, pitch_deg)`.
    pub views: Vec<(f64, f64)>,
    /// Headings to project the photo with, degrees.
    pub headings_deg: Vec<f64>,
    /// Index cells to rank: the region grown by the farthest ground in view.
    pub cells: Vec<Cell>,
    /// Ranked places to verify, best first.
    pub verify: usize,
}

/// What to do when the index does not serve the request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Fallback {
    /// The index serves every planned query.
    None,
    /// Use the dense candidate grid of the locate pipeline: the region is small.
    DenseGrid,
    /// The index does not cover the region and the region is too large for the dense
    /// grid.
    Uncovered {
        /// Reason for a person.
        reason: String,
    },
}

/// The plan of one search.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchPlan {
    /// Queries by crop size, most likely first.
    pub scales: Vec<ScalePlan>,
    /// What to do for the part that the index does not serve.
    pub fallback: Fallback,
    /// Model inputs that the plan needs: views × headings over all scales.
    pub canvases: usize,
}

fn guesses(prior: Option<(f64, f64)>, defaults: &[f64]) -> Vec<f64> {
    match prior {
        // A wide prior gives the guess and one standard deviation each side.
        Some((value, sigma)) if sigma > 0.25 * value.abs().max(1.0) => {
            vec![value - sigma, value, value + sigma]
        }
        Some((value, _)) => vec![value],
        None => defaults.to_vec(),
    }
}

/// Headings to try: one for a narrow prior, a fan for a wide one, all for none.
fn headings(prior: Option<HeadingPrior>) -> Vec<f64> {
    let step = 360.0 / HEADINGS as f64;
    match prior {
        Some(h) if h.sigma_deg <= step / 2.0 => vec![h.deg.rem_euclid(360.0)],
        Some(h) => {
            let reach = (2.0 * h.sigma_deg).min(180.0);
            let n = (2.0 * reach / step).ceil() as i32;
            (-n / 2..=n / 2)
                .map(|k| (h.deg + f64::from(k) * step).rem_euclid(360.0))
                .collect()
        }
        None => (0..HEADINGS).map(|k| k as f64 * step).collect(),
    }
}

/// Plan a search with an index that has `scales` crop sizes and serves `covered`
/// cells for each of them. Pass no scales for a region without an index.
pub fn plan(request: &SearchRequest, scales: &[(f64, Vec<Cell>)]) -> SearchPlan {
    let heights = guesses(request.height.map(|h| (h.m, h.sigma_m)), &HEIGHTS_M);
    let pitches = guesses(request.pitch.map(|p| (p.deg, p.sigma_deg)), &PITCHES_DEG);
    let headings_deg = headings(request.heading);
    let mut by_scale: Vec<ScalePlan> = Vec::new();
    for &height_m in heights.iter().filter(|h| **h > 1.0) {
        for &pitch_deg in pitches.iter().filter(|p| **p > 0.0) {
            let footprint = Footprint::of(
                request.image_px,
                request.focal_px,
                height_m,
                pitch_deg.min(90.0),
            );
            let Some((side_m, covered)) = scales.iter().min_by(|a, b| {
                (footprint.extent_m() / a.0)
                    .ln()
                    .abs()
                    .total_cmp(&(footprint.extent_m() / b.0).ln().abs())
            }) else {
                continue;
            };
            let wanted = cells(&request.region.grown(footprint.far_m.max(0.0)));
            let served: Vec<Cell> = wanted.into_iter().filter(|c| covered.contains(c)).collect();
            if served.is_empty() {
                continue;
            }
            match by_scale.iter_mut().find(|s| s.side_m == *side_m) {
                Some(scale) => {
                    scale.views.push((height_m, pitch_deg));
                    for c in served {
                        if !scale.cells.contains(&c) {
                            scale.cells.push(c);
                        }
                    }
                }
                None => by_scale.push(ScalePlan {
                    side_m: *side_m,
                    views: vec![(height_m, pitch_deg)],
                    headings_deg: headings_deg.clone(),
                    cells: served,
                    verify: 0,
                }),
            }
        }
    }
    let area_km2 = request.region.area_km2();
    for scale in &mut by_scale {
        scale.cells.sort();
        // A larger area holds more look-alike places, and each unknown heading adds
        // look-alikes: verify more of the ranked list.
        let base = if area_km2 < 50.0 {
            10
        } else if area_km2 < 1_000.0 {
            20
        } else {
            50
        };
        scale.verify = if scale.headings_deg.len() > 1 {
            base * 5 / 2
        } else {
            base
        };
    }
    by_scale.sort_by_key(|s| std::cmp::Reverse(s.views.len()));
    let canvases = by_scale
        .iter()
        .map(|s| s.views.len() * s.headings_deg.len())
        .sum();
    let fallback = if !by_scale.is_empty() {
        Fallback::None
    } else if area_km2 <= DENSE_KM2 {
        Fallback::DenseGrid
    } else {
        Fallback::Uncovered {
            reason: format!(
                "no index covers this region, and {area_km2:.0} km² is too large for the dense search (at most {DENSE_KM2} km²)"
            ),
        }
    };
    SearchPlan {
        scales: by_scale,
        fallback,
        canvases,
    }
}

#[cfg(test)]
mod tests;
