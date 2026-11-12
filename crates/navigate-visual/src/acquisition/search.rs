//! The search of one frame: the coarse correlation over the radius at each
//! rotation, in slices of a budget of cell pairs, then the refinement of
//! the best coarse peaks and the decision.

use std::sync::Arc;

use nalgebra::Vector2;

use super::{Acquisition, AcquisitionConfig, GroundRaster, Peak};
use crate::VisualError;

/// Half side, in cells, of the fine search about each coarse peak.
const REFINE_CELLS: i64 = 2;

/// Smallest share of the frame cells that an offset must overlap with
/// known map cells to be scored.
const MIN_OVERLAP: f64 = 0.9;

/// The map raster at its cell and at twice its cell. A host can keep it
/// for the searches of several frames.
#[derive(Clone, Debug)]
pub struct MapPyramid {
    fine: GroundRaster,
    coarse: GroundRaster,
}

impl MapPyramid {
    /// The pyramid of `raster`.
    #[must_use]
    pub fn new(raster: GroundRaster) -> Self {
        let coarse = raster.coarse();
        Self {
            fine: raster,
            coarse,
        }
    }

    /// The raster at its cell.
    #[must_use]
    pub fn raster(&self) -> &GroundRaster {
        &self.fine
    }
}

/// One laid frame: its rotation and its known cells at the cell and at
/// twice the cell.
#[derive(Clone, Debug)]
struct Laid {
    rotation_rad: f64,
    fine: Vec<(i64, i64, f32)>,
    coarse: Vec<(i64, i64, f32)>,
}

/// A search that can be advanced in slices. Each slice scores coarse
/// offsets until it has compared its budget of cell pairs; the last one
/// also refines and decides.
#[derive(Clone, Debug)]
pub struct AcquisitionSearch {
    laid: Vec<Laid>,
    map: Arc<MapPyramid>,
    /// The centre of the searched disc, coarse cells from the map centre,
    /// the half side of the square about it, and its radius.
    centre: (i64, i64),
    reach: i64,
    radius_cells: f64,
    /// The next coarse offset to score, counted over the rotations.
    next: usize,
    /// Score of each coarse offset of each rotation, row-major; negative
    /// infinity where the frame and the map do not overlap enough.
    scores: Vec<f64>,
    config: AcquisitionConfig,
    pairs: u64,
}

impl AcquisitionSearch {
    /// The search for the frames laid at each rotation, `frames` as
    /// `(rotation_rad, raster)`, within `radius_m` of `centre_m` east and
    /// north of the map centre.
    ///
    /// # Errors
    /// Rejects no frames, rasters of different cells, a radius or a centre
    /// that is not finite, and an invalid configuration.
    pub fn new(
        frames: Vec<(f64, GroundRaster)>,
        map: Arc<MapPyramid>,
        centre_m: Vector2<f64>,
        radius_m: f64,
        config: &AcquisitionConfig,
    ) -> Result<Self, VisualError> {
        let valid = config.min_score.is_finite()
            && config.min_margin.is_finite()
            && config.exclusion_m.is_finite()
            && config.exclusion_m >= 0.0
            && config.peaks > 0;
        let place =
            radius_m.is_finite() && radius_m >= 0.0 && centre_m.iter().all(|v| v.is_finite());
        if !valid || !place || frames.is_empty() {
            return Err(VisualError::Invalid {
                field: "acquisition search",
            });
        }
        let cell_m = map.fine.cell_m;
        if frames
            .iter()
            .any(|(r, f)| !r.is_finite() || (f.cell_m - cell_m).abs() > 1e-9 * cell_m)
        {
            return Err(VisualError::Invalid {
                field: "acquisition cells differ",
            });
        }
        let coarse_m = map.coarse.cell_m;
        let laid: Vec<Laid> = frames
            .into_iter()
            .map(|(rotation_rad, raster)| Laid {
                rotation_rad,
                fine: raster.known(),
                coarse: raster.coarse().known(),
            })
            .collect();
        let reach = (radius_m / coarse_m).ceil() as i64;
        let side = (2 * reach + 1) as usize;
        Ok(Self {
            scores: vec![f64::NEG_INFINITY; laid.len() * side * side],
            laid,
            map,
            centre: (
                (centre_m.x / coarse_m).round() as i64,
                (centre_m.y / coarse_m).round() as i64,
            ),
            reach,
            radius_cells: radius_m / coarse_m,
            next: 0,
            config: *config,
            pairs: 0,
        })
    }

    /// Cell pairs that the whole coarse correlation compares: the measure
    /// of its cost.
    #[must_use]
    pub fn pairs_total(&self) -> u64 {
        let inside = (-self.reach..=self.reach)
            .flat_map(|n| (-self.reach..=self.reach).map(move |e| (e, n)))
            .filter(|&(e, n)| self.inside(e, n))
            .count() as u64;
        self.laid.iter().map(|l| l.coarse.len() as u64).sum::<u64>() * inside
    }

    /// Whether the coarse offset `(east, north)` from the centre is in the
    /// searched disc.
    fn inside(&self, east: i64, north: i64) -> bool {
        (east as f64).hypot(north as f64) <= self.radius_cells
    }

    /// Cell pairs compared so far.
    #[must_use]
    pub const fn pairs_done(&self) -> u64 {
        self.pairs
    }

    /// Scores coarse offsets until `budget` cell pairs are compared, at
    /// least one; when every offset is scored, refines and gives what the
    /// search found.
    pub fn advance(&mut self, budget: u64) -> Option<Acquisition> {
        let side = (2 * self.reach + 1) as usize;
        let per_rotation = side * side;
        let mut spent = 0_u64;
        while self.next < self.scores.len() && (spent == 0 || spent < budget) {
            let (rotation, at) = (self.next / per_rotation, self.next % per_rotation);
            let (east, north) = (
                (at % side) as i64 - self.reach,
                (at / side) as i64 - self.reach,
            );
            let shift = (self.centre.0 + east, self.centre.1 + north);
            if let Some(laid) = self.laid.get(rotation).filter(|_| self.inside(east, north)) {
                let score = ncc(&laid.coarse, &self.map.coarse, shift);
                if let (Some(score), Some(slot)) = (score, self.scores.get_mut(self.next)) {
                    *slot = score;
                }
                let pairs = laid.coarse.len() as u64;
                spent = spent.saturating_add(pairs.max(1));
                self.pairs = self.pairs.saturating_add(pairs);
            }
            self.next = self.next.saturating_add(1);
        }
        (self.next >= self.scores.len()).then(|| self.decide())
    }

    /// The refined peaks of every rotation, and the decision over them.
    fn decide(&self) -> Acquisition {
        let side = (2 * self.reach + 1) as usize;
        let mut peaks = Vec::new();
        for (index, laid) in self.laid.iter().enumerate() {
            let scores = self
                .scores
                .get(index * side * side..(index + 1) * side * side)
                .unwrap_or_default();
            for (east, north) in local_maxima(scores, self.reach, self.config.peaks) {
                let coarse = (self.centre.0 + east, self.centre.1 + north);
                if let Some((offset, score)) = refine(&laid.fine, &self.map.fine, coarse) {
                    peaks.push(Peak {
                        offset_m: offset,
                        rotation_rad: laid.rotation_rad,
                        score,
                    });
                }
            }
        }
        decide(peaks, &self.config)
    }
}

/// The coarse offsets, from the centre of the search, that score highest
/// among their eight neighbours, best first, at most `count`.
fn local_maxima(scores: &[f64], reach: i64, count: usize) -> Vec<(i64, i64)> {
    let side = 2 * reach + 1;
    let at = |east: i64, north: i64| {
        if east.abs() > reach || north.abs() > reach {
            f64::NEG_INFINITY
        } else {
            scores
                .get(((north + reach) * side + east + reach) as usize)
                .copied()
                .unwrap_or(f64::NEG_INFINITY)
        }
    };
    let mut maxima = Vec::new();
    for north in -reach..=reach {
        for east in -reach..=reach {
            let score = at(east, north);
            let highest = score.is_finite()
                && (-1..=1).all(|dn| {
                    (-1..=1).all(|de| (de, dn) == (0, 0) || at(east + de, north + dn) <= score)
                });
            if highest {
                maxima.push(((east, north), score));
            }
        }
    }
    maxima.sort_by(|a, b| b.1.total_cmp(&a.1));
    maxima.into_iter().take(count).map(|(at, _)| at).collect()
}

/// The best fine offset, metres from the map centre, within
/// [`REFINE_CELLS`] of the coarse offset `coarse`, and its score.
fn refine(
    template: &[(i64, i64, f32)],
    map: &GroundRaster,
    coarse: (i64, i64),
) -> Option<(Vector2<f64>, f64)> {
    let centre = (2 * coarse.0, 2 * coarse.1);
    let mut best: Option<(Vector2<f64>, f64)> = None;
    for north in centre.1 - REFINE_CELLS..=centre.1 + REFINE_CELLS {
        for east in centre.0 - REFINE_CELLS..=centre.0 + REFINE_CELLS {
            if let Some(score) = ncc(template, map, (east, north))
                && best.is_none_or(|(_, b)| score > b)
            {
                best = Some((Vector2::new(east as f64, north as f64) * map.cell_m, score));
            }
        }
    }
    best
}

/// The decision over the refined peaks: each place is its best peak, and
/// a place counts against the best when it is farther than the exclusion
/// distance and scores within the margin.
fn decide(mut peaks: Vec<Peak>, config: &AcquisitionConfig) -> Acquisition {
    peaks.sort_by(|a, b| b.score.total_cmp(&a.score));
    let Some(best) = peaks.first().copied() else {
        return Acquisition::Weak(None);
    };
    if best.score < config.min_score {
        return Acquisition::Weak(Some(best));
    }
    let mut places: Vec<Peak> = Vec::new();
    for peak in peaks {
        let new_place = places
            .iter()
            .all(|p| (p.offset_m - peak.offset_m).norm() > config.exclusion_m);
        if new_place {
            places.push(peak);
        }
    }
    let contenders: Vec<Peak> = places
        .into_iter()
        .filter(|p| p.score >= config.min_score && best.score - p.score < config.min_margin)
        .collect();
    if contenders.len() > 1 {
        Acquisition::Ambiguous(contenders)
    } else {
        Acquisition::Found(best)
    }
}

/// Normalized cross-correlation of `template` with `map` at `shift`
/// cells, over the cells that both know; none when they overlap on less
/// than [`MIN_OVERLAP`] of the template or either is flat there.
fn ncc(template: &[(i64, i64, f32)], map: &GroundRaster, shift: (i64, i64)) -> Option<f64> {
    let (mut n, mut st, mut sm, mut stt, mut smm, mut stm) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for &(east, north, t) in template {
        if let Some(m) = map.at(east + shift.0, north + shift.1) {
            let (t, m) = (f64::from(t), f64::from(m));
            n += 1.0;
            st += t;
            sm += m;
            stt += t * t;
            smm += m * m;
            stm += t * m;
        }
    }
    if n < MIN_OVERLAP * template.len() as f64 || n < 2.0 {
        return None;
    }
    let vt = n * stt - st * st;
    let vm = n * smm - sm * sm;
    (vt > 0.0 && vm > 0.0).then(|| (n * stm - st * sm) / (vt * vm).sqrt())
}
