//! Acquisition of a position that a frame-sized search cannot reach: the
//! frame, laid on the ground below the believed eye, is correlated with an
//! orthographic raster of the map about the belief.
//!
//! The tracker of a reference view reaches a few hundred metres; a belief
//! kilometres wide, as after a long gap of fixes, needs the offset found
//! first. Both rasters hold the mean luma of square ground cells, east and
//! north about their centre. The search scores each offset by normalized
//! cross-correlation, first at twice the cell over the whole radius, then
//! at the cell about the best coarse peaks, and finds an offset only when
//! its peak stands clear of every other: repeated ground gives a peak per
//! repetition, and the search then says so instead of choosing one.

use image::GrayImage;
use nalgebra::{UnitQuaternion, Vector2, Vector3};

use crate::{CameraModel, VisualError};

/// Sub-samples on a side of a cell: the mean of these is the cell.
const SUBSAMPLES: u32 = 4;

/// Half side, in cells, of the fine search about each coarse peak.
const REFINE_CELLS: i64 = 2;

/// Smallest share of the frame cells that an offset must overlap with
/// known map cells to be scored.
const MIN_OVERLAP: f64 = 0.9;

/// A square raster of ground cells about a centre: `side` cells east by
/// `side` cells north, row-major from the south-west corner. An unknown
/// cell is NaN.
#[derive(Clone, Debug)]
pub struct GroundRaster {
    cell_m: f64,
    half_cells: usize,
    values: Vec<f32>,
}

impl GroundRaster {
    /// The raster of `(2 half_cells + 1)²` cells of `cell_m` metres whose
    /// values are the means of `luma` over a grid of sub-samples in each
    /// cell; `luma` takes the east and north offset from the centre, in
    /// metres, and gives none where it does not know the ground. A cell
    /// with an unknown sub-sample is unknown.
    ///
    /// # Errors
    /// Rejects a cell that is not finite and positive.
    pub fn from_samples(
        cell_m: f64,
        half_cells: usize,
        mut luma: impl FnMut(Vector2<f64>) -> Option<f64>,
    ) -> Result<Self, VisualError> {
        if !cell_m.is_finite() || cell_m <= 0.0 {
            return Err(VisualError::Invalid {
                field: "ground raster cell",
            });
        }
        let side = 2 * half_cells + 1;
        let half = half_cells as f64;
        let step = 1.0 / f64::from(SUBSAMPLES);
        let mut values = Vec::with_capacity(side * side);
        for row in 0..side {
            for col in 0..side {
                let corner = Vector2::new(col as f64 - half - 0.5, row as f64 - half - 0.5);
                let mut sum = 0.0;
                let mut known = true;
                for sy in 0..SUBSAMPLES {
                    for sx in 0..SUBSAMPLES {
                        let at =
                            corner + Vector2::new(f64::from(sx) + 0.5, f64::from(sy) + 0.5) * step;
                        match luma(at * cell_m) {
                            Some(v) if v.is_finite() => sum += v,
                            _ => known = false,
                        }
                    }
                }
                values.push(if known {
                    (sum * step * step) as f32
                } else {
                    f32::NAN
                });
            }
        }
        Ok(Self {
            cell_m,
            half_cells,
            values,
        })
    }

    /// The ground below a camera at `height_m` above flat ground, with the
    /// eye-to-local `orientation` (eye axes right, up, back; local axes
    /// east, north, up), as `camera` sees it in `image`: the raster is
    /// centred on the point below the eye, so it holds no horizontal
    /// position.
    ///
    /// # Errors
    /// Rejects an image of another size than `camera`, a height that is
    /// not finite and positive, and an invalid cell.
    pub fn of_frame(
        image: &GrayImage,
        camera: &CameraModel,
        orientation: &UnitQuaternion<f64>,
        height_m: f64,
        (cell_m, half_cells): (f64, usize),
    ) -> Result<Self, VisualError> {
        camera.validate()?;
        if image.dimensions() != (camera.width, camera.height) {
            return Err(VisualError::Dimensions {
                query: image.dimensions(),
                reference: (camera.width, camera.height),
            });
        }
        if !height_m.is_finite() || height_m <= 0.0 {
            return Err(VisualError::Invalid {
                field: "acquisition height",
            });
        }
        let to_eye = orientation.inverse();
        Self::from_samples(cell_m, half_cells, |ground| {
            let ray = to_eye * Vector3::new(ground.x, ground.y, -height_m);
            if ray.z >= 0.0 {
                return None;
            }
            let u = camera.cx + camera.fx * ray.x / -ray.z;
            let v = camera.cy - camera.fy * ray.y / -ray.z;
            bilinear(image, u, v)
        })
    }

    /// Side of a cell, metres.
    #[must_use]
    pub fn cell_m(&self) -> f64 {
        self.cell_m
    }

    /// The raster at twice the cell: the coarse cell at `(e, n)` is the
    /// mean of the cells at `(2e, 2n)` to `(2e + 1, 2n + 1)`, unknown when
    /// one of them is. Every raster takes the same four, so a coarse
    /// offset of `(e, n)` is a fine offset of `(2e, 2n)`.
    fn coarse(&self) -> Self {
        let half = (self.half_cells / 2) as i64;
        let side = (2 * half + 1) as usize;
        let mut values = Vec::with_capacity(side * side);
        for north in -half..=half {
            for east in -half..=half {
                let (e, n) = (2 * east, 2 * north);
                let sum: f32 = [(e, n), (e + 1, n), (e, n + 1), (e + 1, n + 1)]
                    .iter()
                    .map(|&(e, n)| self.at(e, n).unwrap_or(f32::NAN))
                    .sum();
                values.push(sum * 0.25);
            }
        }
        Self {
            cell_m: 2.0 * self.cell_m,
            half_cells: half as usize,
            values,
        }
    }

    /// The mean luma of the cell `east`, `north` cells from the centre;
    /// none for an unknown cell or one outside the raster.
    #[must_use]
    pub fn at(&self, east: i64, north: i64) -> Option<f32> {
        let half = self.half_cells as i64;
        let side = 2 * half + 1;
        let (col, row) = (east + half, north + half);
        if col < 0 || row < 0 || col >= side || row >= side {
            return None;
        }
        self.values
            .get((row * side + col) as usize)
            .copied()
            .filter(|v| v.is_finite())
    }

    /// The known cells as offsets from the centre with their values.
    fn known(&self) -> Vec<(i64, i64, f32)> {
        let half = self.half_cells as i64;
        let side = 2 * half + 1;
        self.values
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, v)| (i as i64 % side - half, i as i64 / side - half, *v))
            .collect()
    }
}

/// Bilinear luma at `(u, v)`, none outside the image.
fn bilinear(image: &GrayImage, u: f64, v: f64) -> Option<f64> {
    let (w, h) = image.dimensions();
    if !(u >= 0.0 && v >= 0.0 && u < f64::from(w - 1) && v < f64::from(h - 1)) {
        return None;
    }
    let (x, y) = (u.floor() as u32, v.floor() as u32);
    let (fx, fy) = (u - u.floor(), v - v.floor());
    let at = |x: u32, y: u32| image.get_pixel_checked(x, y).map(|p| f64::from(p.0[0]));
    let top = at(x, y)? * (1.0 - fx) + at(x + 1, y)? * fx;
    let bottom = at(x, y + 1)? * (1.0 - fx) + at(x + 1, y + 1)? * fx;
    Some(top * (1.0 - fy) + bottom * fy)
}

/// What the search requires of a peak.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AcquisitionConfig {
    /// Smallest correlation of an offset that is found.
    pub min_score: f64,
    /// Smallest lead of the best peak over the best peak farther than
    /// `exclusion_m` from it.
    pub min_margin: f64,
    /// Distance beyond which another peak is another place, metres.
    pub exclusion_m: f64,
    /// Coarse peaks searched again at the cell.
    pub peaks: usize,
}

impl Default for AcquisitionConfig {
    /// Thresholds measured over farmland and over regular and irregular
    /// town blocks from 1760 m: a true peak scores 0.43 to 0.98 with a
    /// lead of at least 0.2; a wrong one leads by a few hundredths.
    fn default() -> Self {
        Self {
            min_score: 0.3,
            min_margin: 0.15,
            exclusion_m: 300.0,
            peaks: 8,
        }
    }
}

/// One scored offset of the frame on the map: east and north from the
/// centre of the map raster to the point below the eye, metres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Peak {
    /// East and north offset, metres.
    pub offset_m: Vector2<f64>,
    /// Normalized cross-correlation, `[-1, 1]`.
    pub score: f64,
}

/// What the search found.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Acquisition {
    /// One offset stands clear of every other.
    Found(Peak),
    /// Another place scores too close to the best: the ground repeats, or
    /// the frame is not distinctive at the cell.
    Ambiguous {
        /// The best peak.
        best: Peak,
        /// The best peak farther than the exclusion distance.
        runner_up: Peak,
    },
    /// No offset correlates enough, or none overlaps the map.
    Weak(Option<Peak>),
}

/// Searches `map` for `frame` at the offsets within `radius_m` of the map
/// centre.
///
/// # Errors
/// Rejects rasters of different cells, a radius that is not finite and
/// non-negative, and an invalid configuration.
pub fn acquire(
    frame: &GroundRaster,
    map: &GroundRaster,
    radius_m: f64,
    config: &AcquisitionConfig,
) -> Result<Acquisition, VisualError> {
    let valid = config.min_score.is_finite()
        && config.min_margin.is_finite()
        && config.exclusion_m.is_finite()
        && config.exclusion_m >= 0.0
        && config.peaks > 0;
    if !valid || !radius_m.is_finite() || radius_m < 0.0 {
        return Err(VisualError::Invalid {
            field: "acquisition search",
        });
    }
    if (frame.cell_m - map.cell_m).abs() > 1e-9 * map.cell_m {
        return Err(VisualError::Invalid {
            field: "acquisition cells differ",
        });
    }
    let (coarse_frame, coarse_map) = (frame.coarse(), map.coarse());
    let reach = (radius_m / coarse_map.cell_m).ceil() as i64;
    let template = coarse_frame.known();
    let mut scored = Vec::new();
    for north in -reach..=reach {
        for east in -reach..=reach {
            if let Some(score) = ncc(&template, &coarse_map, (east, north)) {
                scored.push(((east, north), score));
            }
        }
    }
    let seeds = local_maxima(&scored, reach, config.peaks);
    let template = frame.known();
    let peaks: Vec<Peak> = seeds
        .iter()
        .filter_map(|&(east, north)| refine(&template, map, (2 * east, 2 * north)))
        .collect();
    Ok(decide(&peaks, config))
}

/// The coarse offsets that score highest among their eight neighbours,
/// best first, at most `count`.
fn local_maxima(scored: &[((i64, i64), f64)], reach: i64, count: usize) -> Vec<(i64, i64)> {
    let side = 2 * reach + 1;
    let mut grid = vec![f64::NEG_INFINITY; (side * side) as usize];
    for &((east, north), score) in scored {
        if let Some(cell) = grid.get_mut(((north + reach) * side + east + reach) as usize) {
            *cell = score;
        }
    }
    let at = |east: i64, north: i64| {
        if east.abs() > reach || north.abs() > reach {
            f64::NEG_INFINITY
        } else {
            grid.get(((north + reach) * side + east + reach) as usize)
                .copied()
                .unwrap_or(f64::NEG_INFINITY)
        }
    };
    let mut maxima: Vec<((i64, i64), f64)> = scored
        .iter()
        .filter(|&&((east, north), score)| {
            (-1..=1).all(|dn| {
                (-1..=1).all(|de| (de, dn) == (0, 0) || at(east + de, north + dn) <= score)
            })
        })
        .copied()
        .collect();
    maxima.sort_by(|a, b| b.1.total_cmp(&a.1));
    maxima.into_iter().take(count).map(|(at, _)| at).collect()
}

/// The best fine offset within [`REFINE_CELLS`] of `centre`.
fn refine(template: &[(i64, i64, f32)], map: &GroundRaster, centre: (i64, i64)) -> Option<Peak> {
    let mut best: Option<Peak> = None;
    for north in centre.1 - REFINE_CELLS..=centre.1 + REFINE_CELLS {
        for east in centre.0 - REFINE_CELLS..=centre.0 + REFINE_CELLS {
            if let Some(score) = ncc(template, map, (east, north))
                && best.is_none_or(|b| score > b.score)
            {
                best = Some(Peak {
                    offset_m: Vector2::new(east as f64, north as f64) * map.cell_m,
                    score,
                });
            }
        }
    }
    best
}

/// The decision over the refined peaks.
fn decide(peaks: &[Peak], config: &AcquisitionConfig) -> Acquisition {
    let Some(best) = peaks
        .iter()
        .copied()
        .max_by(|a, b| a.score.total_cmp(&b.score))
    else {
        return Acquisition::Weak(None);
    };
    if best.score < config.min_score {
        return Acquisition::Weak(Some(best));
    }
    let runner_up = peaks
        .iter()
        .copied()
        .filter(|p| (p.offset_m - best.offset_m).norm() > config.exclusion_m)
        .max_by(|a, b| a.score.total_cmp(&b.score));
    match runner_up {
        Some(runner_up) if best.score - runner_up.score < config.min_margin => {
            Acquisition::Ambiguous { best, runner_up }
        }
        _ => Acquisition::Found(best),
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

#[cfg(test)]
mod tests;
