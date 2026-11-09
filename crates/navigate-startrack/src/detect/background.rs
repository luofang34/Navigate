//! A robust estimate of the sky background and the pixel noise.

/// Scale from the median absolute deviation to the standard deviation of
/// Gaussian noise.
const MAD_TO_SIGMA: f64 = 1.482_6;

/// The background level and the noise of each tile of an image. Values
/// between tile centres come from a bilinear interpolation, so a smooth
/// gradient of the sky does not make a step at a tile edge.
#[derive(Debug, Clone)]
pub(super) struct BackgroundGrid {
    tiles_x: usize,
    tiles_y: usize,
    tile_px: f64,
    /// Median of each tile, row after row.
    level: Vec<f64>,
    /// Noise of each tile, row after row.
    sigma: Vec<f64>,
}

impl BackgroundGrid {
    /// The grid of an image of `width` by `height` samples with square
    /// tiles of `tile_px` samples. The median and the median absolute
    /// deviation ignore the few star pixels in a tile. The noise is at
    /// least `min_sigma`.
    pub(super) fn estimate(
        samples: &[f32],
        width: usize,
        height: usize,
        tile_px: usize,
        min_sigma: f64,
    ) -> Self {
        let tiles_x = width.div_ceil(tile_px);
        let tiles_y = height.div_ceil(tile_px);
        let mut level = Vec::with_capacity(tiles_x * tiles_y);
        let mut sigma = Vec::with_capacity(tiles_x * tiles_y);
        let mut scratch = Vec::with_capacity(tile_px * tile_px);
        for ty in 0..tiles_y {
            for tx in 0..tiles_x {
                scratch.clear();
                let rows = ty * tile_px..((ty + 1) * tile_px).min(height);
                let cols = tx * tile_px..((tx + 1) * tile_px).min(width);
                for y in rows {
                    let start = y * width;
                    if let Some(row) = samples.get(start + cols.start..start + cols.end) {
                        scratch.extend_from_slice(row);
                    }
                }
                let median = median_of(&mut scratch);
                for value in &mut scratch {
                    *value = (*value - median).abs();
                }
                let mad = median_of(&mut scratch);
                level.push(f64::from(median));
                sigma.push((f64::from(mad) * MAD_TO_SIGMA).max(min_sigma));
            }
        }
        Self {
            tiles_x,
            tiles_y,
            tile_px: tile_px as f64,
            level,
            sigma,
        }
    }

    /// The background level and the noise at the pixel `(x, y)`.
    pub(super) fn at(&self, x: usize, y: usize) -> (f64, f64) {
        let (x0, x1, fx) = Self::axis(x, self.tile_px, self.tiles_x);
        let (y0, y1, fy) = Self::axis(y, self.tile_px, self.tiles_y);
        let blend = |grid: &[f64]| {
            let get = |tx: usize, ty: usize| grid.get(ty * self.tiles_x + tx).copied();
            let top = lerp(get(x0, y0), get(x1, y0), fx);
            let bottom = lerp(get(x0, y1), get(x1, y1), fx);
            lerp(Some(top), Some(bottom), fy)
        };
        (blend(&self.level), blend(&self.sigma))
    }

    /// The two tile indices around one pixel coordinate and the weight of
    /// the second. Outside the outer tile centres the value is constant.
    fn axis(pixel: usize, tile_px: f64, tiles: usize) -> (usize, usize, f64) {
        let position = ((pixel as f64 + 0.5) / tile_px - 0.5).max(0.0);
        let last = tiles.saturating_sub(1);
        let first = (position.floor() as usize).min(last);
        let second = (first + 1).min(last);
        let weight = if second == first {
            0.0
        } else {
            (position - first as f64).clamp(0.0, 1.0)
        };
        (first, second, weight)
    }
}

/// The linear blend of two values. A missing value gives zero weight to its
/// side, which does not occur for indices inside the grid.
fn lerp(a: Option<f64>, b: Option<f64>, weight: f64) -> f64 {
    match (a, b) {
        (Some(a), Some(b)) => a + (b - a) * weight,
        (Some(v), None) | (None, Some(v)) => v,
        (None, None) => 0.0,
    }
}

/// The median of `values`, which the function reorders. An empty slice
/// gives zero.
fn median_of(values: &mut [f32]) -> f32 {
    let middle = values.len() / 2;
    if values.is_empty() {
        return 0.0;
    }
    let (_, median, _) = values.select_nth_unstable_by(middle, f32::total_cmp);
    *median
}
