//! The centroid window of a pixel group and the covariance of its centre.

use nalgebra::{Matrix2, Vector2};

use super::{DetectConfig, Field, StarCentroid};

/// The pixels of one centroid window, with a mark per image pixel so that
/// the window holds each pixel once.
pub(super) struct Window {
    mark: Vec<u32>,
    stamp: u32,
    pixels: Vec<usize>,
}

/// The weighted sums of a window.
#[derive(Debug, Default)]
struct Sums {
    weight: f64,
    weight_x: f64,
    weight_y: f64,
    variance: f64,
    variance_x: f64,
    variance_y: f64,
    variance_xx: f64,
    variance_yy: f64,
    variance_xy: f64,
}

impl Window {
    /// The window state of an image with `pixels` pixels.
    pub(super) fn new(pixels: usize) -> Self {
        Self {
            mark: vec![0; pixels],
            stamp: 0,
            pixels: Vec::new(),
        }
    }

    /// The star of `shape` with the centre, the flux, and the covariance of
    /// the window around `group`. The window leaves out the pixels of other
    /// groups and the pixels above the growth threshold that are not in
    /// `group`, which belong to another source. A window without positive
    /// weight gives `None`.
    pub(super) fn centroid(
        &mut self,
        field: &Field<'_>,
        group: &[usize],
        visited: &[bool],
        config: &DetectConfig,
        shape: StarCentroid,
    ) -> Option<StarCentroid> {
        self.collect(field, group, visited, config);
        let mut sums = Sums::default();
        for &index in &self.pixels {
            let (x, y) = (index % field.width, index / field.width);
            let (level, noise) = field.grid.at(x, y);
            let weight = field.raw(index) - level;
            let variance = noise * noise + weight.max(0.0) / config.electrons_per_count;
            sums.add(weight, variance, x as f64, y as f64);
        }
        sums.star(shape)
    }

    /// Puts `group` and its margin into the window.
    fn collect(
        &mut self,
        field: &Field<'_>,
        group: &[usize],
        visited: &[bool],
        config: &DetectConfig,
    ) {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.mark.fill(0);
            self.stamp = 1;
        }
        self.pixels.clear();
        for &index in group {
            self.insert(index);
        }
        let margin = config.window_margin_px as usize;
        for &index in group {
            let (x, y) = (index % field.width, index / field.width);
            let rows = y.saturating_sub(margin)..=(y + margin).min(field.height - 1);
            for ny in rows {
                let cols = x.saturating_sub(margin)..=(x + margin).min(field.width - 1);
                for nx in cols {
                    let neighbour = ny * field.width + nx;
                    let taken = self.mark.get(neighbour) == Some(&self.stamp)
                        || visited.get(neighbour).copied().unwrap_or(true)
                        || field.excess(neighbour) > config.grow_sigmas;
                    if !taken {
                        self.insert(neighbour);
                    }
                }
            }
        }
    }

    fn insert(&mut self, index: usize) {
        if let Some(mark) = self.mark.get_mut(index) {
            *mark = self.stamp;
            self.pixels.push(index);
        }
    }
}

impl Sums {
    fn add(&mut self, weight: f64, variance: f64, x: f64, y: f64) {
        self.weight += weight;
        self.weight_x += weight * x;
        self.weight_y += weight * y;
        self.variance += variance;
        self.variance_x += variance * x;
        self.variance_y += variance * y;
        self.variance_xx += variance * x * x;
        self.variance_yy += variance * y * y;
        self.variance_xy += variance * x * y;
    }

    /// `shape` with the centre `c = Σ wᵢ·pᵢ / W`, the flux `W`, and the
    /// covariance `Σ (pᵢ − c)(pᵢ − c)ᵀ·Var(wᵢ) / W²`.
    fn star(&self, shape: StarCentroid) -> Option<StarCentroid> {
        if self.weight.is_nan() || self.weight <= 0.0 {
            return None;
        }
        let w = self.weight;
        let (cx, cy) = (self.weight_x / w, self.weight_y / w);
        let w2 = w * w;
        let xx = (self.variance_xx - 2.0 * cx * self.variance_x + cx * cx * self.variance) / w2;
        let yy = (self.variance_yy - 2.0 * cy * self.variance_y + cy * cy * self.variance) / w2;
        let xy = (self.variance_xy - cx * self.variance_y - cy * self.variance_x
            + cx * cy * self.variance)
            / w2;
        Some(StarCentroid {
            pixel: Vector2::new(cx, cy),
            flux: w,
            covariance_px2: Matrix2::new(xx.max(0.0), xy, xy, yy.max(0.0)),
            ..shape
        })
    }
}
