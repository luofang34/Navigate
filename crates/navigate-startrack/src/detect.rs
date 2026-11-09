//! Star centroids from a star camera image.
//!
//! A pixel is a seed when it is more than [`DetectConfig::seed_sigmas`]
//! noise deviations above the local background. A star is the
//! 8-connected group of pixels around a seed that are more than
//! [`DetectConfig::grow_sigmas`] deviations above the background. The
//! lower growth threshold keeps more of the faint wings of the point
//! spread function, which decreases the centroid bias.

mod background;

use nalgebra::Vector2;
use navigate_camera::{ImageView, Pixels};

use crate::StarTrackError;
use background::BackgroundGrid;

/// The settings of [`detect_stars`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectConfig {
    /// Side of a square background tile in pixels.
    pub tile_px: u32,
    /// Noise deviations above the background that start a star.
    pub seed_sigmas: f64,
    /// Noise deviations above the background that a pixel of a star must
    /// have. It is not more than `seed_sigmas`.
    pub grow_sigmas: f64,
    /// Smallest noise deviation in sensor counts. It keeps the threshold
    /// above the background in an image without noise.
    pub min_noise_counts: f64,
    /// Fewest pixels of a star. A value of 2 or more rejects a single hot
    /// pixel.
    pub min_pixels: usize,
    /// Most pixels of a star. A larger group, such as a planet disc or a
    /// lit cloud, is not a star.
    pub max_pixels: usize,
    /// Most pixel groups in one image. More groups give
    /// [`StarTrackError::TooManyComponents`].
    pub max_components: usize,
}

impl Default for DetectConfig {
    fn default() -> Self {
        Self {
            tile_px: 32,
            seed_sigmas: 5.0,
            grow_sigmas: 2.0,
            min_noise_counts: 0.5,
            min_pixels: 2,
            max_pixels: 400,
            max_components: 4096,
        }
    }
}

/// One star image in the raw (distorted) image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StarCentroid {
    /// Intensity-weighted centre in pixels, with the pixel convention of
    /// [`navigate_camera::Intrinsics`].
    pub pixel: Vector2<f64>,
    /// Sum of the background-subtracted values in sensor counts.
    pub flux: f64,
    /// Pixels in the group.
    pub pixel_count: usize,
    /// Largest raw value of the group in sensor counts.
    pub peak: f64,
    /// Whether a pixel of the group is at the full scale of the sensor. The
    /// centroid of a saturated star has a larger error.
    pub saturated: bool,
}

/// The sums of one pixel group.
#[derive(Debug, Clone, Copy, Default)]
struct Moments {
    weight: f64,
    weighted_x: f64,
    weighted_y: f64,
    count: usize,
    peak: f64,
}

/// The stars of `image`, brightest first.
///
/// Sensor counts are raw values for [`Pixels::Gray8`] and
/// [`Pixels::Gray16`], and the Rec. 709 luma times 255 for
/// [`Pixels::Rgb8`]. The result is a function of the pixels and the
/// configuration only.
///
/// # Errors
/// Refuses an invalid image or configuration, and returns
/// [`StarTrackError::TooManyComponents`] for an image with more groups than
/// [`DetectConfig::max_components`].
pub fn detect_stars(
    image: &ImageView<'_>,
    config: &DetectConfig,
) -> Result<Vec<StarCentroid>, StarTrackError> {
    image.validate()?;
    validate(config)?;
    let width = image.width() as usize;
    let height = image.height() as usize;
    let (samples, full_scale) = samples_of(image);
    let grid = BackgroundGrid::estimate(
        &samples,
        width,
        height,
        config.tile_px as usize,
        config.min_noise_counts,
    );
    let field = Field {
        samples: &samples,
        width,
        height,
        grid: &grid,
    };
    let mut visited = vec![false; samples.len()];
    let mut stack = Vec::new();
    let mut components = 0_usize;
    let mut stars = Vec::new();
    for seed in 0..samples.len() {
        let is_new = visited.get(seed).is_some_and(|v| !v);
        if !is_new || field.excess(seed) <= config.seed_sigmas {
            continue;
        }
        components = components.saturating_add(1);
        if components > config.max_components {
            return Err(StarTrackError::TooManyComponents {
                limit: config.max_components,
            });
        }
        let moments = field.grow(seed, config.grow_sigmas, &mut visited, &mut stack);
        if (config.min_pixels..=config.max_pixels).contains(&moments.count) && moments.weight > 0.0
        {
            stars.push(StarCentroid {
                pixel: Vector2::new(
                    moments.weighted_x / moments.weight,
                    moments.weighted_y / moments.weight,
                ),
                flux: moments.weight,
                pixel_count: moments.count,
                peak: moments.peak,
                saturated: moments.peak >= full_scale - 0.5,
            });
        }
    }
    stars.sort_by(|a, b| b.flux.total_cmp(&a.flux));
    Ok(stars)
}

fn validate(config: &DetectConfig) -> Result<(), StarTrackError> {
    let thresholds = config.grow_sigmas.is_finite()
        && config.seed_sigmas.is_finite()
        && config.grow_sigmas >= 0.0
        && config.grow_sigmas <= config.seed_sigmas;
    if !thresholds {
        return Err(StarTrackError::Invalid {
            field: "detection thresholds",
        });
    }
    if config.tile_px == 0
        || !(config.min_noise_counts > 0.0 && config.min_noise_counts.is_finite())
    {
        return Err(StarTrackError::Invalid {
            field: "background tile or noise floor",
        });
    }
    if config.min_pixels == 0 || config.min_pixels > config.max_pixels {
        return Err(StarTrackError::Invalid {
            field: "star pixel count limits",
        });
    }
    Ok(())
}

/// The samples of `image` in sensor counts, row after row without padding,
/// and the full-scale value.
fn samples_of(image: &ImageView<'_>) -> (Vec<f32>, f64) {
    let full_scale = match image.pixels() {
        Pixels::Gray16 {
            significant_bits, ..
        } => f64::from(u16::MAX >> (16 - significant_bits.clamp(1, 16))),
        _ => 255.0,
    };
    let scale = full_scale as f32;
    let mut samples = Vec::with_capacity(image.width() as usize * image.height() as usize);
    for y in 0..image.height() {
        for x in 0..image.width() {
            samples.push(image.luma(x, y).unwrap_or(0.0) * scale);
        }
    }
    (samples, full_scale)
}

/// The samples of an image with their background.
struct Field<'a> {
    samples: &'a [f32],
    width: usize,
    height: usize,
    grid: &'a BackgroundGrid,
}

impl Field<'_> {
    /// The value above the background in noise deviations. A pixel outside
    /// the image gives negative infinity.
    fn excess(&self, index: usize) -> f64 {
        let (level, sigma) = self.grid.at(index % self.width, index / self.width);
        self.samples
            .get(index)
            .map_or(f64::NEG_INFINITY, |v| (f64::from(*v) - level) / sigma)
    }

    /// Marks the 8-connected group of `seed` above `grow_sigmas` as visited
    /// and gives its sums.
    fn grow(
        &self,
        seed: usize,
        grow_sigmas: f64,
        visited: &mut [bool],
        stack: &mut Vec<usize>,
    ) -> Moments {
        let mut moments = Moments::default();
        stack.clear();
        stack.push(seed);
        if let Some(v) = visited.get_mut(seed) {
            *v = true;
        }
        while let Some(index) = stack.pop() {
            let (x, y) = (index % self.width, index / self.width);
            let (level, _) = self.grid.at(x, y);
            let raw = self.samples.get(index).map_or(0.0, |v| f64::from(*v));
            let weight = (raw - level).max(0.0);
            moments.weight += weight;
            moments.weighted_x += weight * x as f64;
            moments.weighted_y += weight * y as f64;
            moments.count = moments.count.saturating_add(1);
            moments.peak = moments.peak.max(raw);
            for neighbour in self.neighbours(x, y) {
                let fresh = visited.get(neighbour).is_some_and(|v| !v);
                if fresh && self.excess(neighbour) > grow_sigmas {
                    if let Some(v) = visited.get_mut(neighbour) {
                        *v = true;
                    }
                    stack.push(neighbour);
                }
            }
        }
        moments
    }

    /// The indices of the up to eight pixels around `(x, y)`.
    fn neighbours(&self, x: usize, y: usize) -> impl Iterator<Item = usize> + '_ {
        let rows = y.saturating_sub(1)..=(y + 1).min(self.height - 1);
        rows.flat_map(move |ny| {
            let cols = x.saturating_sub(1)..=(x + 1).min(self.width - 1);
            cols.filter_map(move |nx| (nx != x || ny != y).then_some(ny * self.width + nx))
        })
    }
}

#[cfg(test)]
mod tests;
