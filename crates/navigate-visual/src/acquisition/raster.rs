//! A square raster of ground cells: the mean luma of the map or of a
//! frame laid on the ground, east and north about a centre.

use image::GrayImage;
use nalgebra::{UnitQuaternion, Vector2, Vector3};

use crate::{CameraModel, VisualError};

/// Sub-samples on a side of a cell: the mean of these is the cell.
const SUBSAMPLES: u32 = 4;

/// A square raster of ground cells about a centre: `side` cells east by
/// `side` cells north, row-major from the south-west corner. An unknown
/// cell is NaN.
#[derive(Clone, Debug)]
pub struct GroundRaster {
    pub(super) cell_m: f64,
    pub(super) half_cells: usize,
    pub(super) values: Vec<f32>,
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

    /// The raster of `(2 half_cells + 1)²` cells of `cell_m` metres put
    /// together from `parts`, each with the east and north offset of its
    /// centre in cells: a cell takes the value of the first part that
    /// holds it, and is unknown when none does.
    ///
    /// # Errors
    /// Rejects a part of another cell and an invalid cell.
    pub fn mosaic(
        cell_m: f64,
        half_cells: usize,
        parts: &[((i64, i64), Self)],
    ) -> Result<Self, VisualError> {
        if parts
            .iter()
            .any(|(_, p)| (p.cell_m - cell_m).abs() > 1e-9 * cell_m)
        {
            return Err(VisualError::Invalid {
                field: "mosaic cells differ",
            });
        }
        Self::from_cells(cell_m, half_cells, |east, north| {
            parts.iter().find_map(|((ce, cn), part)| {
                let h = part.half_cells as i64;
                ((east - ce).abs() <= h && (north - cn).abs() <= h)
                    .then(|| part.at(east - ce, north - cn))
                    .flatten()
            })
        })
    }

    /// The raster whose cell `(east, north)`, in cells from the centre, is
    /// `value(east, north)`; unknown where that is none.
    fn from_cells(
        cell_m: f64,
        half_cells: usize,
        mut value: impl FnMut(i64, i64) -> Option<f32>,
    ) -> Result<Self, VisualError> {
        if !cell_m.is_finite() || cell_m <= 0.0 {
            return Err(VisualError::Invalid {
                field: "ground raster cell",
            });
        }
        let half = half_cells as i64;
        let mut values = Vec::with_capacity((2 * half_cells + 1).pow(2));
        for north in -half..=half {
            for east in -half..=half {
                values.push(value(east, north).unwrap_or(f32::NAN));
            }
        }
        Ok(Self {
            cell_m,
            half_cells,
            values,
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
    pub(super) fn coarse(&self) -> Self {
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
    pub(super) fn known(&self) -> Vec<(i64, i64, f32)> {
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
