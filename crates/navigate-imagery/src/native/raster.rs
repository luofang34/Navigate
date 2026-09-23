use super::raster_error;
use crate::{ImageryError, Tile, error::invalid};
use gdal::{
    Dataset, GeoTransformEx,
    raster::GdalDataType,
    spatial_ref::{CoordTransform, SpatialRef},
};
use image::{Rgba, RgbaImage};

/// Source adapter for georeferenced imagery and validity.
pub trait RasterSource {
    /// Render one Web Mercator tile. Invalid source geometry stays transparent.
    ///
    /// # Errors
    /// Returns a source or reprojection failure.
    fn tile_blocking(&mut self, tile: Tile) -> Result<RgbaImage, ImageryError>;
}

/// GDAL range reader with Rust bilinear resampling and strict validity masks.
pub struct GdalRaster {
    dataset: Dataset,
    transform: CoordTransform,
    inverse: [f64; 6],
}

impl GdalRaster {
    /// Open a local raster or a GDAL `/vsicurl/` URL.
    ///
    /// # Errors
    /// Requires three unsigned 8-bit RGB bands and an invertible geotransform.
    pub fn open_blocking(path: &str) -> Result<Self, ImageryError> {
        let dataset = Dataset::open(path).map_err(raster_error("open source"))?;
        Self::from_dataset(dataset)
    }

    fn from_dataset(dataset: Dataset) -> Result<Self, ImageryError> {
        if dataset.raster_count() < 3 {
            return Err(invalid("source requires RGB bands"));
        }
        for i in 1..=3 {
            if dataset
                .rasterband(i)
                .map_err(raster_error("read band type"))?
                .band_type()
                != GdalDataType::UInt8
            {
                return Err(invalid("source RGB bands must be unsigned 8-bit"));
            }
        }
        let source =
            SpatialRef::from_epsg(3857).map_err(raster_error("create Mercator coordinates"))?;
        let target = dataset
            .spatial_ref()
            .map_err(raster_error("read source projection"))?;
        let transform = CoordTransform::new(&source, &target)
            .map_err(raster_error("create source transform"))?;
        let inverse = dataset
            .geo_transform()
            .and_then(|g| g.invert())
            .map_err(raster_error("invert source transform"))?;
        Ok(Self {
            dataset,
            transform,
            inverse,
        })
    }

    fn coordinates(&self, tile: Tile, row: u32, rows: u32) -> Result<Vec<[f64; 2]>, ImageryError> {
        let step = 40_075_016.685_578_49 / (2_f64.powi(tile.0 as i32) * 512.0);
        let half = 20_037_508.342_789_244;
        let mut x = Vec::with_capacity((512 * rows) as usize);
        let mut y = Vec::with_capacity((512 * rows) as usize);
        for yy in row..row + rows {
            for xx in 0..512 {
                x.push((f64::from(tile.1) * 512.0 + f64::from(xx) + 0.5) * step - half);
                y.push(half - (f64::from(tile.2) * 512.0 + f64::from(yy) + 0.5) * step);
            }
        }
        self.transform
            .transform_coords(&mut x, &mut y, &mut [])
            .map_err(raster_error("transform tile samples"))?;
        Ok(x.into_iter()
            .zip(y)
            .map(|(x, y)| {
                let (px, py) = self.inverse.apply(x, y);
                [px - 0.5, py - 0.5]
            })
            .collect())
    }

    fn render_rows(
        &self,
        tile: Tile,
        row: u32,
        output: &mut RgbaImage,
    ) -> Result<(), ImageryError> {
        let points = self.coordinates(tile, row, 32)?;
        let size = self.dataset.raster_size();
        let Some(window) = Window::for_points(&points, size)? else {
            return Ok(());
        };
        let mut channels = Vec::new();
        let mut valid = vec![255_u8; window.width * window.height];
        for index in 1..=3 {
            let band = self
                .dataset
                .rasterband(index)
                .map_err(raster_error("open source band"))?;
            channels.push(
                band.read_as::<u8>(window.origin(), window.size(), window.size(), None)
                    .map_err(raster_error("read source range"))?,
            );
            let mask = band
                .open_mask_band()
                .and_then(|mask| {
                    mask.read_as::<u8>(window.origin(), window.size(), window.size(), None)
                })
                .map_err(raster_error("read source validity"))?;
            for (v, m) in valid.iter_mut().zip(mask.data()) {
                *v = (*v).min(*m);
            }
        }
        for (i, point) in points.iter().enumerate() {
            if let Some(pixel) = sample(&channels, &valid, window, *point) {
                output.put_pixel((i % 512) as u32, row + (i / 512) as u32, Rgba(pixel));
            }
        }
        Ok(())
    }
}

impl RasterSource for GdalRaster {
    fn tile_blocking(&mut self, tile: Tile) -> Result<RgbaImage, ImageryError> {
        if tile.0 > 24 || tile.1 >= 1 << tile.0 || tile.2 >= 1 << tile.0 {
            return Err(invalid("invalid raster tile"));
        }
        let mut output = RgbaImage::new(512, 512);
        for row in (0..512).step_by(32) {
            self.render_rows(tile, row, &mut output)?;
        }
        Ok(output)
    }
}

#[derive(Clone, Copy)]
struct Window {
    x: usize,
    y: usize,
    width: usize,
    height: usize,
}
impl Window {
    fn origin(self) -> (isize, isize) {
        (self.x as isize, self.y as isize)
    }
    fn size(self) -> (usize, usize) {
        (self.width, self.height)
    }
    fn for_points(points: &[[f64; 2]], size: (usize, usize)) -> Result<Option<Self>, ImageryError> {
        if points.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid("nonfinite raster transform"));
        }
        let xmin = points
            .iter()
            .map(|p| p[0])
            .fold(f64::INFINITY, f64::min)
            .floor()
            .max(0.0) as usize;
        let ymin = points
            .iter()
            .map(|p| p[1])
            .fold(f64::INFINITY, f64::min)
            .floor()
            .max(0.0) as usize;
        let xmax = (points
            .iter()
            .map(|p| p[0])
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil()
            + 1.0)
            .max(0.0)
            .min(size.0 as f64) as usize;
        let ymax = (points
            .iter()
            .map(|p| p[1])
            .fold(f64::NEG_INFINITY, f64::max)
            .ceil()
            + 1.0)
            .max(0.0)
            .min(size.1 as f64) as usize;
        if xmin >= xmax || ymin >= ymax {
            return Ok(None);
        }
        if (xmax - xmin) * (ymax - ymin) > 16_777_216 {
            return Err(invalid("raster sample window exceeds memory limit"));
        }
        Ok(Some(Self {
            x: xmin,
            y: ymin,
            width: xmax - xmin,
            height: ymax - ymin,
        }))
    }
}

fn sample(
    channels: &[gdal::raster::Buffer<u8>],
    valid: &[u8],
    window: Window,
    p: [f64; 2],
) -> Option<[u8; 4]> {
    let x = p[0] - window.x as f64;
    let y = p[1] - window.y as f64;
    if x < 0.0 || y < 0.0 || x + 1.0 >= window.width as f64 || y + 1.0 >= window.height as f64 {
        return None;
    }
    let index = y.floor() as usize * window.width + x.floor() as usize;
    let ids = [
        index,
        index + 1,
        index + window.width,
        index + window.width + 1,
    ];
    if ids.iter().any(|i| valid[*i] != 255) {
        return None;
    }
    let dx = x.fract();
    let dy = y.fract();
    let weights = [
        (1.0 - dx) * (1.0 - dy),
        dx * (1.0 - dy),
        (1.0 - dx) * dy,
        dx * dy,
    ];
    let mut pixel = [0, 0, 0, 255];
    for c in 0..3 {
        pixel[c] = ids
            .iter()
            .zip(weights)
            .map(|(i, w)| f64::from(channels[c].data()[*i]) * w)
            .sum::<f64>()
            .round() as u8;
    }
    Some(pixel)
}

#[cfg(test)]
mod tests;
