use crate::RenderError;
use navigate_visual::LocalFrame;

/// Pixel grid on a bounded rectangle of Web Mercator tiles.
#[derive(Clone, Copy, Debug)]
pub struct AtlasLayout {
    /// Tile zoom, from zero through 22.
    pub zoom: u32,
    /// Tile x and y at the top-left of the grid.
    pub origin_tile: [u32; 2],
    /// Samples per tile side, a power of two no larger than 256.
    pub tile_pixels: u32,
    /// Grid dimensions in pixels.
    pub size: [u32; 2],
}
impl AtlasLayout {
    pub(crate) fn validate(&self) -> Result<usize, RenderError> {
        if self.zoom > 22
            || !self.tile_pixels.is_power_of_two()
            || self.tile_pixels > 256
            || self
                .size
                .iter()
                .any(|n| *n == 0 || *n % self.tile_pixels != 0)
        {
            return Err(RenderError::Invalid("atlas grid dimensions"));
        }
        let side = 1_u64 << self.zoom;
        for (origin, size) in self.origin_tile.into_iter().zip(self.size) {
            if u64::from(origin) + u64::from(size / self.tile_pixels) > side {
                return Err(RenderError::Invalid(
                    "atlas extends outside tile addressing",
                ));
            }
        }
        (self.size[0] as usize)
            .checked_mul(self.size[1] as usize)
            .ok_or(RenderError::Invalid("atlas size overflow"))
    }

    pub(crate) fn projection(&self, frame: LocalFrame) -> [f32; 4] {
        let [lat, lon] = frame.anchor();
        let [x, y] = LocalFrame::AnchorMercator {
            anchor_lat_deg: lat,
            anchor_lon_deg: lon,
        }
        .mercator_xy(nalgebra::Vector3::zeros());
        let scale = f64::from(self.tile_pixels) * f64::from(1_u32 << self.zoom);
        [
            (x * scale - f64::from(self.origin_tile[0] * self.tile_pixels) - 0.5) as f32,
            (y * scale - f64::from(self.origin_tile[1] * self.tile_pixels) - 0.5) as f32,
            (scale / std::f64::consts::TAU) as f32,
            (scale / frame.mercator_scale_m()) as f32,
        ]
    }
}

/// Grayscale imagery and explicit coverage. Missing pixels never become ground evidence.
pub struct ImageAtlas {
    /// Geographic grid layout.
    pub layout: AtlasLayout,
    /// Row-major grayscale values.
    pub pixels: Vec<u8>,
    /// Row-major coverage, zero or 255.
    pub coverage: Vec<u8>,
}
impl ImageAtlas {
    pub(crate) fn validate(&self) -> Result<usize, RenderError> {
        let count = self.layout.validate()?;
        if self.pixels.len() != count
            || self.coverage.len() != count
            || self.coverage.iter().any(|v| ![0, 255].contains(v))
        {
            return Err(RenderError::Invalid("imagery pixels or coverage"));
        }
        Ok(count)
    }
}

/// Numeric map-datum elevations and coverage on a regular tile grid.
pub struct TerrainAtlas {
    /// Geographic grid layout.
    pub layout: AtlasLayout,
    /// Row-major terrain heights, in metres. Uncovered values are ignored.
    pub heights_m: Vec<f32>,
    /// Row-major coverage, zero or 255.
    pub coverage: Vec<u8>,
}
impl TerrainAtlas {
    pub(crate) fn validate(&self) -> Result<usize, RenderError> {
        let count = self.layout.validate()?;
        if self.heights_m.len() != count
            || self.coverage.len() != count
            || self.coverage.iter().any(|v| ![0, 255].contains(v))
            || self
                .heights_m
                .iter()
                .zip(&self.coverage)
                .any(|(h, c)| *c == 255 && !h.is_finite())
        {
            return Err(RenderError::Invalid("terrain heights or coverage"));
        }
        Ok(count)
    }
}
