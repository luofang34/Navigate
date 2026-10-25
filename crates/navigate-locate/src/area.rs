//! Web Mercator tile grids around a point.

/// Mean Earth radius of the Web Mercator sphere, in metres.
const EARTH_RADIUS_M: f64 = 6_378_137.0;

/// A rectangle of tiles at one zoom, with exclusive ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileGrid {
    /// Tile zoom.
    pub zoom: u32,
    /// First and one-past-last tile column.
    pub x: [u32; 2],
    /// First and one-past-last tile row.
    pub y: [u32; 2],
}

impl TileGrid {
    /// Tiles that cover a square of half side `radius_m` around `[lat, lon]`.
    pub fn around([lat, lon]: [f64; 2], radius_m: f64, zoom: u32) -> Self {
        let dlat = (radius_m / EARTH_RADIUS_M).to_degrees();
        let dlon = dlat / lat.to_radians().cos().max(1e-6);
        let [x0, y0] = tile_xy([lat + dlat, lon - dlon], zoom);
        let [x1, y1] = tile_xy([lat - dlat, lon + dlon], zoom);
        let last = (1_u64 << zoom) as f64 - 1.0;
        let clamp = |v: f64| v.floor().clamp(0.0, last) as u32;
        Self {
            zoom,
            x: [clamp(x0), clamp(x1) + 1],
            y: [clamp(y0), clamp(y1) + 1],
        }
    }

    /// Number of tile columns and rows.
    pub fn dimensions(&self) -> [u32; 2] {
        [self.x[1] - self.x[0], self.y[1] - self.y[0]]
    }

    /// Number of tiles.
    pub fn count(&self) -> usize {
        let [w, h] = self.dimensions();
        w as usize * h as usize
    }

    /// Every `(x, y)` in row order.
    pub fn tiles(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        (self.y[0]..self.y[1]).flat_map(move |y| (self.x[0]..self.x[1]).map(move |x| (x, y)))
    }
}

/// Fractional tile coordinates of `[lat, lon]` at `zoom`.
pub fn tile_xy([lat, lon]: [f64; 2], zoom: u32) -> [f64; 2] {
    let n = (1_u64 << zoom) as f64;
    let lat = lat.clamp(-85.051_128, 85.051_128).to_radians();
    let x = (lon + 180.0) / 360.0 * n;
    let y = (1.0 - (lat.tan() + 1.0 / lat.cos()).ln() / std::f64::consts::PI) / 2.0 * n;
    [x, y]
}

#[cfg(test)]
mod tests;
