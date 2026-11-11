//! Web Mercator (XYZ) tile arithmetic and the Terrarium height decoder.

use core::f64::consts::PI;

use navigate_geodesy::float_math::{atan, cos, exp, floor, ln, powi, tan};
use serde::{Deserialize, Serialize};

use crate::DataError;

/// The Web Mercator latitude limit in degrees.
pub const MAX_TILE_LATITUDE_DEG: f64 = 85.051_128_779_806_59;
/// The largest zoom that the contract accepts.
pub const MAX_ZOOM: u32 = 24;
/// Equatorial circumference of the WGS84 ellipsoid in metres: the Web
/// Mercator scale at zoom 0.
const EQUATOR_M: f64 = 40_075_016.685_578_49;

/// One XYZ tile. Rows increase to the south.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TileAddress {
    /// Zoom level.
    pub zoom: u32,
    /// Column, from the antimeridian to the east.
    pub x: u32,
    /// Row, from the north edge to the south.
    pub y: u32,
}

impl TileAddress {
    /// The tile at `zoom`, `x`, `y`.
    ///
    /// # Errors
    ///
    /// Refuses a zoom past [`MAX_ZOOM`] and a column or row outside the
    /// zoom.
    pub fn new(zoom: u32, x: u32, y: u32) -> Result<Self, DataError> {
        let side = 1_u64 << zoom.min(MAX_ZOOM);
        if zoom > MAX_ZOOM || u64::from(x) >= side || u64::from(y) >= side {
            return Err(DataError::Invalid {
                field: "tile",
                detail: alloc::format!("{zoom}/{x}/{y}"),
            });
        }
        Ok(Self { zoom, x, y })
    }

    /// The tile at `zoom` that holds a point, `[lat, lon]` in degrees.
    ///
    /// # Errors
    ///
    /// Refuses a point that is not finite or that is past
    /// [`MAX_TILE_LATITUDE_DEG`], and a zoom past [`MAX_ZOOM`].
    pub fn containing(point: [f64; 2], zoom: u32) -> Result<Self, DataError> {
        let [lat, lon] = point;
        if !(lat.is_finite() && lon.is_finite()) || lat.abs() > MAX_TILE_LATITUDE_DEG {
            return Err(DataError::Invalid {
                field: "point",
                detail: alloc::format!("{lat}, {lon}"),
            });
        }
        let [x, y] = tile_xy(point, zoom);
        let side = powi(2.0, zoom.min(MAX_ZOOM) as i32);
        let wrap = |v: f64| {
            let v = floor(v);
            if v < 0.0 {
                v + side
            } else if v >= side {
                v - side
            } else {
                v
            }
        };
        let clamp = |v: f64| floor(v).clamp(0.0, side - 1.0);
        Self::new(zoom, wrap(x) as u32, clamp(y) as u32)
    }

    /// North-west corner of the tile, `[lat, lon]` in degrees.
    #[must_use]
    pub fn north_west(self) -> [f64; 2] {
        tile_lat_lon(f64::from(self.x), f64::from(self.y), self.zoom)
    }

    /// The parent tile `levels` zoom levels coarser.
    #[must_use]
    pub const fn ancestor(self, levels: u32) -> Self {
        let levels = if levels > self.zoom {
            self.zoom
        } else {
            levels
        };
        Self {
            zoom: self.zoom - levels,
            x: self.x >> levels,
            y: self.y >> levels,
        }
    }
}

/// Fractional tile coordinates `[x, y]` of a point at `zoom`. The column is
/// not wrapped.
#[must_use]
pub fn tile_xy([lat, lon]: [f64; 2], zoom: u32) -> [f64; 2] {
    let n = powi(2.0, zoom as i32);
    let lat = lat
        .clamp(-MAX_TILE_LATITUDE_DEG, MAX_TILE_LATITUDE_DEG)
        .to_radians();
    let x = (lon + 180.0) / 360.0 * n;
    let y = (1.0 - ln(tan(PI / 4.0 + lat / 2.0)) / PI) / 2.0 * n;
    [x, y]
}

/// `[lat, lon]` in degrees of fractional tile coordinates at `zoom`.
#[must_use]
pub fn tile_lat_lon(x: f64, y: f64, zoom: u32) -> [f64; 2] {
    let n = powi(2.0, zoom as i32);
    let m = PI * (1.0 - 2.0 * y / n);
    let lat = atan((exp(m) - exp(-m)) / 2.0).to_degrees();
    [lat, x / n * 360.0 - 180.0]
}

/// Ground size of one pixel in metres at a latitude, for tiles of
/// `tile_pixels` pixels at `zoom`.
#[must_use]
pub fn ground_sample_distance_m(lat_deg: f64, zoom: u32, tile_pixels: u32) -> f64 {
    EQUATOR_M * cos(lat_deg.to_radians()) / (powi(2.0, zoom as i32) * f64::from(tile_pixels))
}

/// Height in metres of one Terrarium pixel: `R·256 + G + B/256 − 32768`.
#[must_use]
pub fn decode_terrarium_m([r, g, b]: [u8; 3]) -> f64 {
    f64::from(r) * 256.0 + f64::from(g) + f64::from(b) / 256.0 - 32_768.0
}

#[cfg(test)]
mod tests;
