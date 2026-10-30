//! Search regions and the index cells that cover them.
use serde::{Deserialize, Serialize};

/// Zoom level of an index cell in the Web Mercator tile grid.
///
/// A cell at this zoom is about 39 km wide at the equator and 24 km at 53 degrees.
pub const CELL_ZOOM: u32 = 10;
const EARTH_M: f64 = 6_371_000.0;

/// An area where the camera or the ground in view can be. Points are latitude and
/// longitude in degrees.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Region {
    /// All points within `radius_m` of `centre`.
    Circle {
        /// Latitude and longitude of the center.
        centre: [f64; 2],
        /// Radius in meters.
        radius_m: f64,
    },
    /// All points inside a closed outline. The last point connects to the first.
    Polygon {
        /// Latitude and longitude of each corner.
        points: Vec<[f64; 2]>,
    },
}

/// One cell of the index: a Web Mercator tile at [`CELL_ZOOM`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Cell {
    /// Tile column.
    pub x: u32,
    /// Tile row, from the north.
    pub y: u32,
}

/// Meters east and north of `origin` to a point.
pub(crate) fn local([lat0, lon0]: [f64; 2], [lat, lon]: [f64; 2]) -> [f64; 2] {
    let north = (lat - lat0).to_radians() * EARTH_M;
    let east = (lon - lon0).to_radians() * EARTH_M * lat0.to_radians().cos();
    [east, north]
}

/// Ground distance in meters between two near points.
pub(crate) fn distance_m(a: [f64; 2], b: [f64; 2]) -> f64 {
    let [e, n] = local(a, b);
    e.hypot(n)
}

impl Region {
    /// True when the point is in the region.
    pub fn contains(&self, point: [f64; 2]) -> bool {
        match self {
            Self::Circle { centre, radius_m } => distance_m(*centre, point) <= *radius_m,
            Self::Polygon { points } => inside(points, point),
        }
    }

    /// South, west, north and east edges, in degrees.
    pub fn bounds(&self) -> [f64; 4] {
        match self {
            Self::Circle { centre, radius_m } => {
                let d_lat = (radius_m / EARTH_M).to_degrees();
                let d_lon = d_lat / centre[0].to_radians().cos().max(1e-6);
                [
                    centre[0] - d_lat,
                    centre[1] - d_lon,
                    centre[0] + d_lat,
                    centre[1] + d_lon,
                ]
            }
            Self::Polygon { points } => points.iter().fold(
                [
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ],
                |[s, w, n, e], [lat, lon]| [s.min(*lat), w.min(*lon), n.max(*lat), e.max(*lon)],
            ),
        }
    }

    /// Area in square kilometres.
    pub fn area_km2(&self) -> f64 {
        match self {
            Self::Circle { radius_m, .. } => std::f64::consts::PI * radius_m * radius_m / 1e6,
            Self::Polygon { points } => {
                let Some(origin) = points.first() else {
                    return 0.0;
                };
                let xy: Vec<[f64; 2]> = points.iter().map(|p| local(*origin, *p)).collect();
                let twice: f64 = xy
                    .iter()
                    .zip(xy.iter().cycle().skip(1))
                    .map(|(a, b)| a[0] * b[1] - b[0] * a[1])
                    .sum();
                twice.abs() / 2.0 / 1e6
            }
        }
    }

    /// The region grown by `margin_m` on all sides, as a polygon or a circle.
    pub fn grown(&self, margin_m: f64) -> Region {
        match self {
            Self::Circle { centre, radius_m } => Self::Circle {
                centre: *centre,
                radius_m: radius_m + margin_m,
            },
            Self::Polygon { .. } => {
                let [s, w, n, e] = self.bounds();
                let d_lat = (margin_m / EARTH_M).to_degrees();
                let d_lon = d_lat / ((s + n) / 2.0).to_radians().cos().max(1e-6);
                Self::Polygon {
                    points: vec![
                        [s - d_lat, w - d_lon],
                        [s - d_lat, e + d_lon],
                        [n + d_lat, e + d_lon],
                        [n + d_lat, w - d_lon],
                    ],
                }
            }
        }
    }
}

/// Even-odd test in latitude and longitude; regions are small enough for this.
fn inside(points: &[[f64; 2]], [lat, lon]: [f64; 2]) -> bool {
    let mut result = false;
    for (a, b) in points.iter().zip(points.iter().cycle().skip(1)) {
        if (a[0] > lat) != (b[0] > lat) {
            let cross = a[1] + (lat - a[0]) / (b[0] - a[0]) * (b[1] - a[1]);
            if lon < cross {
                result = !result;
            }
        }
    }
    result
}

/// Fractional Web Mercator tile coordinates of a point.
pub(crate) fn tile_xy([lat, lon]: [f64; 2], zoom: u32) -> [f64; 2] {
    let n = f64::from(1_u32 << zoom);
    let phi = lat.to_radians();
    [
        (lon + 180.0) / 360.0 * n,
        (1.0 - (phi.tan() + 1.0 / phi.cos()).ln() / std::f64::consts::PI) / 2.0 * n,
    ]
}

/// The cells whose area touches the region's bounds.
pub fn cells(region: &Region) -> Vec<Cell> {
    let [south, west, north, east] = region.bounds();
    let [x0, y0] = tile_xy([north, west], CELL_ZOOM);
    let [x1, y1] = tile_xy([south, east], CELL_ZOOM);
    let last = (1_u32 << CELL_ZOOM) - 1;
    let clamp = |v: f64| (v.floor().max(0.0) as u32).min(last);
    let mut out = Vec::new();
    for x in clamp(x0)..=clamp(x1) {
        for y in clamp(y0)..=clamp(y1) {
            out.push(Cell { x, y });
        }
    }
    out
}

#[cfg(test)]
mod tests;
