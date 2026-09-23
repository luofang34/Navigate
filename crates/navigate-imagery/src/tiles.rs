use serde::{Deserialize, Serialize};

/// A Web Mercator tile in zoom, column, row order.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct Tile(pub u32, pub u32, pub u32);

/// Get fractional tile coordinates for longitude and latitude in degrees.
pub fn tile_position(longitude: f64, latitude: f64, zoom: u32) -> [f64; 2] {
    let n = 2_f64.powi(zoom as i32);
    [
        (longitude + 180.0) / 360.0 * n,
        (1.0 - latitude.to_radians().tan().asinh() / std::f64::consts::PI) / 2.0 * n,
    ]
}

/// Get west, south, east, north bounds in degrees for a Web Mercator tile.
pub fn tile_bounds(Tile(z, x, y): Tile) -> [f64; 4] {
    let n = 2_f64.powi(z as i32);
    let lat = |v: f64| {
        (std::f64::consts::PI * (1.0 - 2.0 * v / n))
            .sinh()
            .atan()
            .to_degrees()
    };
    [
        f64::from(x) / n * 360.0 - 180.0,
        lat(f64::from(y) + 1.0),
        (f64::from(x) + 1.0) / n * 360.0 - 180.0,
        lat(f64::from(y)),
    ]
}
