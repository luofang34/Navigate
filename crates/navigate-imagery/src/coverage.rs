use crate::{ImageryError, Tile, error::invalid, tile_bounds, tile_position};
use serde::{Deserialize, Serialize};

/// An area or route selection. Specify exactly one of `bounds` and `route`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CoverageRequest {
    /// West, south, east, north in degrees.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bounds: Option<[f64; 4]>,
    /// Longitude, latitude route points in degrees.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route: Option<Vec<[f64; 2]>>,
    /// Route corridor half-width in metres.
    #[serde(default = "default_buffer")]
    pub buffer_m: f64,
    /// Imagery zoom from 14 to 17.
    #[serde(default = "default_zoom")]
    pub zoom: u32,
}
fn default_buffer() -> f64 {
    1000.0
}
fn default_zoom() -> u32 {
    16
}

/// A bounded coverage plan. Creating it sends no network request.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CoveragePlan {
    /// Data provider label.
    pub provider: String,
    /// Full tile bounds in west, south, east, north order.
    pub bounds: [f64; 4],
    /// Original area or route request, separate from the navigation prior.
    pub requested: CoverageRequest,
    /// Required imagery tiles.
    pub imagery_tiles: Vec<Tile>,
    /// Uncompressed imagery byte estimate, excluding terrain and overhead.
    pub estimated_max_bytes: u64,
    /// Requested imagery zoom.
    pub imagery_zoom: u32,
    /// Native terrain zoom.
    pub terrain_zoom: u32,
    /// Offline use condition.
    pub offline_use: String,
}

/// Plan bounded imagery coverage without changing the navigation prior.
///
/// # Errors
/// Rejects malformed, polar, antimeridian, empty, or oversized selections.
pub fn plan(request: CoverageRequest) -> Result<CoveragePlan, ImageryError> {
    if !(14..=17).contains(&request.zoom) {
        return Err(invalid("imagery zoom must be 14..=17"));
    }
    let (limits, points, radius) = selection(&request)?;
    let [x0, y0, x1, y1] = limits.map(|v| v.floor() as i64);
    let n = 1_i64 << request.zoom;
    if x0 < 0 || y0 < 0 || x1 >= n || y1 >= n || (x1 - x0 + 1) * (y1 - y0 + 1) > 100_000 {
        return Err(invalid(
            "selection is too large or crosses supported map bounds",
        ));
    }
    let mut tiles = Vec::new();
    for y in y0..=y1 {
        for x in x0..=x1 {
            let p = [x as f64 + 0.5, y as f64 + 0.5];
            if !points.is_empty()
                && !points
                    .windows(2)
                    .any(|s| distance(p, s[0], s[1]) <= radius + 0.5_f64.sqrt())
            {
                continue;
            }
            tiles.push(Tile(request.zoom, x as u32, y as u32));
            if tiles.len() > 400 {
                return Err(invalid("selection exceeds 400 imagery tiles"));
            }
        }
    }
    let bounds = tile_envelope(&tiles)?;
    Ok(CoveragePlan {
        provider: "Microsoft Planetary Computer / USDA NAIP".into(),
        bounds,
        estimated_max_bytes: tiles.len() as u64 * 512 * 512 * 4,
        imagery_zoom: request.zoom,
        terrain_zoom: 14,
        requested: request,
        imagery_tiles: tiles,
        offline_use: "Public domain; retain attribution".into(),
    })
}

fn point([lon, lat]: [f64; 2]) -> Result<(), ImageryError> {
    if !lon.is_finite()
        || !lat.is_finite()
        || !(-180.0..=180.0).contains(&lon)
        || !(-85.0..=85.0).contains(&lat)
    {
        return Err(invalid(format!(
            "unsupported longitude/latitude: {lon}, {lat}"
        )));
    }
    Ok(())
}

type Selection = ([f64; 4], Vec<[f64; 2]>, f64);
fn selection(r: &CoverageRequest) -> Result<Selection, ImageryError> {
    match (&r.bounds, &r.route) {
        (Some([w, s, e, n]), None) => {
            point([*w, *s])?;
            point([*e, *n])?;
            if w >= e || s >= n {
                return Err(invalid("area is empty or crosses the antimeridian"));
            }
            let [x0, y0] = tile_position(*w, *n, r.zoom);
            let [x1, y1] = tile_position(*e, *s, r.zoom);
            Ok(([x0, y0, x1, y1], Vec::new(), 0.0))
        }
        (None, Some(route)) => route_selection(r, route),
        _ => Err(invalid("specify one area or one route")),
    }
}

fn route_selection(r: &CoverageRequest, route: &[[f64; 2]]) -> Result<Selection, ImageryError> {
    if !(2..=128).contains(&route.len())
        || !r.buffer_m.is_finite()
        || !(100.0..=10000.0).contains(&r.buffer_m)
    {
        return Err(invalid(
            "route requires 2..=128 points and a 100..=10000 m buffer",
        ));
    }
    for &p in route {
        point(p)?;
    }
    if route.windows(2).any(|s| (s[1][0] - s[0][0]).abs() > 180.0) {
        return Err(invalid("antimeridian routes are unsupported"));
    }
    let latitude = route.iter().map(|p| p[1].abs()).fold(0.0, f64::max);
    let radius = r.buffer_m
        / (40_075_016.685_578_49 * latitude.to_radians().cos() / 2_f64.powi(r.zoom as i32));
    let points: Vec<_> = route
        .iter()
        .map(|p| tile_position(p[0], p[1], r.zoom))
        .collect();
    let limits = [
        points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min) - radius,
        points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min) - radius,
        points
            .iter()
            .map(|p| p[0])
            .fold(f64::NEG_INFINITY, f64::max)
            + radius,
        points
            .iter()
            .map(|p| p[1])
            .fold(f64::NEG_INFINITY, f64::max)
            + radius,
    ];
    Ok((limits, points, radius))
}

fn distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let [dx, dy] = [b[0] - a[0], b[1] - a[1]];
    let norm = dx * dx + dy * dy;
    let t = if norm > 0.0 {
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / norm).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p[0] - a[0] - t * dx).hypot(p[1] - a[1] - t * dy)
}

pub(crate) fn tile_envelope(tiles: &[Tile]) -> Result<[f64; 4], ImageryError> {
    if tiles.is_empty() {
        return Err(invalid("selection has no imagery tiles"));
    }
    Ok(tiles.iter().fold([180.0, 90.0, -180.0, -90.0], |a, t| {
        let b = tile_bounds(*t);
        [
            a[0].min(b[0]),
            a[1].min(b[1]),
            a[2].max(b[2]),
            a[3].max(b[3]),
        ]
    }))
}

#[cfg(test)]
mod tests;
