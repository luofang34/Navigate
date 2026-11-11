//! Tiles of a route corridor.
//!
//! The corridor of a route is the set of points within a distance of the
//! route. The selection samples each great-circle leg at a spacing of half
//! the corridor radius, and adds the tiles of a latitude and longitude box
//! around each sample. The half side of the box is the radius plus half
//! the spacing, so each corridor point is in the box of its nearest sample.
//! The box uses the polar radius, the smallest radius of the Earth, and the
//! poleward edge latitude for its longitude span, so each box contains its
//! circle. Columns wrap across the antimeridian. A corridor past the Web
//! Mercator latitude limit is refused.

use alloc::collections::BTreeSet;

use navigate_geodesy::float_math::{ceil, cos, floor};

use crate::resource::{great_circle_m, point_along};
use crate::tile::tile_xy;
use crate::{DataError, MAX_TILE_LATITUDE_DEG, TileAddress};

/// The polar radius of the WGS84 ellipsoid, in metres.
const POLAR_RADIUS_M: f64 = 6_356_752.314_245;

/// Adds to `tiles` the tiles at `zoom` within `radius_m` of the route
/// through `points` (`[lat, lon]` in degrees).
///
/// # Errors
///
/// Refuses a route with no point, a radius that is not finite and positive,
/// a point that is not finite, and a corridor past
/// [`MAX_TILE_LATITUDE_DEG`].
pub fn corridor_tiles(
    points: &[[f64; 2]],
    radius_m: f64,
    zoom: u32,
    tiles: &mut BTreeSet<TileAddress>,
) -> Result<(), DataError> {
    if !(radius_m.is_finite() && radius_m > 0.0) || zoom > crate::MAX_ZOOM {
        return Err(DataError::Invalid {
            field: "corridor",
            detail: alloc::format!("radius {radius_m} m at zoom {zoom}"),
        });
    }
    if points.is_empty() || points.iter().flatten().any(|v| !v.is_finite()) {
        return Err(DataError::Invalid {
            field: "route",
            detail: "a route needs at least one finite point".into(),
        });
    }
    let spacing_m = radius_m / 2.0;
    let half_side_m = radius_m + spacing_m / 2.0;
    if let [only] = points {
        box_tiles(*only, half_side_m, zoom, tiles)?;
    }
    for pair in points.windows(2) {
        let &[a, b] = pair else { continue };
        let samples = ceil(great_circle_m(a, b) / spacing_m).max(1.0) as u32;
        for i in 0..=samples {
            let t = f64::from(i) / f64::from(samples);
            box_tiles(point_along(a, b, t), half_side_m, zoom, tiles)?;
        }
    }
    Ok(())
}

fn box_tiles(
    [lat, lon]: [f64; 2],
    half_side_m: f64,
    zoom: u32,
    tiles: &mut BTreeSet<TileAddress>,
) -> Result<(), DataError> {
    let dlat = (half_side_m / POLAR_RADIUS_M).to_degrees();
    let poleward = lat.abs() + dlat;
    if poleward >= MAX_TILE_LATITUDE_DEG {
        return Err(DataError::Invalid {
            field: "latitude",
            detail: alloc::format!("the corridor reaches {poleward}°, past the tile limit"),
        });
    }
    let dlon = dlat / cos(poleward.to_radians());
    let [x0, y0] = tile_xy([lat + dlat, lon - dlon], zoom);
    let [x1, y1] = tile_xy([lat - dlat, lon + dlon], zoom);
    let columns = 1_i64 << zoom;
    for y in (floor(y0) as i64)..=(floor(y1) as i64) {
        for x in (floor(x0) as i64)..=(floor(x1) as i64) {
            tiles.insert(TileAddress {
                zoom,
                x: x.rem_euclid(columns) as u32,
                y: y.clamp(0, columns - 1) as u32,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
