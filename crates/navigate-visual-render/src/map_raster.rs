//! The orthographic raster of the imagery of a scene, for the acquisition
//! of a belief too wide for a reference view.

use nalgebra::Vector3;
use navigate_visual::acquisition::GroundRaster;

use crate::sparse::mapping::MapCoordinates;
use crate::sparse::{imagery_luma, sample_terrain};
use crate::{RenderError, RenderScene};

/// The raster of `(2 half_cells + 1)²` cells of `cell_m` metres of the
/// imagery of `scene`, east and north about `centre` (`[lat, lon]` in
/// degrees) on ground at `ground_height_m` above the ellipsoid. A cell
/// that the imagery does not cover is unknown.
///
/// # Errors
/// Refuses a centre outside the frame of the scene and an invalid cell.
pub fn map_raster(
    scene: &RenderScene,
    centre: [f64; 2],
    ground_height_m: f64,
    (cell_m, half_cells): (f64, usize),
) -> Result<GroundRaster, RenderError> {
    let frame = scene.frame;
    let origin = frame.local([centre[0], centre[1], ground_height_m])?;
    let reach_m = std::f64::consts::SQRT_2 * cell_m * (half_cells as f64 + 1.0);
    let map = MapCoordinates::about(frame, origin, reach_m);
    let raster = GroundRaster::from_samples(cell_m, half_cells, |offset| {
        let (mercator_xy, _) = map.at(origin + Vector3::new(offset.x, offset.y, 0.0));
        imagery_luma(&scene.imagery, mercator_xy)
    })?;
    Ok(raster)
}

/// The height above the ellipsoid of the terrain of `scene` at `[lat,
/// lon]` in degrees; none where the scene holds no terrain.
#[must_use]
pub fn ground_height_m(scene: &RenderScene, [lat, lon]: [f64; 2]) -> Option<f64> {
    let x = (lon + 180.0) / 360.0;
    let y = (1.0 - lat.to_radians().tan().asinh() / std::f64::consts::PI) * 0.5;
    sample_terrain(&scene.terrain, [x, y])
}

#[cfg(test)]
mod tests;
