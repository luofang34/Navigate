//! The map and the frame of an acquisition: the map raster about a centre,
//! built a block a frame and kept while the beliefs of later frames stay
//! inside it, and the frame laid on the ground at each searched heading.

use std::sync::Arc;

use image::GrayImage;
use nalgebra::{UnitQuaternion, Vector2, Vector3};
use navigate_visual::acquisition::{GroundRaster, MapPyramid};
use navigate_visual::{CameraModel, LocalFrame};
use navigate_visual_render::{RenderError, SceneRequest, SceneWindow, map_raster};

use super::super::map::TerrainMap;
use super::super::names::{BELIEF_OUT_OF_RANGE, OUTSIDE_COVERAGE};
use super::super::view::NoView;

/// A new map raster reaches this much farther than the search of the
/// frame that made it, so that the searches of the next frames of a
/// moving vehicle stay inside it.
const RASTER_MARGIN: f64 = 1.25;

/// Blocks on a side of a map raster: one block is built a frame, so that
/// the reading and the sampling of the map for a raster of some 11 km
/// take a bounded time in each.
const BLOCKS: usize = 3;

/// The map raster of an acquisition about its centre.
#[derive(Debug)]
pub(in super::super) struct MapWindow {
    /// The east, north, up frame at the centre of the raster.
    pub frame: LocalFrame,
    /// Half side of the raster, metres.
    pub half_side_m: f64,
    pub pyramid: Arc<MapPyramid>,
}

impl MapWindow {
    /// The east and north offset of `[lat, lon]` from the centre, when a
    /// square of half side `need_m` about it is inside the raster.
    pub(in super::super) fn offset_of(&self, at: [f64; 2], need_m: f64) -> Option<Vector2<f64>> {
        let p = self.frame.local([at[0], at[1], 0.0]).ok()?;
        (p.x.abs().max(p.y.abs()) + need_m <= self.half_side_m).then(|| p.xy())
    }
}

/// A map raster being built, a block a frame.
#[derive(Debug, Clone)]
pub(in super::super) struct MapBuild {
    frame: LocalFrame,
    ground_m: f64,
    half_cells: usize,
    block_half: usize,
    parts: Vec<((i64, i64), GroundRaster)>,
}

impl MapBuild {
    /// The build of the raster about `centre` that holds a square of half
    /// side `need_m`, on ground at `ground_m` above the ellipsoid.
    pub(in super::super) fn new(
        map: &TerrainMap,
        centre: [f64; 2],
        ground_m: f64,
        need_m: f64,
    ) -> Result<Self, NoView> {
        let cell_m = map.config.acquisition_cell_m;
        let half_cells = (need_m * RASTER_MARGIN / cell_m).ceil() as usize;
        let side = 2 * half_cells + 1;
        // Blocks of an odd side that together cover the side.
        let block_half = side.div_ceil(BLOCKS) / 2;
        let frame = LocalFrame::wgs84_enu(centre[0], centre[1])
            .map_err(|_| NoView::Unused(BELIEF_OUT_OF_RANGE))?;
        Ok(Self {
            frame,
            ground_m,
            half_cells,
            block_half,
            parts: Vec::with_capacity(BLOCKS * BLOCKS),
        })
    }

    /// Whether the raster, when built, holds a square of half side
    /// `need_m` about `at`.
    pub(in super::super) fn covers(&self, at: [f64; 2], need_m: f64, cell_m: f64) -> bool {
        self.frame
            .local([at[0], at[1], 0.0])
            .is_ok_and(|p| p.x.abs().max(p.y.abs()) + need_m <= self.half_cells as f64 * cell_m)
    }

    /// Builds the next block; the raster once the last one is built.
    pub(in super::super) fn step(&mut self, map: &TerrainMap) -> Result<Option<MapWindow>, NoView> {
        let cell_m = map.config.acquisition_cell_m;
        let index = self.parts.len();
        let span = (2 * self.block_half + 1) as i64;
        let first = self.block_half as i64 - self.half_cells as i64;
        let offset = (
            first + (index % BLOCKS) as i64 * span,
            first + (index / BLOCKS) as i64 * span,
        );
        let local = Vector3::new(offset.0 as f64 * cell_m, offset.1 as f64 * cell_m, 0.0);
        let [lat, lon, _] = self.frame.geodetic(local);
        let half_m = (self.block_half as f64 + 2.0) * cell_m;
        let window = scene(map, [lat, lon], half_m)?;
        // The east and north of a block differ from those of the centre by
        // the convergence of the meridians, under a milliradian across the
        // raster: a few metres at its edge, a tenth of a cell.
        let part = map_raster(
            &window.scene,
            [lat, lon],
            self.ground_m,
            (cell_m, self.block_half),
        )
        .map_err(unused_render)?;
        self.parts.push((offset, part));
        if self.parts.len() < BLOCKS * BLOCKS {
            return Ok(None);
        }
        let raster = GroundRaster::mosaic(cell_m, self.half_cells, &self.parts)
            .map_err(|_| NoView::Unused(BELIEF_OUT_OF_RANGE))?;
        Ok(Some(MapWindow {
            frame: self.frame,
            half_side_m: self.half_cells as f64 * cell_m,
            pyramid: Arc::new(MapPyramid::new(raster)),
        }))
    }
}

/// The scene of the square of half side `half_side_m` about `centre`, at
/// the imagery zoom whose texels are a quarter of an acquisition cell and
/// the coarsest elevation zoom.
pub(in super::super) fn scene(
    map: &TerrainMap,
    centre: [f64; 2],
    half_side_m: f64,
) -> Result<SceneWindow, NoView> {
    let imagery_zoom = map
        .imagery
        .zoom_for(map.config.acquisition_cell_m / 4.0)
        .ok_or(NoView::Unused(OUTSIDE_COVERAGE))?;
    let elevation_zoom = map
        .elevation
        .zooms
        .first()
        .copied()
        .ok_or(NoView::Unused(OUTSIDE_COVERAGE))?;
    let request = SceneRequest {
        imagery_layer: map.imagery.layer.key,
        imagery_zoom,
        elevation_layer: map.elevation.layer.key,
        elevation_zoom,
        centre,
        half_side_m,
    };
    map.reader.scene(&request).map_err(unused_render)
}

pub(in super::super) fn unused_render(error: RenderError) -> NoView {
    match error {
        RenderError::Data(e) => NoView::Read(e.to_string()),
        RenderError::Decode(e) => NoView::Read(e.to_string()),
        other => {
            tracing::debug!(error = %other, "no scene window for the acquisition");
            NoView::Unused(OUTSIDE_COVERAGE)
        }
    }
}

/// The largest angle from the local down of a ray to a corner of `camera`
/// with the eye-to-local `orientation`.
pub(in super::super) fn steepest_ray_rad(
    camera: &CameraModel,
    orientation: &UnitQuaternion<f64>,
) -> f64 {
    [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        .iter()
        .map(|&(a, b)| {
            let u = a * f64::from(camera.width);
            let v = b * f64::from(camera.height);
            let eye = Vector3::new(
                (u - camera.cx) / camera.fx,
                (camera.cy - v) / camera.fy,
                -1.0,
            );
            let ray = orientation * eye;
            (-ray.z / ray.norm()).clamp(-1.0, 1.0).acos()
        })
        .fold(0.0_f64, f64::max)
}

/// The rotations about the local up at which a frame is laid: steps of
/// `step_rad` out to the search multiple of the attitude one-sigma, at
/// most `max_rad`, always including the believed heading.
pub(in super::super) fn rotations(
    sigma_rad: Option<f64>,
    multiple: f64,
    step_rad: f64,
    max_rad: f64,
) -> Vec<f64> {
    let wanted = sigma_rad.map_or(max_rad, |s| (multiple * s).min(max_rad));
    let steps = if step_rad > 0.0 {
        (wanted / step_rad).ceil().max(0.0) as i32
    } else {
        0
    };
    (-steps..=steps).map(|k| f64::from(k) * step_rad).collect()
}

/// `image` laid on flat ground `height_m` below the eye with the
/// eye-to-local `orientation`, at each of `rotations`.
pub(in super::super) fn laid(
    image: &GrayImage,
    camera: &CameraModel,
    orientation: &UnitQuaternion<f64>,
    height_m: f64,
    (cell_m, cells): (f64, usize),
    rotations: &[f64],
) -> Result<Vec<(f64, GroundRaster)>, NoView> {
    rotations
        .iter()
        .map(|&rotation| {
            let turned =
                UnitQuaternion::from_axis_angle(&Vector3::z_axis(), rotation) * orientation;
            GroundRaster::of_frame(image, camera, &turned, height_m, (cell_m, cells))
                .map(|raster| (rotation, raster))
                .map_err(|_| NoView::Unused(BELIEF_OUT_OF_RANGE))
        })
        .collect()
}
