//! Acquisition of a belief wider than a reference view reaches: after a
//! long gap of fixes the belief can be kilometres wide, and the tracker of
//! one view finds the ground only a few hundred metres from where the
//! view was rendered. The frame, laid on flat ground below the believed
//! eye, is correlated with the map about the belief; a place that stands
//! clear of every other moves the belief there, and the frame is then
//! matched as any other.
//!
//! The filter cannot refuse a wrong place: a belief that wide admits any
//! fix within it. So the fix of an acquisition is held until the next
//! acquisition corrects the belief by the same amount, and only that
//! second fix goes to the filter.

use image::GrayImage;
use nalgebra::{UnitQuaternion, Vector2, Vector3};
use navigate_visual::acquisition::{Acquisition, GroundRaster, acquire};
use navigate_visual::{CameraModel, Estimate, LocalFrame};
use navigate_visual_render::{RenderError, SceneRequest, SceneWindow, ground_height_m, map_raster};

use super::belief::{Belief, Exposure, body_position};
use super::camera::TerrainCamera;
use super::config::TerrainConfig;
use super::map::TerrainMap;
use super::names::{
    ACQUISITION_AMBIGUOUS, ACQUISITION_WEAK, BELIEF_OUT_OF_RANGE, OUTSIDE_COVERAGE,
};
use super::view::{NoView, model};

/// Largest angle from the local down of a ray to the corner of the frame
/// that an acquisition lays on the ground, radians: past it the flat
/// ground below the eye stretches too far to stand for the terrain.
const MAX_CORNER_ANGLE_RAD: f64 = 1.0;

/// The search of one frame: the belief that the reference view is rendered
/// at, the position and attitude bounds of the match, and, after an
/// acquisition, the belief before it.
pub(super) struct Search {
    pub belief: Belief,
    pub position_radius_m: f64,
    pub attitude_radius_rad: f64,
    pub acquired_from: Option<Belief>,
}

/// An acquisition fix that waits for the next one.
#[derive(Debug, Clone, Copy)]
pub(super) struct Candidate {
    at_ns: u64,
    /// East and north from the belief to the fix, metres.
    correction_m: Vector2<f64>,
}

/// The search radius of a belief that a reference view cannot reach, the
/// search multiple of its one-sigma with the lag, at most the largest
/// acquisition radius; none for a belief that a view reaches, and none
/// when the largest acquisition radius is not above the largest search
/// radius of a view.
fn acquisition_radius_m(config: &TerrainConfig, sigma_h_m: f64, lag_s: f64) -> Option<f64> {
    let radius = config.search_sigma_multiple * sigma_h_m
        + 0.5 * config.lag_acceleration_m_s2 * lag_s * lag_s;
    let acquires = config.max_acquisition_radius_m > config.max_search_radius_m;
    (acquires && radius > config.max_search_radius_m)
        .then(|| radius.min(config.max_acquisition_radius_m))
}

/// The east and north offset from `from` to the horizontal position of
/// `to`, metres.
fn offset_m(from: &Belief, to: [f64; 3]) -> Option<Vector2<f64>> {
    let [lat, lon, _] = from.position;
    let frame = LocalFrame::wgs84_enu(lat, lon).ok()?;
    let a = frame.local(from.position).ok()?;
    let b = frame.local(to).ok()?;
    Some((b - a).xy())
}

/// `belief` moved `offset_m` east and north.
fn moved(belief: &Belief, offset_m: Vector2<f64>) -> Option<Belief> {
    let [lat, lon, _] = belief.position;
    let frame = LocalFrame::wgs84_enu(lat, lon).ok()?;
    let at = frame.local(belief.position).ok()? + Vector3::new(offset_m.x, offset_m.y, 0.0);
    Some(Belief {
        position: frame.geodetic(at),
        ..*belief
    })
}

/// The largest angle from the local down of a ray to a corner of `camera`
/// with the eye-to-local `orientation`.
fn steepest_ray_rad(camera: &CameraModel, orientation: &UnitQuaternion<f64>) -> f64 {
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

/// The scene of the square of half side `half_side_m` about `centre`, at
/// the imagery zoom whose texels are a quarter of an acquisition cell and
/// the coarsest elevation zoom.
fn window(map: &TerrainMap, centre: [f64; 2], half_side_m: f64) -> Result<SceneWindow, NoView> {
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

fn unused_render(error: RenderError) -> NoView {
    match error {
        RenderError::Data(e) => NoView::Read(e.to_string()),
        RenderError::Decode(e) => NoView::Read(e.to_string()),
        other => {
            tracing::debug!(error = %other, "no scene window for the acquisition");
            NoView::Unused(OUTSIDE_COVERAGE)
        }
    }
}

impl TerrainCamera {
    /// The search of a frame: the belief itself when a reference view
    /// reaches its search radius, else the belief moved to the place that
    /// an acquisition finds.
    pub(super) fn search(
        &self,
        map: &TerrainMap,
        image: &GrayImage,
        exposure: &Exposure,
    ) -> Result<Search, NoView> {
        let config = &map.config;
        let belief = exposure.belief;
        let lag_s = exposure.unmodelled_lag_s;
        let (position_radius_m, attitude_radius_rad) = config.search(belief.sigma_h_m, lag_s);
        let Some(radius_m) = acquisition_radius_m(config, belief.sigma_h_m, lag_s) else {
            return Ok(Search {
                belief,
                position_radius_m,
                attitude_radius_rad,
                acquired_from: None,
            });
        };
        let offset = self.acquire(map, image, &belief, radius_m)?;
        let at = moved(&belief, offset).ok_or(NoView::Unused(BELIEF_OUT_OF_RANGE))?;
        // The place is known to a cell or two: the match searches two
        // cells, as the search of a belief of that one-sigma would.
        let cell_sigma_m = 2.0 * config.acquisition_cell_m / config.search_sigma_multiple;
        let (position_radius_m, _) = config.search(cell_sigma_m, lag_s);
        Ok(Search {
            belief: at,
            position_radius_m,
            attitude_radius_rad,
            acquired_from: Some(belief),
        })
    }

    /// The east and north offset from `belief` to the place where the
    /// frame lies on the map, within `radius_m`.
    fn acquire(
        &self,
        map: &TerrainMap,
        image: &GrayImage,
        belief: &Belief,
        radius_m: f64,
    ) -> Result<Vector2<f64>, NoView> {
        let config = &map.config;
        let cell_m = config.acquisition_cell_m;
        let ([lat, lon, eye_height_m], pose) = belief
            .eye(&self.mount)
            .map_err(|_| NoView::Unused(BELIEF_OUT_OF_RANGE))?;
        let camera = model(&self.declared);
        let steepest = steepest_ray_rad(&camera, &pose.orientation);
        if steepest > MAX_CORNER_ANGLE_RAD {
            return Err(NoView::Unused(BELIEF_OUT_OF_RANGE));
        }
        let probe = window(map, [lat, lon], 2.0 * cell_m)?;
        let ground_m = ground_height_m(&probe.scene, [lat, lon])
            .or(self.ground_height_m)
            .ok_or(NoView::Unused(OUTSIDE_COVERAGE))?;
        let height_m = eye_height_m - ground_m;
        if height_m <= 0.0 {
            return Err(NoView::Unused(BELIEF_OUT_OF_RANGE));
        }
        let reach_m = height_m * steepest.tan();
        let frame_cells = (reach_m / cell_m).ceil() as usize + 1;
        let frame = GroundRaster::of_frame(
            image,
            &camera,
            &pose.orientation,
            height_m,
            (cell_m, frame_cells),
        )
        .map_err(|_| NoView::Unused(BELIEF_OUT_OF_RANGE))?;
        let window = window(map, [lat, lon], radius_m + reach_m + 2.0 * cell_m)?;
        let map_cells = ((radius_m + reach_m) / cell_m).ceil() as usize + 1;
        let ground = map_raster(&window.scene, [lat, lon], ground_m, (cell_m, map_cells))
            .map_err(unused_render)?;
        let found = acquire(&frame, &ground, radius_m, &config.acquisition)
            .map_err(|_| NoView::Unused(BELIEF_OUT_OF_RANGE))?;
        tracing::debug!(sensor = ?self.sensor, ?found, radius_m, "terrain acquisition");
        match found {
            Acquisition::Found(peak) => Ok(peak.offset_m),
            Acquisition::Ambiguous { .. } => Err(NoView::Unused(ACQUISITION_AMBIGUOUS)),
            Acquisition::Weak(_) => Err(NoView::Unused(ACQUISITION_WEAK)),
        }
    }

    /// Whether the pose of a frame at `at_ns` goes on to the filter: a
    /// frame that was not acquired does; a frame acquired from `belief`
    /// does only when the last acquisition, at most the confirmation time
    /// before, corrected the belief by the same amount. A fix that is not
    /// confirmed waits for the next one.
    pub(super) fn confirms(
        &mut self,
        map: &TerrainMap,
        belief: Option<&Belief>,
        estimate: &Estimate,
        at_ns: u64,
    ) -> bool {
        let Some(belief) = belief else {
            return true;
        };
        let config = &map.config;
        let fix = estimate
            .frame
            .geodetic(body_position(&estimate.pose, &self.mount));
        let Some(correction_m) = offset_m(belief, fix) else {
            return false;
        };
        let candidate = Candidate {
            at_ns,
            correction_m,
        };
        let agrees = self
            .candidate
            .is_some_and(|c| c.confirmed_by(&candidate, config));
        self.candidate = (!agrees).then_some(candidate);
        agrees
    }
}

impl Candidate {
    /// Whether `next`, at most the confirmation time later, corrects the
    /// belief as this one did.
    fn confirmed_by(&self, next: &Self, config: &TerrainConfig) -> bool {
        let age_s = next.at_ns.saturating_sub(self.at_ns) as f64 * 1e-9;
        next.at_ns > self.at_ns
            && age_s <= config.acquisition_confirm_s
            && (next.correction_m - self.correction_m).norm() <= config.acquisition_confirm_m
    }
}

#[cfg(test)]
mod tests;
