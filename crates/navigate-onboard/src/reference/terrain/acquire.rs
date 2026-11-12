//! Acquisition of a belief wider than a reference view reaches (ADR-0009):
//! after a long gap of fixes the belief can be kilometres wide, and the
//! tracker of one view finds the ground only a few hundred metres from
//! where the view was rendered.
//!
//! The navigation estimate sets the tier of the search. A local tier
//! matches at the belief. A wider one acquires: the frame, laid on flat
//! ground below the believed eye at a few headings about the believed one,
//! is correlated with a raster of the map about the belief, and the places
//! that score within the margin of the best are the candidates. Each is
//! matched as any frame; a fix needs exactly one geometrically accepted
//! candidate.
//!
//! The filter cannot refuse a wrong place: a belief that wide admits any
//! fix within it. So the fix of an acquisition is held until the next
//! acquisition, from another frame and another reference view, corrects
//! the belief by the same amount, and only that second fix goes to the
//! filter.

mod confirm;
mod window;

use std::sync::Arc;

use image::GrayImage;
use nalgebra::{Matrix3, UnitQuaternion, Vector2, Vector3};
use navigate_visual::acquisition::{Acquisition, AcquisitionSearch};
use navigate_visual::{
    CandidateDecision, CandidateId, CandidateResults, Estimate, FrameStamp, LocalFrame,
    ReferenceView, SearchConfig, SearchPrior, SearchTier, VisualError,
};
use navigate_visual_render::ground_height_m;

pub(super) use confirm::Candidate;
use window::{MapBuild, MapWindow, laid, rotations, scene, steepest_ray_rad};

use super::belief::{Belief, Exposure, body_position};
use super::camera::{Attempt, TerrainCamera};
use super::config::TerrainConfig;
use super::map::TerrainMap;
use super::names::{
    ACQUISITION_AMBIGUOUS, ACQUISITION_PENDING, ACQUISITION_WEAK, BELIEF_OUT_OF_RANGE,
    OUTSIDE_COVERAGE,
};
use super::view::{NoView, model};

/// Largest angle from the local down of a ray to the corner of the frame
/// that an acquisition lays on the ground, radians: past it the flat
/// ground below the eye stretches too far to stand for the terrain.
const MAX_CORNER_ANGLE_RAD: f64 = 1.0;

/// Most places of an ambiguous acquisition that are matched; with more,
/// the ground repeats too much for a match to choose.
const MAX_CANDIDATES: usize = 3;

/// The search of one frame: the beliefs at which reference views are
/// rendered and matched, the position and attitude bounds of each match,
/// and, after an acquisition, the belief before it.
pub(super) struct Search {
    pub candidates: Vec<Belief>,
    pub position_radius_m: f64,
    pub attitude_radius_rad: f64,
    pub acquired_from: Option<Belief>,
}

/// What acquisition keeps between frames: the map raster, or the build of
/// one, a search that goes on over frames, and the fix that waits for
/// confirmation.
#[derive(Debug, Clone, Default)]
pub(super) struct Acquiring {
    window: Option<Arc<MapWindow>>,
    building: Option<MapBuild>,
    pending: Option<Pending>,
    candidate: Option<Candidate>,
}

/// A search begun on an earlier frame.
#[derive(Debug, Clone)]
struct Pending {
    search: AcquisitionSearch,
    /// East and north of the believed eye from the centre of the raster
    /// when the search began, metres.
    eye_offset_m: Vector2<f64>,
    started_ns: u64,
}

/// The tier that the belief supports, ADR-0009: local up to the largest
/// search radius of a view.
fn tier(config: &TerrainConfig, belief: &Belief, pose: navigate_visual::CameraPose) -> SearchTier {
    let variance = belief.sigma_h_m.powi(2).max(f64::MIN_POSITIVE);
    let prior = SearchPrior {
        center: pose,
        position_covariance_m2: Matrix3::from_diagonal(&Vector3::new(variance, variance, variance)),
        attitude_sigma_rad: belief
            .sigma_attitude_rad
            .unwrap_or(config.attitude_radius_rad),
    };
    let search = SearchConfig {
        sigma_scale: config.search_sigma_multiple,
        local_radius_m: config.max_search_radius_m,
        region_radius_m: config.max_acquisition_radius_m,
        candidate_spacing_m: config.acquisition_cell_m,
        max_candidates: MAX_CANDIDATES,
    };
    prior.tier(&search)
}

/// The search radius of an acquisition: the search multiple of the
/// one-sigma with the lag, at most the largest acquisition radius.
fn acquisition_radius_m(config: &TerrainConfig, sigma_h_m: f64, lag_s: f64) -> f64 {
    (config.search_sigma_multiple * sigma_h_m + 0.5 * config.lag_acceleration_m_s2 * lag_s * lag_s)
        .min(config.max_acquisition_radius_m)
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

/// `belief` moved `offset_m` east and north and turned `rotation_rad`
/// counter-clockwise seen from above, which is a turn of the heading to
/// the left.
fn moved(belief: &Belief, offset_m: Vector2<f64>, rotation_rad: f64) -> Option<Belief> {
    let [lat, lon, _] = belief.position;
    let frame = LocalFrame::wgs84_enu(lat, lon).ok()?;
    let at = frame.local(belief.position).ok()? + Vector3::new(offset_m.x, offset_m.y, 0.0);
    // The down axis of north, east, down: a turn to the left is negative.
    let turn = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), -rotation_rad);
    Some(Belief {
        position: frame.geodetic(at),
        body_to_ned: turn * belief.body_to_ned,
        ..*belief
    })
}

impl TerrainCamera {
    /// The search of a frame acquired at `at_ns`: the belief itself in
    /// the local tier, else the places that an acquisition finds.
    pub(super) fn search(
        &mut self,
        map: &TerrainMap,
        image: &GrayImage,
        exposure: &Exposure,
        at_ns: u64,
    ) -> Result<Search, NoView> {
        let config = &map.config;
        let belief = exposure.belief;
        let lag_s = exposure.unmodelled_lag_s;
        let (position_radius_m, attitude_radius_rad) =
            config.search(belief.sigma_h_m, belief.sigma_attitude_rad, lag_s);
        let (_, pose) = belief
            .eye(&self.mount)
            .map_err(|_| NoView::Unused(BELIEF_OUT_OF_RANGE))?;
        let acquires = config.max_acquisition_radius_m > config.max_search_radius_m;
        if !acquires || tier(config, &belief, pose) == SearchTier::Local {
            self.acquiring.pending = None;
            return Ok(Search {
                candidates: vec![belief],
                position_radius_m,
                attitude_radius_rad,
                acquired_from: None,
            });
        }
        let radius_m = acquisition_radius_m(config, belief.sigma_h_m, lag_s);
        let (found, eye_offset_m) = self.acquire(map, image, &belief, radius_m, at_ns)?;
        tracing::debug!(sensor = ?self.sensor, ?found, radius_m, "terrain acquisition");
        let places = match found {
            Acquisition::Found(peak) => vec![peak],
            Acquisition::Ambiguous(places) if places.len() <= MAX_CANDIDATES => places,
            Acquisition::Ambiguous(_) => return Err(NoView::Unused(ACQUISITION_AMBIGUOUS)),
            Acquisition::Weak(_) => return Err(NoView::Unused(ACQUISITION_WEAK)),
        };
        let candidates = places
            .iter()
            .map(|p| moved(&belief, p.offset_m - eye_offset_m, p.rotation_rad))
            .collect::<Option<Vec<_>>>()
            .ok_or(NoView::Unused(BELIEF_OUT_OF_RANGE))?;
        // A place is known to a cell or two: the match searches two cells,
        // as the search of a belief of that one-sigma would.
        let cell_sigma_m = 2.0 * config.acquisition_cell_m / config.search_sigma_multiple;
        let (position_radius_m, _) = config.search(cell_sigma_m, None, lag_s);
        Ok(Search {
            candidates,
            position_radius_m,
            attitude_radius_rad,
            acquired_from: Some(belief),
        })
    }

    /// The result of the acquisition that this frame completes, and the
    /// believed eye's offset from the raster centre when it began; none,
    /// as `acquisition_pending`, while a search goes on.
    fn acquire(
        &mut self,
        map: &TerrainMap,
        image: &GrayImage,
        belief: &Belief,
        radius_m: f64,
        at_ns: u64,
    ) -> Result<(Acquisition, Vector2<f64>), NoView> {
        let config = &map.config;
        let fresh = |p: &Pending| {
            at_ns.saturating_sub(p.started_ns) as f64 * 1e-9 <= config.acquisition_confirm_s
        };
        let mut pending = match self.acquiring.pending.take() {
            Some(p) if fresh(&p) => p,
            _ => self.begin_search(map, image, belief, radius_m, at_ns)?,
        };
        match pending.search.advance(config.acquisition_pairs_per_frame) {
            Some(found) => Ok((found, pending.eye_offset_m)),
            None => {
                self.acquiring.pending = Some(pending);
                Err(NoView::Unused(ACQUISITION_PENDING))
            }
        }
    }

    /// A new search of `image` within `radius_m` of `belief`.
    fn begin_search(
        &mut self,
        map: &TerrainMap,
        image: &GrayImage,
        belief: &Belief,
        radius_m: f64,
        at_ns: u64,
    ) -> Result<Pending, NoView> {
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
        let probe = scene(map, [lat, lon], 2.0 * cell_m)?;
        let ground_m = ground_height_m(&probe.scene, [lat, lon])
            .or(self.ground_height_m)
            .ok_or(NoView::Unused(OUTSIDE_COVERAGE))?;
        let height_m = eye_height_m - ground_m;
        if height_m <= 0.0 {
            return Err(NoView::Unused(BELIEF_OUT_OF_RANGE));
        }
        let reach_m = height_m * steepest.tan();
        let need_m = radius_m + reach_m + 2.0 * cell_m;
        let window = self
            .window_for(map, [lat, lon], ground_m, need_m)?
            .ok_or(NoView::Unused(ACQUISITION_PENDING))?;
        let eye_offset_m = window
            .offset_of([lat, lon], need_m)
            .ok_or(NoView::Unused(OUTSIDE_COVERAGE))?;
        let turns = rotations(
            belief.sigma_attitude_rad,
            config.search_sigma_multiple,
            config.acquisition_rotation_step_rad,
            config.max_acquisition_rotation_rad,
        );
        let cells = (reach_m / cell_m).ceil() as usize + 1;
        let frames = laid(
            image,
            &camera,
            &pose.orientation,
            height_m,
            (cell_m, cells),
            &turns,
        )?;
        let pyramid = Arc::clone(&window.pyramid);
        let search =
            AcquisitionSearch::new(frames, pyramid, eye_offset_m, radius_m, &config.acquisition)
                .map_err(|_| NoView::Unused(BELIEF_OUT_OF_RANGE))?;
        Ok(Pending {
            search,
            eye_offset_m,
            started_ns: at_ns,
        })
    }

    /// The map raster that holds a square of half side `need_m` about
    /// `at`: the kept one, or one built a block a frame, none while it is
    /// built.
    fn window_for(
        &mut self,
        map: &TerrainMap,
        at: [f64; 2],
        ground_m: f64,
        need_m: f64,
    ) -> Result<Option<Arc<MapWindow>>, NoView> {
        let acquiring = &mut self.acquiring;
        if let Some(window) = &acquiring.window
            && window.offset_of(at, need_m).is_some()
        {
            return Ok(Some(Arc::clone(window)));
        }
        let cell_m = map.config.acquisition_cell_m;
        let mut build = match acquiring.building.take() {
            Some(b) if b.covers(at, need_m, cell_m) => b,
            _ => MapBuild::new(map, at, ground_m, need_m)?,
        };
        match build.step(map)? {
            Some(window) => {
                let window = Arc::new(window);
                acquiring.window = Some(Arc::clone(&window));
                Ok(Some(window))
            }
            None => {
                acquiring.building = Some(build);
                Ok(None)
            }
        }
    }

    /// The match of the frame at the one belief of a local search, or at
    /// each place of an acquisition; with several places, the one that
    /// alone gives an accepted pose.
    pub(super) fn choose(
        &mut self,
        map: &TerrainMap,
        search: &Search,
        inputs: (&GrayImage, FrameStamp, f64),
    ) -> Result<Attempt, (NoView, Option<&'static str>)> {
        let mut attempts = Vec::with_capacity(search.candidates.len());
        for belief in &search.candidates {
            match self.attempt(map, search, *belief, inputs) {
                Ok(attempt) => attempts.push(attempt),
                Err((NoView::Read(detail), path)) => return Err((NoView::Read(detail), path)),
                Err(refused) if search.candidates.len() == 1 => return Err(refused),
                Err(_) => {}
            }
        }
        if attempts.len() <= 1 && search.candidates.len() == 1 {
            return attempts
                .pop()
                .ok_or((NoView::Unused(ACQUISITION_AMBIGUOUS), None));
        }
        let ambiguous = |path| (NoView::Unused(ACQUISITION_AMBIGUOUS), path);
        let path = attempts.first().map(|a| a.reference.path);
        let Some(first) = attempts.first() else {
            return Err(ambiguous(None));
        };
        let mut results = CandidateResults::new(&first.frame);
        for (index, attempt) in attempts.iter().enumerate() {
            let result =
                attempt
                    .estimate
                    .as_ref()
                    .map(Clone::clone)
                    .map_err(|_| VisualError::Invalid {
                        field: "candidate refused",
                    });
            results
                .record(CandidateId(index as u64), result)
                .map_err(|_| ambiguous(path))?;
        }
        match results.decision() {
            CandidateDecision::Unique(CandidateId(index)) => {
                let index = usize::try_from(index).map_err(|_| ambiguous(path))?;
                if index < attempts.len() {
                    Ok(attempts.swap_remove(index))
                } else {
                    Err(ambiguous(path))
                }
            }
            CandidateDecision::Rejected => attempts.into_iter().next().ok_or(ambiguous(path)),
            CandidateDecision::Unresolved(_) => Err(ambiguous(path)),
        }
    }

    /// Whether the pose of a frame at `at_ns` goes on to the filter: a
    /// local search's does; an acquisition's does only when the last
    /// acquisition, at most the confirmation time before, corrected its
    /// belief by the same amount within the uncertainty of the two fixes
    /// and of the motion between them. A fix that is not confirmed waits
    /// for the next one.
    pub(super) fn confirms(
        &mut self,
        map: &TerrainMap,
        search: &Search,
        (estimate, view): (&Estimate, &ReferenceView),
        at_ns: u64,
    ) -> bool {
        let Some(belief) = search.acquired_from else {
            return true;
        };
        let fix = estimate
            .frame
            .geodetic(body_position(&estimate.pose, &self.mount));
        let Some(correction_m) = offset_m(&belief, fix) else {
            return false;
        };
        let candidate = Candidate {
            at_ns,
            correction_m,
            fix_variance_m2: self.horizontal_variance_m2(map, estimate, view),
            belief_sigma_m: belief.sigma_h_m,
        };
        let agrees = self
            .acquiring
            .candidate
            .is_some_and(|c| c.confirmed_by(&candidate, &map.config));
        self.acquiring.candidate = (!agrees).then_some(candidate);
        agrees
    }

    /// Drops what acquisition kept: the map changed.
    pub(super) fn forget_acquisition(&mut self) {
        self.acquiring = Acquiring::default();
    }
}

#[cfg(test)]
mod tests;
