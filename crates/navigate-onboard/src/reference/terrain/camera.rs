//! One terrain camera: its declared model, the working resolution of the
//! match, the state of its capture stream, and the steps of one frame.

use std::sync::Arc;

use image::GrayImage;
use image::imageops::{FilterType, resize};
use nalgebra::Vector3;
use navigate_camera::{CameraMount, Distortion, Intrinsics, SpectralBand};
use navigate_contract::{ClockDomainId, DurationNanos, SourceEpoch, SourceId};
use navigate_data::{CorrelationKey, GridCell, TileAddress};
use navigate_visual::fusion_bridge::{VisualSourceIdentity, position_fix_observation_composed};
use navigate_visual::session::{ContinuityEpoch, FrameKey, MapCell, MapCellLedger, StreamId};
use navigate_visual::{
    BoundedPyramidalMatcher, Estimate, Frame, FrameStamp, Localizer, LocalizerConfig, PosePrior,
    ReferenceView,
};

use super::acquire::Acquiring;
use super::belief::{Belief, BeliefSource, Exposure, body_position};
use super::budget::{ViewGeometry, external_covariance_ned};
use super::config::WorkingCamera;
use super::inspection::{Pending, TerrainMatch};
use super::map::TerrainMap;
use super::names::{self, rejection};
use super::view::{NoView, Views, model};
use super::{Sighting, admission};
use crate::{ErrorBudget, OnboardError, SensorInput, SensorSpec, TerrainConfig};

mod attempt;

pub(super) use attempt::Attempt;

/// Largest count of map cells that the check before the match looks at.
/// A search disk that touches more cells is not checked, and the frame is
/// matched.
const MAX_CHECKED_CELLS: u64 = 64;

/// The pixel noise floor of a terrain match, pixels: the precision of the
/// matcher on textured ground, whose reprojection RMS is 0.25 to 0.6
/// pixels when the camera sees the map imagery.
const PIXEL_NOISE_FLOOR_PX: f64 = 0.2;
/// The residual at which the geometry covariance of a terrain match
/// doubles, pixels ([`LocalizerConfig::residual_correlation_px`]). A
/// changed appearance (exposure, response, light, blur, noise,
/// compression, another season), a registration error of the map, and a
/// fine repeated texture raise the residual to 1 to 1.6 pixels, and the
/// error of the fix grows more than the residual alone says. The value is
/// calibrated against the real matcher over heights of 400 m to 1760 m,
/// views of 0° to 45°, and those cases (`tests::geometry`).
const RESIDUAL_CORRELATION_PX: f64 = 1.2;

/// Margin of the offset bound of a match, pixels.
const OFFSET_MARGIN_PX: f64 = 4.0;

/// A bound on the offset of a ground point between the reference view,
/// rendered at the belief, and the frame, pixels of the working camera.
///
/// A position error inside the position radius of the prior, which holds
/// the floor and the ceiling that the host sets for the search, moves a
/// point by the error over the range, times the focal length for the
/// shift and times the distance from the centre for the change of scale.
/// An attitude error turns a point by the angle times the focal length;
/// it is taken at the search multiple of the belief's attitude one-sigma
/// with the lag, because the attitude radius of the prior is a gate on
/// the solved attitude, far wider than a belief of a few milliradians
/// needs. A belief without an attitude one-sigma takes that radius. A
/// wide belief, as after a long gap of fixes, searches as far as the
/// plain matcher.
fn offset_bound_px(
    belief: &Belief,
    (config, lag_s): (&super::config::TerrainConfig, f64),
    prior: &PosePrior,
    working: &WorkingCamera,
) -> f64 {
    let attitude_rad = belief
        .sigma_attitude_rad
        .map_or(prior.attitude_radius_rad, |sigma| {
            let rad = config.search_sigma_multiple * sigma + config.lag_rate_rad_s * lag_s;
            rad.min(prior.attitude_radius_rad)
        });
    let k = &working.intrinsics;
    let focal = k.fx.max(k.fy);
    let range_m = working.ground_pixel_m * k.fx;
    let half_diagonal = f64::from(k.width).hypot(f64::from(k.height)) / 2.0;
    prior.position_radius_m / range_m * (focal + half_diagonal)
        + attitude_rad * focal
        + OFFSET_MARGIN_PX
}

/// The acceptance and the covariance of a terrain match.
fn localizer_config() -> LocalizerConfig {
    LocalizerConfig {
        pixel_noise_floor: PIXEL_NOISE_FLOOR_PX,
        residual_correlation_px: RESIDUAL_CORRELATION_PX,
        ..LocalizerConfig::default()
    }
}

/// One declared terrain camera and the state of its capture stream.
#[derive(Debug, Clone)]
pub(crate) struct TerrainCamera {
    pub(super) sensor: SourceId,
    /// The camera as the host declared it: the size of each frame.
    pub(super) declared: Intrinsics,
    pub(super) mount: CameraMount,
    calibration_sigma_rad: f64,
    identity: VisualSourceIdentity,
    pub(super) localizer: Localizer<BoundedPyramidalMatcher>,
    /// Count of frames taken; the sequence of the next frame.
    frames: u64,
    pub(super) views: Views,
    /// Height above the ellipsoid of the ground that the last match saw,
    /// metres. It gives the range of the next view before a match.
    pub(super) ground_height_m: Option<f64>,
    /// What acquisition keeps between frames.
    pub(super) acquiring: Acquiring,
    /// The camera of the last reference view and match.
    working: Option<WorkingCamera>,
    /// Where the belief of the last frame came from.
    belief_source: Option<BeliefSource>,
    /// Whether the host asked to keep what each frame did.
    inspect: bool,
    /// What the frame being taken made so far.
    pending: Pending,
    /// What the last frame did, when `inspect` is set.
    last: Option<TerrainMatch>,
}

/// The frame of the match: the image at the size of the working camera.
/// The triangle filter averages the source pixels under each working
/// pixel, and it maps pixel centres as `Intrinsics::resized` does.
fn resampled(stamp: FrameStamp, image: GrayImage, working: &Intrinsics) -> Frame {
    let size = (working.width, working.height);
    Frame {
        stamp,
        camera: model(working),
        image: if image.dimensions() == size {
            image
        } else {
            resize(&image, size.0, size.1, FilterType::Triangle)
        },
    }
}

impl TerrainCamera {
    pub(super) fn new(spec: &SensorSpec, clock: ClockDomainId) -> Result<Self, OnboardError> {
        let invalid = |detail: String| OnboardError::Config {
            field: "terrain camera",
            detail,
        };
        let (
            Some(imager),
            Some(mount),
            ErrorBudget::TerrainCamera {
                calibration_sigma_rad,
            },
        ) = (spec.imager, spec.mount, spec.budget)
        else {
            return Err(invalid(format!(
                "{:?} needs an imager, a mount, and a budget",
                spec.id
            )));
        };
        if imager.band != SpectralBand::Visible {
            return Err(OnboardError::UnsupportedBand {
                sensor: spec.id,
                band: imager.band,
            });
        }
        // The means does not remove lens distortion. A declared distortion
        // says that the frames have it, so their geometry would be wrong.
        if imager.distortion != Distortion::None {
            return Err(invalid(format!(
                "{:?} declares a lens distortion; the terrain means takes undistorted frames",
                spec.id
            )));
        }
        let declared = imager.intrinsics;
        let usable = calibration_sigma_rad.is_finite() && calibration_sigma_rad >= 0.0;
        if model(&declared).validate().is_err() || !usable {
            return Err(invalid(format!("{:?} intrinsics or budget", spec.id)));
        }
        Ok(Self {
            sensor: spec.id,
            declared,
            mount,
            calibration_sigma_rad,
            identity: VisualSourceIdentity {
                source: spec.id,
                epoch: SourceEpoch::new(0),
                clock,
            },
            localizer: Localizer::new(
                BoundedPyramidalMatcher::new(f64::INFINITY),
                localizer_config(),
            )
            .map_err(|e| invalid(e.to_string()))?,
            frames: 0,
            views: Views::default(),
            ground_height_m: None,
            acquiring: Acquiring::default(),
            working: None,
            belief_source: None,
            inspect: false,
            pending: Pending::default(),
            last: None,
        })
    }

    /// Drops what the camera kept for another map: a loaded map begins
    /// with no resident view.
    pub(super) fn reset(&mut self) {
        self.views = Views::default();
        self.ground_height_m = None;
        self.forget_acquisition();
        self.working = None;
        self.belief_source = None;
        self.forget_match();
    }

    /// Keeps, or stops keeping, what each frame did.
    pub(super) fn set_inspect(&mut self, on: bool) {
        self.inspect = on;
        self.forget_match();
    }

    /// What the last frame did, when the host asked to keep it.
    pub(super) const fn last_match(&self) -> Option<&TerrainMatch> {
        self.last.as_ref()
    }

    /// Drops what earlier frames made: a new frame starts.
    pub(super) fn begin(&mut self) {
        self.pending = Pending::default();
    }

    /// Records what the frame did, when the host asked to keep it.
    pub(super) fn finish(
        &mut self,
        acquired_at: navigate_contract::MonotonicNanos,
        outcome: crate::Outcome,
        path: Option<&'static str>,
    ) {
        let pending = core::mem::take(&mut self.pending);
        if self.inspect {
            self.last = Some(pending.finish(acquired_at, outcome, path));
        }
    }

    /// Drops the record: the last frame failed, or the map changed.
    pub(super) fn forget_match(&mut self) {
        self.pending = Pending::default();
        self.last = None;
    }

    /// Where the belief of the last frame came from.
    pub(super) const fn belief_source(&self) -> Option<BeliefSource> {
        self.belief_source
    }

    /// The camera of the last reference view and match.
    pub(super) const fn working(&self) -> Option<WorkingCamera> {
        self.working
    }

    /// The image of a frame, when its size is the declared size.
    pub(super) fn image(
        &self,
        width: u32,
        height: u32,
        luma: &[u8],
    ) -> Result<GrayImage, OnboardError> {
        let declared = (self.declared.width, self.declared.height);
        ((width, height) == declared)
            .then(|| GrayImage::from_raw(width, height, luma.to_vec()))
            .flatten()
            .ok_or(OnboardError::FrameSize {
                sensor: self.sensor,
                declared,
                frame: (width, height),
                bytes: luma.len(),
            })
    }

    /// Whether each map cell that the frame can show already has a fix, so
    /// that a match can only give a fix that the filter is not told. The
    /// believed ground point of the optical axis is the centre of a disk
    /// of the position search radius and of the attitude uncertainty times
    /// the range. A fix that the match can give has its ground in that
    /// disk, so a frame that is not matched loses no fix of a new cell
    /// while the error of the belief is inside the search bounds.
    fn all_cells_have_a_fix(
        &self,
        map: &TerrainMap,
        ledger: &MapCellLedger,
        belief: &Belief,
        position_radius_m: f64,
        frame: FrameKey,
    ) -> Option<&'static str> {
        let ground_height_m = self.ground_height_m?;
        let ([lat, lon], range_m) =
            belief.ground(&self.mount, ground_height_m, map.config.far_m)?;
        let radius_m = position_radius_m
            + map.config.search_sigma_multiple * belief.sigma_attitude_rad? * range_m;
        let layer = &map.imagery.layer;
        let dlat = (radius_m / 6_356_752.0).to_degrees();
        let dlon = dlat / lat.to_radians().cos().max(1e-6);
        let zoom = layer.correlation_zoom;
        let north_west = TileAddress::containing([lat + dlat, lon - dlon], zoom).ok()?;
        let south_east = TileAddress::containing([lat - dlat, lon + dlon], zoom).ok()?;
        let columns = u64::from(south_east.x.checked_sub(north_west.x)?) + 1;
        let rows = u64::from(south_east.y.checked_sub(north_west.y)?) + 1;
        if columns * rows > MAX_CHECKED_CELLS {
            return None;
        }
        let mut reason = None;
        for y in north_west.y..=south_east.y {
            for x in north_west.x..=south_east.x {
                let cell = MapCell::Source(CorrelationKey {
                    source: layer.key.source,
                    revision: layer.revision,
                    cell: GridCell(TileAddress { zoom, x, y }),
                });
                // One cell that the filter can still take a fix from is
                // enough to match the frame.
                reason = Some(admission(&ledger.label(&cell, frame))?);
            }
        }
        reason
    }

    /// Matches one frame and gives the fix, or why there is none.
    pub(super) fn sight(
        &mut self,
        map: &TerrainMap,
        ledger: &MapCellLedger,
        input: &SensorInput,
        image: GrayImage,
        exposure: &Exposure,
    ) -> Result<Sighting, OnboardError> {
        let sequence = self.frames;
        self.frames = self.frames.wrapping_add(1);
        let acquired = input.acquired_at.as_nanos();
        self.belief_source = Some(exposure.source);
        let search = match self.search(map, &image, exposure, acquired) {
            Ok(search) => search,
            Err(NoView::Unused(reason)) => return Ok(Sighting::Unused(reason, None)),
            Err(NoView::Read(detail)) => return Err(self.map_read(detail)),
        };
        let key = FrameKey {
            stream: StreamId(self.sensor.get()),
            continuity: ContinuityEpoch(0),
            index: sequence,
        };
        if search.acquired_from.is_none()
            && let Some(belief) = search.candidates.first()
            && let Some(reason) =
                self.all_cells_have_a_fix(map, ledger, belief, search.position_radius_m, key)
        {
            return Ok(Sighting::Unused(reason, None));
        }
        let stamp = FrameStamp {
            sequence,
            capture_time_ns: acquired,
        };
        let inputs = (&image, stamp, exposure.unmodelled_lag_s);
        let attempt = match self.choose(map, &search, inputs) {
            Ok(attempt) => attempt,
            Err((NoView::Unused(reason), path)) => return Ok(Sighting::Unused(reason, path)),
            Err((NoView::Read(detail), _)) => return Err(self.map_read(detail)),
        };
        let (view, path) = (Arc::clone(&attempt.reference.view), attempt.reference.path);
        match attempt.estimate {
            Ok(estimate) if !self.confirms(map, &search, (&estimate, &view), acquired) => {
                Ok(Sighting::Unused(names::ACQUISITION_UNCONFIRMED, Some(path)))
            }
            Ok(estimate) => Ok(self.fix(map, ledger, estimate, &view, (key, path))),
            Err(error) => {
                tracing::debug!(sensor = ?self.sensor, %error, "the terrain frame did not match");
                Ok(Sighting::Unused(rejection(&error), Some(path)))
            }
        }
    }

    /// The horizontal variance of the fix of `estimate`, the sum of its
    /// east and north variances: the image geometry and the budget of the
    /// map, the calibration, and the relief.
    pub(super) fn horizontal_variance_m2(
        &self,
        map: &TerrainMap,
        estimate: &Estimate,
        view: &ReferenceView,
    ) -> f64 {
        let geometry = ViewGeometry::of(estimate, view, &self.declared);
        let external = external_covariance_ned(&map.sigmas, self.calibration_sigma_rad, &geometry);
        let image = estimate
            .geometry_covariance
            .fixed_view::<2, 2>(0, 0)
            .trace();
        image + external.fixed_view::<2, 2>(0, 0).trace()
    }

    /// The error of a store that failed or gave changed data.
    fn map_read(&self, detail: String) -> OnboardError {
        OnboardError::MapRead {
            sensor: self.sensor,
            detail,
        }
    }

    /// Whether a frame of the contrast `content` is under the block
    /// contrast floor of the believed view. A belief without a view has
    /// the floor of an oblique camera.
    pub(super) fn is_faint(
        &self,
        config: &TerrainConfig,
        content: &super::content::Content,
        belief: &Belief,
    ) -> bool {
        let off_nadir_rad = belief
            .off_nadir_rad(&self.mount)
            .unwrap_or(config.faint_floor_full_rad);
        content.is_faint(config.faint_floor(off_nadir_rad))
    }

    /// The fix of a solved camera pose, when the filter may take it.
    fn fix(
        &mut self,
        map: &TerrainMap,
        ledger: &MapCellLedger,
        mut estimate: Estimate,
        view: &ReferenceView,
        (frame, path): (FrameKey, &'static str),
    ) -> Sighting {
        if self.inspect {
            self.pending.solved(&estimate);
        }
        let geometry = ViewGeometry::of(&estimate, view, &self.declared);
        // The map error of a fix is the error of the imagery that the
        // camera saw: the cell is that of the matched ground, not of the
        // vehicle.
        let axis = estimate.pose.orientation * -Vector3::z();
        let [lat, lon, ground_height_m] = estimate
            .frame
            .geodetic(estimate.pose.position + axis * geometry.range_m);
        self.ground_height_m = Some(ground_height_m);
        let Ok(cell) = MapCell::of_layer(&map.imagery.layer, [lat, lon]) else {
            return Sighting::Unused(names::OUTSIDE_GRID, Some(path));
        };
        if let Some(reason) = admission(&ledger.label(&cell, frame)) {
            return Sighting::Unused(reason, Some(path));
        }
        let external = external_covariance_ned(&map.sigmas, self.calibration_sigma_rad, &geometry);
        estimate.pose.position = body_position(&estimate.pose, &self.mount);
        match position_fix_observation_composed(&estimate, &self.identity, &external) {
            Ok(observation) => {
                // The fixes come one for each map cell, less often than
                // the assessment window of the filter is long.
                let hold_ns = (map.config.fix_hold_s * 1e9).round() as u64;
                let observation = observation.with_hold(DurationNanos::from_nanos(hold_ns));
                if self.inspect {
                    let model_m = estimate.frame.model_error_m(estimate.pose.position);
                    self.pending.offered(&observation, &external, model_m);
                }
                Sighting::Fix {
                    observation: Box::new(observation),
                    cell,
                    frame,
                    path,
                }
            }
            Err(error) => {
                tracing::debug!(sensor = ?self.sensor, %error, "the terrain fix was not built");
                Sighting::Unused(names::FIX_REFUSED, Some(path))
            }
        }
    }
}
