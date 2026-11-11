//! The terrain-camera means of the reference navigator: each frame of a
//! declared terrain camera is matched against the terrain map, and the
//! match becomes a position fix with a composed covariance.
//!
//! One frame takes these steps:
//!
//! 1. The belief of the navigator at the middle of the exposure seeds and
//!    bounds the search ([`belief`]).
//! 2. The reference view is a stored keyframe inside its envelope, else a
//!    view rendered from the package ([`view`]).
//! 3. `navigate_visual::Localizer` solves the camera pose from the image.
//! 4. The lever arm comes off with the solved attitude, and the external
//!    error is a root sum square of independent terms ([`budget`]).
//! 5. The map cell of the matched ground decides what the filter is told
//!    ([`admission`]).

use std::collections::BTreeMap;
use std::sync::Arc;

use image::GrayImage;
use nalgebra::Vector3;
use navigate_camera::{CameraMount, Distortion, SpectralBand};
use navigate_contract::{ClockDomainId, NavigationSolution, SourceEpoch, SourceId};
use navigate_data::product::{Calibration, ProductKind};
use navigate_data::{LayerKind, ReadPort};
use navigate_fusion::{IngestOutcome, Observation};
use navigate_visual::fusion_bridge::{
    EvidenceIndependence, VisualSourceIdentity, position_fix_observation_composed,
};
use navigate_visual::session::{
    ContinuityEpoch, FrameKey, FusionEligibility, MapCell, MapCellLedger, StreamId,
};
use navigate_visual::{
    CameraModel, Estimate, Frame, FrameStamp, Localizer, LocalizerConfig, PosePrior,
    PyramidalMatcher, ReferenceView,
};

use crate::{
    AcceptanceRecord, AcceptanceSink, DataRequirement, ErrorBudget, MapRefusal, Need, OnboardError,
    Outcome, ProductReport, ReferenceNavigator, SensorData, SensorInput, SensorKind, SensorSpec,
    SensorSuite,
};

mod belief;
mod budget;
mod config;
mod keyframes;
mod map;
mod names;
mod view;

pub use config::TerrainConfig;
pub use map::{
    AccuracySource, AssumedAccuracy, HostGeoid, LayerReport, MapReport, TerrainMapRequest,
};

use belief::{Belief, body_position};
use budget::{ViewGeometry, external_covariance_ned};
use map::TerrainMap;
pub(crate) use names::MEASUREMENT;
use names::rejection;
use view::{NoView, Views, lat_lon};

/// Largest count of map cells whose first fix the navigator records. A
/// zoom 16 cell is some hundreds of metres on a side, so this covers some
/// thousands of kilometres of route.
const MAX_LEDGER_CELLS: usize = 16_384;

/// One declared terrain camera and the state of its capture stream.
#[derive(Debug, Clone)]
pub(crate) struct TerrainCamera {
    sensor: SourceId,
    camera: CameraModel,
    calibration: Calibration,
    mount: CameraMount,
    calibration_sigma_rad: f64,
    identity: VisualSourceIdentity,
    localizer: Localizer<PyramidalMatcher>,
    /// Count of frames taken; the sequence of the next frame.
    frames: u64,
    views: Views,
}

/// What one frame gave.
pub(crate) enum Sighting {
    /// A fix that the filter may take as independent evidence.
    Fix {
        observation: Box<Observation>,
        cell: MapCell,
        frame: FrameKey,
        path: &'static str,
    },
    /// No fix, for the stated reason, on the stated path if a reference
    /// view was made.
    Unused(&'static str, Option<&'static str>),
}

/// The terrain cameras of a suite, their map, and the ledger of the map
/// cells that already gave the filter a fix.
#[derive(Debug, Clone)]
pub(crate) struct TerrainMeans {
    map: Option<Arc<TerrainMap>>,
    /// One ledger for all cameras: two cameras that see one cell share its
    /// map error.
    ledger: MapCellLedger,
    /// Keyed by `SourceId::get`.
    cameras: BTreeMap<u32, TerrainCamera>,
}

/// What the filter is told about a fix with `label`: `None` offers it as
/// an independent position fix, and a reason withholds it. The filter has
/// no state for an error that fixes share, so a fix that shares the map
/// error of an earlier fix is withheld. A declared correlated error of the
/// camera changes this answer here and nowhere else.
pub(crate) fn admission(label: &FusionEligibility) -> Option<&'static str> {
    // The navigator made the fix from the image and the map alone, so the
    // map cell is the one error that it can share with the filter state.
    match EvidenceIndependence::ValidatedIndependent.restricted_by(label) {
        EvidenceIndependence::ValidatedIndependent => None,
        EvidenceIndependence::Unknown => Some(match label {
            FusionEligibility::SharedMapError { .. } => names::SHARED_MAP_ERROR,
            _ => names::UNTRACKED_MAP_CELL,
        }),
    }
}

impl TerrainCamera {
    fn new(spec: &SensorSpec, clock: ClockDomainId) -> Result<Self, OnboardError> {
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
        let k = imager.intrinsics;
        let camera = CameraModel {
            width: k.width,
            height: k.height,
            fx: k.fx,
            fy: k.fy,
            cx: k.cx,
            cy: k.cy,
        };
        let usable = calibration_sigma_rad.is_finite() && calibration_sigma_rad >= 0.0;
        if camera.validate().is_err() || !usable {
            return Err(invalid(format!("{:?} intrinsics or budget", spec.id)));
        }
        Ok(Self {
            sensor: spec.id,
            calibration: keyframes::calibration_of(&camera),
            camera,
            mount,
            calibration_sigma_rad,
            identity: VisualSourceIdentity {
                source: spec.id,
                epoch: SourceEpoch::new(0),
                clock,
            },
            localizer: Localizer::new(PyramidalMatcher, LocalizerConfig::default())
                .map_err(|e| invalid(e.to_string()))?,
            frames: 0,
            views: Views::default(),
        })
    }

    /// The image of a frame, when its size is the declared size.
    fn image(&self, width: u32, height: u32, luma: &[u8]) -> Result<GrayImage, OnboardError> {
        let declared = (self.camera.width, self.camera.height);
        let image = ((width, height) == declared)
            .then(|| GrayImage::from_raw(width, height, luma.to_vec()))
            .flatten();
        image.ok_or(OnboardError::FrameSize {
            sensor: self.sensor,
            declared,
            frame: (width, height),
            bytes: luma.len(),
        })
    }

    /// Matches one frame and gives the fix, or why there is none.
    fn sight(
        &mut self,
        map: &TerrainMap,
        ledger: &MapCellLedger,
        input: &SensorInput,
        image: GrayImage,
        solution: Option<&NavigationSolution>,
    ) -> Result<Sighting, OnboardError> {
        let sequence = self.frames;
        self.frames = self.frames.wrapping_add(1);
        let (acquired, delivered) = (input.acquired_at.as_nanos(), input.delivered_at.as_nanos());
        let Some(lag_ns) = delivered.checked_sub(acquired) else {
            return Ok(Sighting::Unused(names::ACQUIRED_AFTER_DELIVERY, None));
        };
        let lag_s = lag_ns as f64 * 1e-9;
        if lag_s > map.config.max_capture_lag_s {
            return Ok(Sighting::Unused(names::CAPTURE_LAG, None));
        }
        let Some(belief) = solution.and_then(|s| Belief::before(s, lag_s)) else {
            return Ok(Sighting::Unused(names::NO_SOLUTION, None));
        };
        let (position_radius_m, attitude_radius_rad) = map.config.search(belief.sigma_h_m, lag_s);
        let reference = match self.views.reference(
            map,
            &self.camera,
            self.calibration,
            &self.mount,
            &belief,
            position_radius_m,
        ) {
            Ok(reference) => reference,
            Err(NoView::Unused(reason)) => return Ok(Sighting::Unused(reason, None)),
            Err(NoView::Read(detail)) => {
                return Err(OnboardError::MapRead {
                    sensor: self.sensor,
                    detail,
                });
            }
        };
        let (view, path) = (&*reference.view, reference.path);
        let Ok(pose) = belief.camera_in(view.frame, &self.mount) else {
            return Ok(Sighting::Unused(names::BELIEF_OUT_OF_RANGE, Some(path)));
        };
        let prior = PosePrior {
            pose,
            position_radius_m,
            attitude_radius_rad,
        };
        let frame = Frame {
            stamp: FrameStamp {
                sequence,
                capture_time_ns: acquired,
            },
            camera: self.camera,
            image,
        };
        match self.localizer.estimate_blocking(&frame, view, &prior) {
            Ok(estimate) => Ok(self.fix(map, ledger, estimate, view, path)),
            Err(error) => {
                tracing::debug!(sensor = ?self.sensor, %error, "the terrain frame did not match");
                Ok(Sighting::Unused(rejection(&error), Some(path)))
            }
        }
    }

    /// The fix of a solved camera pose, when the filter may take it.
    fn fix(
        &self,
        map: &TerrainMap,
        ledger: &MapCellLedger,
        mut estimate: Estimate,
        view: &ReferenceView,
        path: &'static str,
    ) -> Sighting {
        let geometry = ViewGeometry::of(&estimate, view);
        // The map error of a fix is the error of the imagery that the
        // camera saw: the cell is that of the matched ground, not of the
        // vehicle.
        let axis = estimate.pose.orientation * -Vector3::z();
        let ground = lat_lon(
            estimate.frame,
            estimate.pose.position + axis * geometry.range_m,
        );
        let Ok(cell) = MapCell::of_layer(&map.imagery.layer, ground) else {
            return Sighting::Unused(names::OUTSIDE_GRID, Some(path));
        };
        let frame = FrameKey {
            stream: StreamId(self.sensor.get()),
            continuity: ContinuityEpoch(0),
            index: estimate.stamp.sequence,
        };
        if let Some(reason) = admission(&ledger.label(&cell, frame)) {
            return Sighting::Unused(reason, Some(path));
        }
        let external = external_covariance_ned(&map.sigmas, self.calibration_sigma_rad, &geometry);
        estimate.pose.position = body_position(&estimate.pose, &self.mount);
        match position_fix_observation_composed(&estimate, &self.identity, &external) {
            Ok(observation) => Sighting::Fix {
                observation: Box::new(observation),
                cell,
                frame,
                path,
            },
            Err(error) => {
                tracing::debug!(sensor = ?self.sensor, %error, "the terrain fix was not built");
                Sighting::Unused(names::FIX_REFUSED, Some(path))
            }
        }
    }
}

impl TerrainMeans {
    /// The means of the terrain cameras of `suite`.
    pub(crate) fn new(suite: &SensorSuite, clock: ClockDomainId) -> Result<Self, OnboardError> {
        let mut cameras = BTreeMap::new();
        for spec in suite.of_kind(SensorKind::TerrainCamera) {
            cameras.insert(spec.id.get(), TerrainCamera::new(spec, clock)?);
        }
        Ok(Self {
            map: None,
            ledger: MapCellLedger::new(MAX_LEDGER_CELLS),
            cameras,
        })
    }

    /// Whether the suite declares a terrain camera.
    pub(crate) fn has_cameras(&self) -> bool {
        !self.cameras.is_empty()
    }

    /// The report of the loaded map.
    pub(crate) fn report(&self) -> Option<&MapReport> {
        self.map.as_ref().map(|map| &map.report)
    }

    /// The data that the means needs or can use: imagery and elevation,
    /// and keyframes if the host built them. None without a camera.
    pub(crate) fn requirements(&self) -> Vec<DataRequirement> {
        if !self.has_cameras() {
            return Vec::new();
        }
        let layer = |kind| DataRequirement {
            layer: Some(kind),
            product: None,
            altitude_bands: None,
            need: Need::Required,
        };
        vec![
            layer(LayerKind::Imagery),
            layer(LayerKind::Elevation),
            DataRequirement {
                layer: None,
                product: Some(ProductKind::Keyframes),
                altitude_bands: None,
                need: Need::Optional,
            },
        ]
    }

    /// What the means did with each product chunk at load.
    pub(crate) fn products(&self) -> &[ProductReport] {
        self.report()
            .map_or(&[], |report| report.products.as_slice())
    }
}

impl ReferenceNavigator {
    /// Loads the terrain map: the package `request.package` on the store
    /// that the host opened. The navigator reads the store and never
    /// fetches. A second load replaces the map and keeps the record of the
    /// map cells that already gave a fix, because that record names the
    /// source of the imagery and not the package.
    ///
    /// # Errors
    ///
    /// [`OnboardError::MapRefused`] with the reason: no terrain camera in
    /// the suite, a package that is missing, changed, or of another
    /// version, a missing or unreadable layer, an accuracy that no one
    /// states, and a product list that is missing or of another package.
    /// A refused load leaves the navigator as it was.
    pub fn load_terrain_map<S>(
        &mut self,
        store: Arc<S>,
        request: &TerrainMapRequest,
        geoid: Option<HostGeoid>,
    ) -> Result<&MapReport, OnboardError>
    where
        S: ReadPort + Send + Sync + 'static,
        S::File: Send + Sync + 'static,
    {
        let refused = |refusal| OnboardError::MapRefused {
            package: request.package,
            refusal,
        };
        if !self.terrain.has_cameras() {
            return Err(refused(MapRefusal::NoTerrainCamera));
        }
        let calibrations: Vec<Calibration> = self
            .terrain
            .cameras
            .values()
            .map(|camera| camera.calibration)
            .collect();
        let loaded = map::load(store, request, geoid, &calibrations).map_err(refused)?;
        for camera in self.terrain.cameras.values_mut() {
            camera.views = Views::default();
        }
        let map = self.terrain.map.insert(Arc::new(loaded));
        Ok(&map.report)
    }

    /// What the navigator loaded as its terrain map, or `None` before a
    /// load.
    #[must_use]
    pub fn terrain_map(&self) -> Option<&MapReport> {
        self.terrain.report()
    }

    /// Takes one frame of a terrain camera.
    pub(super) fn terrain_frame(
        &mut self,
        input: &SensorInput,
        sink: &mut dyn AcceptanceSink,
    ) -> Result<(), OnboardError> {
        let SensorData::TerrainFrame {
            width,
            height,
            luma,
        } = &input.data
        else {
            return Ok(());
        };
        let Some(camera) = self.terrain.cameras.get_mut(&input.sensor.get()) else {
            return Err(OnboardError::UnknownSensor(input.sensor));
        };
        let image = camera.image(*width, *height, luma)?;
        let Some(map) = self.terrain.map.clone() else {
            return Err(OnboardError::MapNotLoaded {
                sensor: input.sensor,
            });
        };
        let solution = self.filter.tick(input.delivered_at);
        let sighting = camera.sight(&map, &self.terrain.ledger, input, image, solution.as_ref())?;
        let (outcome, context) = match sighting {
            Sighting::Unused(reason, path) => (Outcome::Unused(reason), path),
            Sighting::Fix {
                observation,
                cell,
                frame,
                path,
            } => match self.filter.ingest(&observation, input.delivered_at) {
                IngestOutcome::Accepted => {
                    // Only a fix that the filter took carries the map
                    // error of its cell into the state.
                    self.terrain.ledger.commit(cell, frame);
                    (Outcome::Accepted, Some(path))
                }
                IngestOutcome::Rejected(reason) => (Outcome::Rejected(reason), Some(path)),
            },
        };
        sink.record(AcceptanceRecord {
            sensor: input.sensor,
            acquired_at: input.acquired_at,
            measurement: MEASUREMENT,
            outcome,
            context,
        });
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests;
