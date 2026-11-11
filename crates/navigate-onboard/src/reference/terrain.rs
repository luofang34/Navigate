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
//!
//! Before step 2, a frame that can only show map cells with a fix is
//! recorded and not matched ([`camera`]), and a frame larger than the
//! working resolution is resampled to it.

use std::collections::BTreeMap;
use std::sync::Arc;

use navigate_camera::Intrinsics;
use navigate_contract::{ClockDomainId, SourceId};
use navigate_data::product::ProductKind;
use navigate_data::{LayerKind, ReadPort};
use navigate_fusion::{IngestOutcome, Observation};
use navigate_visual::fusion_bridge::EvidenceIndependence;
use navigate_visual::session::{FrameKey, FusionEligibility, MapCell, MapCellLedger};

use crate::{
    AcceptanceRecord, AcceptanceSink, DataRequirement, MapRefusal, Need, OnboardError, Outcome,
    ProductReport, ReferenceNavigator, SensorData, SensorInput, SensorKind, SensorSuite,
};

mod belief;
mod budget;
mod camera;
mod config;
mod content;
mod keyframes;
mod map;
mod names;
mod view;

pub use belief::BeliefSource;
pub use config::{TerrainConfig, WorkingCamera, WorkingLimit};
pub use map::{
    AccuracySource, AssumedAccuracy, HostGeoid, LayerReport, MapReport, TerrainMapRequest,
};

use belief::at_exposure;
use camera::TerrainCamera;
use map::TerrainMap;
pub(crate) use names::MEASUREMENT;

/// Largest count of map cells whose first fix the navigator records. A
/// zoom 16 cell is some hundreds of metres on a side, so this covers some
/// thousands of kilometres of route.
const MAX_LEDGER_CELLS: usize = 16_384;

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

    /// Forgets which map cells gave a fix and what each camera kept: the
    /// filter has a new state, and no map error is in it.
    pub(crate) fn forget_fixes(&mut self) {
        self.ledger = MapCellLedger::new(MAX_LEDGER_CELLS);
        for camera in self.cameras.values_mut() {
            camera.reset();
        }
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
    /// The load and every later frame read the store by blocking on each
    /// read, because a navigator step is not asynchronous. The store must
    /// therefore complete each read without help from the calling thread,
    /// as a [`DirectoryStore`](navigate_data::DirectoryStore) and a
    /// [`MemoryStore`](navigate_data::MemoryStore) do. A store whose reads
    /// complete on an event loop that the caller runs, such as a browser
    /// store on the page's thread, would never return: a host with such a
    /// store awaits [`navigate_data::resident_copy`] of the package and its
    /// product list, and gives this load the memory store of the copy.
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
        let cameras: Vec<Intrinsics> = self
            .terrain
            .cameras
            .values()
            .map(|camera| camera.declared)
            .collect();
        let loaded = map::load(store, request, geoid, &cameras).map_err(refused)?;
        for camera in self.terrain.cameras.values_mut() {
            camera.reset();
        }
        let map = self.terrain.map.insert(Arc::new(loaded));
        Ok(&map.report)
    }

    /// The camera of the last reference view and match of the terrain
    /// camera `sensor`: the working resolution, the ground size of its
    /// pixels, and what set it. `None` before the first matched frame.
    #[must_use]
    pub fn terrain_working_camera(&self, sensor: SourceId) -> Option<WorkingCamera> {
        self.terrain.cameras.get(&sensor.get())?.working()
    }

    /// Where the belief of the last frame of the terrain camera `sensor`
    /// came from: the state of the filter at the exposure, or the solution
    /// on delivery moved back. `None` before the first frame with a
    /// belief.
    #[must_use]
    pub fn terrain_belief_source(&self, sensor: SourceId) -> Option<BeliefSource> {
        self.terrain.cameras.get(&sensor.get())?.belief_source()
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
        // A frame with no structure gives no match, and it is found before
        // the filter, the map, or the matcher do any work for it.
        if content::Content::of(&image).is_featureless(map.config.min_structure) {
            sink.record(AcceptanceRecord {
                sensor: input.sensor,
                acquired_at: input.acquired_at,
                measurement: MEASUREMENT,
                outcome: Outcome::Unused(names::FEATURELESS_FRAME),
                context: None,
            });
            return Ok(());
        }
        let times = (input.acquired_at, input.delivered_at);
        let sighting = match at_exposure(&mut self.filter, times, map.config.max_capture_lag_s) {
            Ok(exposure) => camera.sight(&map, &self.terrain.ledger, input, image, &exposure)?,
            Err(reason) => Sighting::Unused(reason, None),
        };
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
