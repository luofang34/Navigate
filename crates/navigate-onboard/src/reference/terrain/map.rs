//! The terrain map of the navigator: one navigation data package that the
//! host installed, opened through the host's read port.

use std::fmt;
use std::sync::Arc;

use navigate_data::product::{Calibration, ProductUse};
use navigate_data::tile::ground_sample_distance_m;
use navigate_data::{
    DataError, DayNumber, GeoidModel, Layer, LayerKey, Package, ReadPort, Revision, Sha256Digest,
    read_blob,
};
use navigate_geodesy::GeoidUndulation;
use navigate_visual::MapRevision;
use navigate_visual_render::{RenderError, SceneRequest, SceneWindow, scene_around};

use super::budget::MapSigmas;
use super::config::TerrainConfig;
use super::keyframes::KeyframeSet;
use crate::ProductReport;

mod load;
pub(crate) use load::load;

/// The geoid model of the host and the model that it implements, for an
/// elevation layer with orthometric heights.
pub type HostGeoid = (GeoidModel, Arc<dyn GeoidUndulation + Send + Sync>);

/// The accuracy that the host declares for a layer that states none.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AssumedAccuracy {
    /// Horizontal one-sigma of the imagery, each axis, metres.
    pub horizontal_sigma_m: f64,
    /// Vertical one-sigma of the elevation, metres.
    pub vertical_sigma_m: f64,
}

/// What the host states when it gives the navigator a store.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct TerrainMapRequest {
    /// The identity of the package that the host installed: the digest of
    /// its canonical manifest. The navigator refuses another package.
    pub package: Sha256Digest,
    /// The digest of the product list of the mission, if the host built
    /// derived products. A read port cannot list a store, so the navigator
    /// finds products only through this list.
    pub products: Option<Sha256Digest>,
    /// The digest of the route of the mission. With it, the navigator
    /// uses only the route products of that route.
    pub route: Option<Sha256Digest>,
    /// The present day. The navigator computes the present accuracy of
    /// each layer from it and reads no clock.
    pub today: DayNumber,
    /// The accuracy to take for a layer that states none.
    pub assumed_accuracy: Option<AssumedAccuracy>,
    /// Tuning of the means.
    pub config: TerrainConfig,
}

impl TerrainMapRequest {
    /// A request for `package` on day `today`, with no products, no
    /// assumed accuracy, and the default tuning.
    #[must_use]
    pub fn new(package: Sha256Digest, today: DayNumber) -> Self {
        Self {
            package,
            products: None,
            route: None,
            today,
            assumed_accuracy: None,
            config: TerrainConfig::default(),
        }
    }
}

/// Where the accuracy of a layer came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccuracySource {
    /// The package states it.
    Package,
    /// The package states none, and the host assumed it.
    Assumed,
}

/// What the navigator loaded of one layer.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerReport {
    /// Kind and source.
    pub key: LayerKey,
    /// The content version that the source states.
    pub revision: Revision,
    /// The zoom levels that hold tiles, in increasing order.
    pub zooms: Vec<u32>,
    /// Ground size of one pixel at the finest zoom, at the latitude of the
    /// layer, metres.
    pub ground_sample_m: f64,
    /// Count of tiles.
    pub tiles: u64,
    /// Count of declared gaps.
    pub gaps: u64,
    /// One-sigma error on the present day, metres: horizontal for imagery,
    /// vertical for elevation.
    pub sigma_m: f64,
    /// Where that value came from.
    pub sigma_from: AccuracySource,
}

/// What the navigator loaded: the answer of a successful load.
#[derive(Debug, Clone, PartialEq)]
pub struct MapReport {
    /// The package identity.
    pub package: Sha256Digest,
    /// The imagery layer.
    pub imagery: LayerReport,
    /// The elevation layer.
    pub elevation: LayerReport,
    /// Bound on the height error from an unknown geoid undulation, metres;
    /// zero when the heights are ellipsoidal or the host geoid applies.
    pub undulation_bound_m: f64,
    /// What the navigator did with each chunk of the product list.
    pub products: Vec<ProductReport>,
    /// Count of route keyframes in use.
    pub keyframes: usize,
}

/// Blocking reads of one opened package. The trait hides the store type,
/// so the navigator is not generic over it.
pub(crate) trait MapReader: Send + Sync {
    /// Assembles a render scene from the tiles of the package.
    fn scene(&self, request: &SceneRequest) -> Result<SceneWindow, RenderError>;

    /// Reads one blob and checks it.
    fn blob(&self, digest: Sha256Digest, bytes: u64) -> Result<Vec<u8>, DataError>;
}

/// A package on the host store.
///
/// Each read waits for the store inside a navigator step, which is not
/// asynchronous. The store must complete a read without help from the
/// calling thread, as a directory store and a memory store do.
pub(crate) struct PackageReader<S: ReadPort> {
    pub store: Arc<S>,
    pub package: Package<S::File>,
    pub geoid: Option<HostGeoid>,
}

impl<S> MapReader for PackageReader<S>
where
    S: ReadPort + Send + Sync,
    S::File: Send + Sync,
{
    fn scene(&self, request: &SceneRequest) -> Result<SceneWindow, RenderError> {
        let geoid = self
            .geoid
            .as_ref()
            .map(|(model, undulation)| (*model, &**undulation as &dyn GeoidUndulation));
        pollster::block_on(scene_around(&self.package, &*self.store, request, geoid))
    }

    fn blob(&self, digest: Sha256Digest, bytes: u64) -> Result<Vec<u8>, DataError> {
        pollster::block_on(read_blob(&*self.store, digest, bytes))
    }
}

/// One layer as the means uses it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct LayerUse {
    pub layer: Layer,
    /// Zoom levels that hold tiles, in increasing order.
    pub zooms: Vec<u32>,
    /// Latitude of the layer, degrees, for the size of its pixels.
    pub lat_deg: f64,
}

impl LayerUse {
    /// The coarsest zoom whose pixels are no larger than `footprint_m` on
    /// the ground, or the finest zoom of the layer: one texel for one
    /// pixel of the camera.
    pub(crate) fn zoom_for(&self, footprint_m: f64) -> Option<u32> {
        let fits = |zoom: &&u32| {
            ground_sample_distance_m(self.lat_deg, **zoom, self.layer.tile_pixels) <= footprint_m
        };
        self.zooms.iter().find(fits).or(self.zooms.last()).copied()
    }
}

/// The loaded terrain map. It does not change after the load, so each
/// snapshot of a navigator shares it.
pub(crate) struct TerrainMap {
    pub reader: Box<dyn MapReader>,
    pub report: MapReport,
    pub imagery: LayerUse,
    pub elevation: LayerUse,
    pub sigmas: MapSigmas,
    pub keyframes: Vec<KeyframeSet>,
    pub config: TerrainConfig,
}

impl TerrainMap {
    /// The map revision that each reference and fix of this map names.
    pub(crate) fn revision(&self) -> MapRevision {
        let identity = self.report.package.to_string();
        MapRevision {
            release_id: identity.clone(),
            manifest_sha256: identity,
        }
    }

    /// The keyframe sets that were rendered for a camera of `calibration`.
    pub(crate) fn keyframes_for(
        &self,
        calibration: Calibration,
    ) -> impl Iterator<Item = &KeyframeSet> {
        self.keyframes
            .iter()
            .filter(move |set| set.calibration == calibration)
    }
}

impl fmt::Debug for TerrainMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TerrainMap")
            .field("report", &self.report)
            .finish_non_exhaustive()
    }
}

/// A product report with a decision.
pub(crate) fn decided(
    entry: &navigate_data::product::ListEntry,
    decision: ProductUse,
) -> ProductReport {
    ProductReport {
        kind: entry.kind.clone(),
        chunk: entry.chunk,
        decision,
    }
}
