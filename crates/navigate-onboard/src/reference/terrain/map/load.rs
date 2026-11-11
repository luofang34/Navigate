//! Loading the terrain map. The load refuses what would fail on each frame:
//! a package that is not the expected one, a layer that the renderer cannot
//! read, and an accuracy that no one states.

use std::collections::BTreeSet;
use std::sync::Arc;

use navigate_data::index::IndexEntry;
use navigate_data::product::{
    Calibration, KindName, ListEntry, ProductKind, ProductUse, open_chunk, open_list, read_recipe,
};
use navigate_data::tile::{ground_sample_distance_m, tile_lat_lon};
use navigate_data::{
    EllipsoidalHeight, Encoding, Layer, LayerKind, Package, RandomAccess, ReadPort, Sha256Digest,
};
use navigate_geodesy::GeoidUndulation;
use navigate_visual_render::KeyframeSpec;

use super::{
    AccuracySource, HostGeoid, LayerReport, LayerUse, MapReport, PackageReader, TerrainMap,
    TerrainMapRequest, decided,
};
use crate::reference::terrain::budget::MapSigmas;
use crate::reference::terrain::keyframes::{Keyframe, KeyframeSet};
use crate::{MapRefusal, ProductReport};

/// Pixels on a tile side that the scene assembly of the renderer reads.
const TILE_PIXELS: u32 = 256;

/// Opens the package of `request` on `store` and checks it for the
/// cameras of `cameras`.
pub(crate) fn load<S>(
    store: Arc<S>,
    request: &TerrainMapRequest,
    geoid: Option<HostGeoid>,
    cameras: &[Calibration],
) -> Result<TerrainMap, MapRefusal>
where
    S: ReadPort + Send + Sync + 'static,
    S::File: Send + Sync + 'static,
{
    request.config.validate()?;
    let package =
        pollster::block_on(Package::open(&*store, request.package)).map_err(MapRefusal::Package)?;
    let mut blobs = BTreeSet::new();
    let (imagery, imagery_report) = layer(&package, LayerKind::Imagery, request, &mut blobs)?;
    let (elevation, elevation_report) = layer(&package, LayerKind::Elevation, request, &mut blobs)?;
    let undulation_bound_m = undulation_bound_m(&elevation, geoid.as_ref());
    let (products, keyframes) = products(&*store, request, &blobs, cameras)?;
    // The undulation bound is a limit, not a one-sigma value. The budget
    // takes it as one, which is the cautious reading.
    let sigmas = MapSigmas {
        horizontal_m: imagery_report.sigma_m,
        height_m: elevation_report.sigma_m.hypot(undulation_bound_m),
    };
    let report = MapReport {
        package: request.package,
        imagery: imagery_report,
        elevation: elevation_report,
        undulation_bound_m,
        products,
        keyframes: keyframes.iter().map(|set| set.entries.len()).sum(),
    };
    Ok(TerrainMap {
        reader: Box::new(PackageReader {
            store,
            package,
            geoid,
        }),
        report,
        imagery,
        elevation,
        sigmas,
        keyframes,
        config: request.config,
    })
}

/// Refuses a layer that the scene assembly of the renderer cannot read.
fn readable(layer: &Layer) -> Result<(), MapRefusal> {
    let kind = layer.key.kind;
    let refuse = |detail: String| MapRefusal::UnsupportedLayer {
        layer: kind,
        detail,
    };
    if layer.tile_pixels != TILE_PIXELS {
        return Err(refuse(format!(
            "tiles of {} pixels, not {TILE_PIXELS}",
            layer.tile_pixels
        )));
    }
    let decodes = match kind {
        LayerKind::Imagery => matches!(layer.encoding, Encoding::Png | Encoding::Jpeg),
        _ => layer.encoding == Encoding::TerrariumPng,
    };
    if !decodes {
        return Err(refuse(format!("encoding {:?}", layer.encoding)));
    }
    Ok(())
}

/// The present one-sigma of a layer: horizontal for imagery, vertical for
/// elevation.
fn sigma(layer: &Layer, request: &TerrainMapRequest) -> Result<(f64, AccuracySource), MapRefusal> {
    let kind = layer.key.kind;
    let stated = match kind {
        LayerKind::Imagery => layer.accuracy.horizontal_sigma_m(request.today),
        _ => layer.accuracy.vertical_mm.map(|mm| f64::from(mm) / 1_000.0),
    };
    let assumed = request.assumed_accuracy.map(|a| match kind {
        LayerKind::Imagery => a.horizontal_sigma_m,
        _ => a.vertical_sigma_m,
    });
    let (value, from) = match (stated, assumed) {
        (Some(value), _) => (value, AccuracySource::Package),
        (None, Some(value)) => (value, AccuracySource::Assumed),
        (None, None) => return Err(MapRefusal::UnknownAccuracy(kind)),
    };
    if !(value.is_finite() && value >= 0.0) {
        return Err(MapRefusal::Request {
            field: "assumed accuracy",
        });
    }
    Ok((value, from))
}

/// Checks one layer and lists what it holds. The digests of its tile blobs
/// go into `blobs`.
fn layer<F: RandomAccess>(
    package: &Package<F>,
    kind: LayerKind,
    request: &TerrainMapRequest,
    blobs: &mut BTreeSet<Sha256Digest>,
) -> Result<(LayerUse, LayerReport), MapRefusal> {
    let layer = *package
        .manifest()
        .layer_of_kind(kind)
        .ok_or(MapRefusal::MissingLayer(kind))?;
    readable(&layer)?;
    let (sigma_m, sigma_from) = sigma(&layer, request)?;
    let entries =
        pollster::block_on(package.layer_entries(layer.key)).map_err(MapRefusal::Package)?;
    let mut zooms = BTreeSet::new();
    let mut first = None;
    for (address, entry) in &entries {
        if let IndexEntry::Tile { blob, .. } = entry {
            zooms.insert(address.zoom);
            blobs.insert(*blob);
            first.get_or_insert(*address);
        }
    }
    let first = first.ok_or(MapRefusal::EmptyLayer(kind))?;
    let [lat_deg, _] = tile_lat_lon(
        f64::from(first.x) + 0.5,
        f64::from(first.y) + 0.5,
        first.zoom,
    );
    let zooms: Vec<u32> = zooms.into_iter().collect();
    let finest = zooms.last().copied().unwrap_or(first.zoom);
    let report = LayerReport {
        key: layer.key,
        revision: layer.revision,
        ground_sample_m: ground_sample_distance_m(lat_deg, finest, layer.tile_pixels),
        zooms: zooms.clone(),
        tiles: layer.index.tiles,
        gaps: layer.index.gaps,
        sigma_m,
        sigma_from,
    };
    Ok((
        LayerUse {
            layer,
            zooms,
            lat_deg,
        },
        report,
    ))
}

/// The bound of the height error from the geoid undulation: zero for
/// ellipsoidal heights and for orthometric heights with the host model.
fn undulation_bound_m(elevation: &LayerUse, geoid: Option<&HostGeoid>) -> f64 {
    let geoid = geoid.map(|(model, undulation)| (*model, &**undulation as &dyn GeoidUndulation));
    let Some(datum) = elevation.layer.vertical_datum else {
        return navigate_data::UNKNOWN_UNDULATION_BOUND_M;
    };
    match datum.to_ellipsoidal(0.0, elevation.lat_deg.to_radians(), 0.0, geoid) {
        EllipsoidalHeight::Known(_) => 0.0,
        EllipsoidalHeight::UnknownUndulation { bound_m, .. } => bound_m,
    }
}

/// Reads the product list and decides the use of each chunk.
fn products<S: ReadPort + ?Sized>(
    store: &S,
    request: &TerrainMapRequest,
    blobs: &BTreeSet<Sha256Digest>,
    cameras: &[Calibration],
) -> Result<(Vec<ProductReport>, Vec<KeyframeSet>), MapRefusal> {
    let Some(list) = request.products else {
        return Ok((Vec::new(), Vec::new()));
    };
    let list = pollster::block_on(open_list(store, list, request.package))
        .map_err(MapRefusal::ProductList)?;
    let mut reports = Vec::with_capacity(list.chunks.len());
    let mut sets = Vec::new();
    for entry in &list.chunks {
        let decision = match &entry.kind {
            KindName::Known(ProductKind::Keyframes) => {
                match keyframes(store, entry, request, blobs, cameras) {
                    Ok(set) => {
                        sets.push(set);
                        ProductUse::Use
                    }
                    Err(decision) => decision,
                }
            }
            KindName::Known(_) => {
                ProductUse::Fallback("the reference navigator does not use this product".into())
            }
            KindName::Unknown(_) => ProductUse::Fallback("unknown product kind, not read".into()),
        };
        reports.push(decided(entry, decision));
    }
    Ok((reports, sets))
}

/// The keyframes of one chunk, or why the navigator does not use them. A
/// fallback leaves the frames to the reference that the navigator renders
/// from the package.
fn keyframes<S: ReadPort + ?Sized>(
    store: &S,
    entry: &ListEntry,
    request: &TerrainMapRequest,
    blobs: &BTreeSet<Sha256Digest>,
    cameras: &[Calibration],
) -> Result<KeyframeSet, ProductUse> {
    let refuse = |reason: String| ProductUse::Refusal(reason);
    let chunk =
        pollster::block_on(open_chunk(store, entry.chunk)).map_err(|e| refuse(e.to_string()))?;
    if !chunk.applies_to(blobs) {
        return Err(refuse(
            "the package does not hold each input blob of the chunk".into(),
        ));
    }
    if request.route.is_some() && chunk.binding.route != request.route {
        return Err(ProductUse::Fallback(
            "the chunk is for another route".into(),
        ));
    }
    let (Some(envelope), Some(calibration)) = (chunk.binding.envelope, chunk.binding.calibration)
    else {
        return Err(refuse(
            "the chunk states no envelope or no calibration".into(),
        ));
    };
    if !cameras.contains(&calibration) {
        return Err(ProductUse::Fallback(
            "the calibration is not that of a declared terrain camera".into(),
        ));
    }
    let recipe =
        pollster::block_on(read_recipe(store, chunk.recipe)).map_err(|e| refuse(e.to_string()))?;
    let depression_rad = KeyframeSpec::recipe_depression_rad(&recipe)
        .ok_or_else(|| refuse("the recipe is of another renderer or encoding".into()))?;
    let entries: Option<Vec<Keyframe>> = chunk
        .entries
        .iter()
        .map(|e| {
            e.pose.map(|pose| Keyframe {
                pose,
                blob: e.blob,
                bytes: e.bytes,
            })
        })
        .collect();
    let entries = entries.ok_or_else(|| refuse("an entry states no pose".into()))?;
    Ok(KeyframeSet {
        calibration,
        envelope,
        depression_rad,
        entries,
    })
}
