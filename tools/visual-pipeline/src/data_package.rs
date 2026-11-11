//! The `package` subcommand: a navigation data package for a flight plan.
//! The tool is the host: it fetches through URL templates or reads a
//! catalog, and `navigate-data` plans, checks, and seals.

mod fetch;
mod ground;

pub use fetch::{CatalogFetch, TemplateFetch, TemplateLayer};
pub use ground::{GROUND_ZOOM, TerrariumGround};

use crate::{PipelineError, json_file::read_json};
use navigate_data::{
    Accuracy, DirectoryStore, Encoding, FetchPort, LayerKind, LayerLimits, LayerTemplate,
    PlannedCamera, PlannedLeg, ResourceRequest, SourceRef, TileKey, VerticalDatum, plan_resources,
    plan_tiles, prepare,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::PathBuf;

const FETCH_THREADS: usize = 16;

/// One waypoint of a plan file. The altitude is above mean sea level, the
/// reference of Terrarium heights.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PlanWaypoint {
    /// Latitude in degrees.
    pub lat_deg: f64,
    /// Longitude in degrees.
    pub lon_deg: f64,
    /// Altitude above mean sea level in metres.
    pub altitude_m: f64,
}

/// One camera of a plan file.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct PlanCamera {
    /// Image width in pixels.
    pub width_px: u32,
    /// Image height in pixels.
    pub height_px: u32,
    /// Horizontal field of view in degrees.
    pub horizontal_fov_deg: f64,
    /// Angle of the optical axis below the horizontal in degrees.
    pub depression_deg: f64,
}

/// The plan file of the `package` subcommand.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanFile {
    /// Release identifier of the package.
    pub release_id: String,
    /// Waypoints in fly order.
    pub waypoints: Vec<PlanWaypoint>,
    /// Cameras of the vehicle.
    pub cameras: Vec<PlanCamera>,
    /// Largest search radius around the navigation estimate, metres.
    pub search_radius_m: f64,
    /// Allowed deviation from the track, metres.
    pub deviation_m: f64,
    /// Farthest slant range that the navigator uses, metres.
    pub max_range_m: f64,
}

impl PlanFile {
    /// The legs between the waypoints.
    pub fn legs(&self) -> Vec<PlannedLeg> {
        self.waypoints
            .windows(2)
            .filter_map(|w| match w {
                [a, b] => Some(PlannedLeg {
                    from: [a.lat_deg, a.lon_deg, a.altitude_m],
                    to: [b.lat_deg, b.lon_deg, b.altitude_m],
                }),
                _ => None,
            })
            .collect()
    }

    /// The resource request with the measured layer sizes.
    pub fn request(&self) -> ResourceRequest {
        ResourceRequest {
            legs: self.legs(),
            cameras: self
                .cameras
                .iter()
                .map(|c| PlannedCamera {
                    focal_px: f64::from(c.width_px)
                        / 2.0
                        / (c.horizontal_fov_deg.to_radians() / 2.0).tan(),
                    width_px: c.width_px,
                    height_px: c.height_px,
                    depression_rad: c.depression_deg.to_radians(),
                })
                .collect(),
            search_radius_m: self.search_radius_m,
            deviation_m: self.deviation_m,
            max_range_m: self.max_range_m,
            imagery: LayerLimits::imagery(),
            elevation: LayerLimits::terrarium(),
            elevation_offset: 3,
        }
    }
}

/// Inputs of the `package` subcommand.
#[derive(Clone, Debug)]
pub struct PackageOptions {
    /// The plan file.
    pub plan: PathBuf,
    /// The store directory (`chunks/` and `packages/`).
    pub store: PathBuf,
    /// The tile cache directory.
    pub cache: PathBuf,
    /// Imagery template and source.
    pub imagery: TemplateLayer,
    /// Terrarium template and source.
    pub elevation: TemplateLayer,
    /// Largest tile count to fetch.
    pub max_tiles: usize,
    /// Compute and print the resource plan only.
    pub plan_only: bool,
}

/// The layer templates of a Terrarium elevation layer and a JPEG imagery
/// layer, with unknown accuracy.
pub fn default_templates() -> Vec<LayerTemplate> {
    vec![
        LayerTemplate {
            id: "imagery".into(),
            kind: LayerKind::Imagery,
            encoding: Encoding::Jpeg,
            tile_pixels: 256,
            vertical_datum: None,
            accuracy: Accuracy::unknown(),
        },
        LayerTemplate {
            id: "elevation".into(),
            kind: LayerKind::Elevation,
            encoding: Encoding::TerrariumPng,
            tile_pixels: 256,
            vertical_datum: Some(VerticalDatum::Orthometric {
                geoid_model: "egm96".into(),
            }),
            accuracy: Accuracy::unknown(),
        },
    ]
}

/// Plans the flight, fetches the planned tiles, and seals a package.
/// Returns the resource plan and the package summary as JSON.
///
/// # Errors
///
/// Returns [`PipelineError`] for an unusable plan file, a failed fetch, and
/// a refused or failed package write.
pub fn build_package_blocking(options: &PackageOptions) -> Result<Value, PipelineError> {
    let plan_file: PlanFile = read_json(&options.plan)?;
    let fetch = TemplateFetch::new(&options.cache)
        .with_layer(LayerKind::Imagery, options.imagery.clone())
        .with_layer(LayerKind::Elevation, options.elevation.clone());
    let points: Vec<[f64; 2]> = plan_file
        .waypoints
        .iter()
        .map(|w| [w.lat_deg, w.lon_deg])
        .collect();
    let ground = TerrariumGround::fetch_blocking(&fetch, &points)?;
    let request = plan_file.request();
    let plan = plan_resources(&request, &ground)?;
    let tiles = plan_tiles(&request.legs, &plan)?;
    let count: usize = tiles.values().map(std::collections::BTreeSet::len).sum();
    let mut out = json!({ "resource_plan": plan, "planned_tiles": count });
    if options.plan_only {
        return Ok(out);
    }
    let keys = keys(&fetch, &tiles)?;
    if keys.len() > options.max_tiles {
        return Err(PipelineError::Invalid(format!(
            "{} tiles planned, limit {}",
            keys.len(),
            options.max_tiles
        )));
    }
    tracing::info!(tiles = keys.len(), "fetching planned tiles");
    fetch.prefetch_blocking(&keys, FETCH_THREADS)?;
    let mut store = DirectoryStore::new(&options.store);
    let summary = pollster::block_on(prepare(
        &plan_file.release_id,
        &tiles,
        &default_templates(),
        options.max_tiles,
        &fetch,
        &mut store,
    ))?;
    out["package"] = serde_json::to_value(&summary).map_err(|source| PipelineError::Json {
        context: "package summary".into(),
        source,
    })?;
    Ok(out)
}

fn keys(
    fetch: &TemplateFetch,
    tiles: &navigate_data::PlannedTiles,
) -> Result<Vec<TileKey>, PipelineError> {
    let mut keys = Vec::new();
    for (&kind, addresses) in tiles {
        let source: SourceRef = fetch
            .source(kind)
            .ok_or_else(|| PipelineError::Invalid(format!("no {kind:?} source")))?;
        keys.extend(addresses.iter().map(|&address| TileKey {
            source: source.id.clone(),
            revision: source.revision.clone(),
            layer: kind,
            address,
        }));
    }
    Ok(keys)
}

#[cfg(test)]
mod tests;
