//! The `package` subcommand: a navigation data package for a flight plan.
//! The tool is the host: it fetches through URL templates or reads a
//! catalog, and `navigate-data` plans, checks, and seals.

mod fetch;
mod ground;

pub use fetch::{CatalogFetch, TemplateFetch, TemplateLayer};
pub use ground::{fetch_ground_blocking, insert_png};
pub use navigate_data::package_plan::{GROUND_ZOOM, PlanCamera, TerrariumGround, layer_spec};

use crate::{PipelineError, json_file::read_json};
use navigate_data::package_plan::{PackagePlan, PlanBounds, route_request};
use navigate_data::{
    Descriptor, DirectoryStore, LayerKind, LayerLabel, LayerLimits, ResourceRequest, Revision,
    SourceKey, prepare,
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
    /// The waypoints as `[lat, lon, altitude]`.
    pub fn points(&self) -> Vec<[f64; 3]> {
        self.waypoints
            .iter()
            .map(|w| [w.lat_deg, w.lon_deg, w.altitude_m])
            .collect()
    }

    /// The bounds of the plan.
    pub const fn bounds(&self) -> PlanBounds {
        PlanBounds {
            search_radius_m: self.search_radius_m,
            deviation_m: self.deviation_m,
            max_range_m: self.max_range_m,
        }
    }

    /// The resource request with the measured layer sizes.
    pub fn request(&self) -> ResourceRequest {
        route_request(
            &self.points(),
            &self.cameras,
            self.bounds(),
            LayerLimits::imagery(),
            LayerLimits::terrarium(),
        )
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

/// A source from its text form: a registry name, for example
/// `esri_world_imagery`, or `declared:<sha256>` of a source declaration.
///
/// # Errors
///
/// Refuses a name that the registry does not hold.
pub fn parse_source(text: &str) -> Result<SourceKey, PipelineError> {
    match serde_json::from_value(Value::String(text.to_owned())) {
        Ok(SourceKey::Unsupported) | Err(_) => {
            Err(PipelineError::Invalid(format!("unknown source {text}")))
        }
        Ok(source) => Ok(source),
    }
}

/// A revision from `unversioned`, `release:N`, or `collected:YYYY-MM-DD`.
///
/// # Errors
///
/// Refuses any other text.
pub fn parse_revision(text: &str) -> Result<Revision, PipelineError> {
    let bad = || {
        PipelineError::Invalid(format!(
            "revision {text}: use unversioned, release:N, or collected:YYYY-MM-DD"
        ))
    };
    if text == "unversioned" {
        return Ok(Revision::Unversioned);
    }
    if let Some(n) = text.strip_prefix("release:") {
        return n.parse().map(Revision::Release).map_err(|_| bad());
    }
    let date = text.strip_prefix("collected:").ok_or_else(bad)?;
    let parts: Vec<&str> = date.split('-').collect();
    let [y, m, d] = parts.as_slice() else {
        return Err(bad());
    };
    let (y, m, d) = (
        y.parse().map_err(|_| bad())?,
        m.parse().map_err(|_| bad())?,
        d.parse().map_err(|_| bad())?,
    );
    navigate_data::DayNumber::from_ymd(y, m, d)
        .map(Revision::Collected)
        .map_err(PipelineError::from)
}

/// The descriptor of a package from the plan release and the layers.
pub fn descriptor(release_id: &str, layers: &[&TemplateLayer]) -> Descriptor {
    Descriptor {
        release_id: release_id.to_owned(),
        layers: layers
            .iter()
            .map(|l| LayerLabel {
                key: l.spec.key,
                label: format!("{:?}", l.spec.key.source),
                attribution: l.attribution.clone(),
                licence: String::new(),
            })
            .collect(),
        note: String::new(),
    }
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
    let ground = fetch_ground_blocking(&fetch, &options.elevation.spec, &points)?;
    let plan = PackagePlan::new(plan_file.request(), &ground)?;
    let count = plan.tile_count();
    let mut out =
        json!({ "resource_plan": plan.resources, "planned_tiles": count, "size": plan.size });
    if options.plan_only {
        return Ok(out);
    }
    let keys = plan.keys(&fetch.specs())?;
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
    let descriptor = descriptor(
        &plan_file.release_id,
        &[&options.imagery, &options.elevation],
    );
    let summary = pollster::block_on(prepare(
        &plan.tiles,
        &fetch.specs(),
        &descriptor,
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

#[cfg(test)]
mod tests;
