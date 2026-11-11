//! The plan of a package for a route: the host values around the
//! [`resource`](crate::resource) model that a package identity depends on.
//!
//! The identity of a package is the digest of its manifest, and the
//! manifest records each layer as declared and each planned tile. Two hosts
//! that plan the same route, sources and camera seal the same package only
//! if they declare the layers alike, sample the planning ground alike,
//! derive the camera alike, and put the elevation band at the same zoom
//! offset. This module holds those choices, so a host calls them instead of
//! copying them; it is `no_std` and builds for `wasm32-unknown-unknown`.
//!
//! A host plans in this order:
//!
//! 1. [`ground_tiles`] about the route, decoded into a [`TerrariumGround`]
//!    (the host decodes the PNG; [`TerrariumGround::insert_rgb`] takes the
//!    pixels);
//! 2. [`route_request`] from the route, the [`PlanCamera`]s, the
//!    [`PlanBounds`] and the source limits;
//! 3. [`PackagePlan::new`] over that ground, which gives the tiles, their
//!    keys at the declared [`layer_spec`]s and the size estimate;
//! 4. a fetch of [`PackagePlan::keys`], then [`prepare`](crate::prepare).

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use navigate_geodesy::float_math::tan;
use serde::{Deserialize, Serialize};

use crate::{
    Accuracy, DEFAULT_CORRELATION_ZOOM, DataError, Encoding, GeoidModel, GroundHeight,
    HorizontalDatum, LayerKey, LayerKind, LayerLimits, LayerSpec, PlannedCamera, PlannedLeg,
    PlannedTiles, ResourcePlan, ResourceRequest, Revision, SizeEstimate, SourceKey, TileKey,
    VerticalDatum, estimate_size, plan_resources, plan_tiles,
};

mod ground;
pub use ground::{GROUND_CORRIDOR_M, GROUND_ZOOM, TerrariumGround, ground_tiles};

#[cfg(test)]
mod tests;

/// Zoom levels between the imagery and the elevation of one band: an
/// elevation texel eight imagery texels wide.
pub const ELEVATION_OFFSET: u32 = 3;

/// How far about the track a package reaches.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanBounds {
    /// Largest search radius about the navigation estimate, metres.
    pub search_radius_m: f64,
    /// Allowed deviation from the planned track, metres.
    pub deviation_m: f64,
    /// Farthest slant range that the navigator renders or matches, metres.
    pub max_range_m: f64,
}

impl PlanBounds {
    /// The bounds of the onboard terrain-camera package: 500 m of search
    /// radius and of track deviation, and 4 km of slant range.
    ///
    /// These are engineering choices, not values from a source or a
    /// measurement. The search radius and the deviation bound how far the
    /// onboard belief may drift from the planned track and still find its
    /// map; the range bounds the slant distance at which the camera can
    /// use the ground. A route whose belief can drift further, or whose
    /// camera sees further, needs other bounds.
    pub const ONBOARD: Self = Self {
        search_radius_m: 500.0,
        deviation_m: 500.0,
        max_range_m: 4_000.0,
    };
}

/// A camera as a flight plan states it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanCamera {
    /// Image width in pixels.
    pub width_px: u32,
    /// Image height in pixels.
    pub height_px: u32,
    /// Horizontal field of view in degrees.
    pub horizontal_fov_deg: f64,
    /// Angle of the optical axis below the horizontal in degrees: 90 for a
    /// nadir camera.
    pub depression_deg: f64,
}

impl PlanCamera {
    /// The camera as the resource model sees it: the focal length in
    /// pixels from the width and the horizontal field of view.
    #[must_use]
    pub fn planned(&self) -> PlannedCamera {
        PlannedCamera {
            focal_px: f64::from(self.width_px)
                / 2.0
                / tan(self.horizontal_fov_deg.to_radians() / 2.0),
            width_px: self.width_px,
            height_px: self.height_px,
            depression_rad: self.depression_deg.to_radians(),
        }
    }
}

/// How a package declares a layer of `kind` from `source` at `revision`:
/// WGS84, 256-pixel tiles, JPEG imagery or Terrarium elevation above
/// EGM96, unknown accuracy and the default correlation zoom.
#[must_use]
pub fn layer_spec(kind: LayerKind, source: SourceKey, revision: Revision) -> LayerSpec {
    let elevation = kind == LayerKind::Elevation;
    LayerSpec {
        key: LayerKey { kind, source },
        revision,
        horizontal_datum: HorizontalDatum::Wgs84,
        vertical_datum: elevation.then_some(VerticalDatum::Orthometric {
            geoid: GeoidModel::Egm96,
        }),
        encoding: if elevation {
            Encoding::TerrariumPng
        } else {
            Encoding::Jpeg
        },
        tile_pixels: 256,
        accuracy: Accuracy::unknown(),
        correlation_zoom: DEFAULT_CORRELATION_ZOOM,
    }
}

/// The legs between consecutive `points`, each `[lat, lon, height]` in
/// degrees and metres in the reference of the planning ground.
#[must_use]
pub fn route_legs(points: &[[f64; 3]]) -> Vec<PlannedLeg> {
    points
        .windows(2)
        .filter_map(|pair| match pair {
            [from, to] => Some(PlannedLeg {
                from: *from,
                to: *to,
            }),
            _ => None,
        })
        .collect()
}

/// The resource request of a route seen by `cameras` from the sources of
/// `imagery` and `elevation`.
#[must_use]
pub fn route_request(
    points: &[[f64; 3]],
    cameras: &[PlanCamera],
    bounds: PlanBounds,
    imagery: LayerLimits,
    elevation: LayerLimits,
) -> ResourceRequest {
    ResourceRequest {
        legs: route_legs(points),
        cameras: cameras.iter().map(PlanCamera::planned).collect(),
        search_radius_m: bounds.search_radius_m,
        deviation_m: bounds.deviation_m,
        max_range_m: bounds.max_range_m,
        imagery,
        elevation,
        elevation_offset: ELEVATION_OFFSET,
    }
}

/// What a route and its cameras need of the sources.
#[derive(Clone, Debug, PartialEq)]
pub struct PackagePlan {
    /// The inputs of the resource model.
    pub request: ResourceRequest,
    /// What each leg needs: its height above ground, zooms and strips.
    pub resources: ResourcePlan,
    /// The tiles selected, by layer.
    pub tiles: PlannedTiles,
    /// Their count by layer and zoom, with the estimate and the upper
    /// bound in bytes.
    pub size: SizeEstimate,
}

impl PackagePlan {
    /// Runs the resource model for `request` over `ground` and selects the
    /// tiles.
    ///
    /// # Errors
    ///
    /// Refuses fewer than two points, a leg over unknown ground or below
    /// it, and a corridor past the Web Mercator latitudes.
    pub fn new<G: GroundHeight + ?Sized>(
        request: ResourceRequest,
        ground: &G,
    ) -> Result<Self, DataError> {
        let resources = plan_resources(&request, ground)?;
        let tiles = plan_tiles(&request.legs, &resources)?;
        let size = estimate_size(&request, &tiles);
        Ok(Self {
            request,
            resources,
            tiles,
            size,
        })
    }

    /// Tiles selected.
    #[must_use]
    pub fn tile_count(&self) -> usize {
        self.tiles.values().map(BTreeSet::len).sum()
    }

    /// The key of every selected tile at the layers `specs` declare, in
    /// layer then address order.
    ///
    /// # Errors
    ///
    /// Refuses a planned layer that `specs` does not declare.
    pub fn keys(&self, specs: &[LayerSpec]) -> Result<Vec<TileKey>, DataError> {
        let mut keys = Vec::with_capacity(self.tile_count());
        for (&layer, addresses) in &self.tiles {
            let spec = specs
                .iter()
                .find(|spec| spec.key.kind == layer)
                .ok_or_else(|| DataError::Invalid {
                    field: "layers",
                    detail: alloc::format!("no {layer:?} layer is declared"),
                })?;
            keys.extend(addresses.iter().map(|&address| TileKey {
                source: spec.key.source,
                revision: spec.revision,
                layer,
                address,
            }));
        }
        Ok(keys)
    }
}
