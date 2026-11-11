//! The resource model: what each leg of a flight needs, before a fetch.
//!
//! The zoom rule is the rule of the map renderer: one zoom level for each
//! doubling of the slant range, with one texel for one pixel. A texel of a
//! tile of `T` pixels at zoom `z` and latitude `φ` is
//! `C × cos φ / (T × 2^z)` metres, with `C` the equatorial circumference.
//! A pixel at slant range `d` of a camera with focal length `f` pixels is
//! `d / f` metres. The two are equal at
//! `z = log2(f × C × cos φ / (T × d))`.
//!
//! A higher leg has a larger slant range, so it needs a coarser level. The
//! strip that a camera sees is wider, but a level coarser has a quarter of
//! the tiles for each square kilometre, so a higher leg costs less for
//! each kilometre. The far field of an oblique camera uses coarser levels
//! than its near field.

use alloc::vec::Vec;
use core::f64::consts::{FRAC_PI_2, LN_2};

use navigate_geodesy::float_math::{atan, ceil, cos, floor, ln, powi, sqrt, tan};
use serde::{Deserialize, Serialize};

use crate::{DataError, LayerKind};

mod geometry;
pub use geometry::{great_circle_m, point_along};

/// Equatorial circumference of the WGS84 ellipsoid in metres.
pub const EQUATOR_M: f64 = 40_075_016.685_578_49;
/// The lower edge of altitude band 0, in metres above ground. Band `k`
/// covers `[ALTITUDE_BAND_BASE_M × 2^k, ALTITUDE_BAND_BASE_M × 2^(k+1))`.
pub const ALTITUDE_BAND_BASE_M: f64 = 100.0;
/// Spacing of the ground samples along a leg, in metres.
const GROUND_SAMPLE_M: f64 = 500.0;

/// The zoom at which a texel of a `tile_pixels` tile is one pixel at slant
/// range `slant_m` for a camera of focal length `focal_px` pixels, at
/// latitude `lat_deg`. The value is not rounded.
#[must_use]
pub fn texel_zoom(focal_px: f64, slant_m: f64, lat_deg: f64, tile_pixels: u32) -> f64 {
    let ratio = focal_px * EQUATOR_M * cos(lat_deg.to_radians())
        / (f64::from(tile_pixels) * slant_m.max(1e-3));
    ln(ratio) / LN_2
}

/// The slant range in metres at which [`texel_zoom`] is `zoom`.
#[must_use]
pub fn slant_at_zoom(focal_px: f64, zoom: u32, lat_deg: f64, tile_pixels: u32) -> f64 {
    focal_px * EQUATOR_M * cos(lat_deg.to_radians())
        / (f64::from(tile_pixels) * powi(2.0, zoom as i32))
}

/// The altitude band of a height above ground.
#[must_use]
pub fn altitude_band(agl_m: f64) -> u32 {
    let ratio = (agl_m / ALTITUDE_BAND_BASE_M).max(1.0);
    floor(ln(ratio) / LN_2) as u32
}

/// One camera as the resource model sees it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlannedCamera {
    /// Focal length in pixels.
    pub focal_px: f64,
    /// Image width in pixels.
    pub width_px: u32,
    /// Image height in pixels.
    pub height_px: u32,
    /// Angle of the optical axis below the horizontal, in radians: `π/2`
    /// for a nadir camera.
    pub depression_rad: f64,
}

/// One leg of a flight plan: `[lat, lon, altitude]`, degrees and metres.
/// The altitude and the ground heights use the same vertical reference.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlannedLeg {
    /// Start of the leg.
    pub from: [f64; 3],
    /// End of the leg.
    pub to: [f64; 3],
}

/// The limits and the measured size of one source layer.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct LayerLimits {
    /// Pixels on one side of a tile.
    pub tile_pixels: u32,
    /// Coarsest zoom to use.
    pub min_zoom: u32,
    /// Finest zoom that the source has.
    pub max_zoom: u32,
    /// Measured encoded size in bytes for each square kilometre at
    /// `reference_zoom`.
    pub bytes_per_km2: f64,
    /// The zoom of `bytes_per_km2`.
    pub reference_zoom: u32,
    /// Size ratio between one zoom and the next coarser one.
    pub growth_per_zoom: f64,
}

impl LayerLimits {
    /// Measured imagery: approximately 18 KB/km² at zoom 15 and 70 KB/km²
    /// at zoom 16.
    #[must_use]
    pub const fn imagery() -> Self {
        Self {
            tile_pixels: 256,
            min_zoom: 8,
            max_zoom: 19,
            bytes_per_km2: 18_000.0,
            reference_zoom: 15,
            growth_per_zoom: 3.9,
        }
    }

    /// Measured Terrarium elevation: approximately 2 KB/km² at zoom 12 and
    /// 7.5 KB/km² at zoom 13.
    #[must_use]
    pub const fn terrarium() -> Self {
        Self {
            tile_pixels: 256,
            min_zoom: 6,
            max_zoom: 15,
            bytes_per_km2: 2_000.0,
            reference_zoom: 12,
            growth_per_zoom: 3.75,
        }
    }

    /// Estimated bytes for each square kilometre at `zoom`.
    #[must_use]
    pub fn bytes_per_km2_at(&self, zoom: u32) -> f64 {
        let levels = i64::from(zoom) - i64::from(self.reference_zoom);
        self.bytes_per_km2 * powi(self.growth_per_zoom, levels as i32)
    }
}

/// The inputs of the resource model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResourceRequest {
    /// The legs, in fly order.
    pub legs: Vec<PlannedLeg>,
    /// The cameras of the vehicle.
    pub cameras: Vec<PlannedCamera>,
    /// Largest search radius around the navigation estimate, in metres.
    pub search_radius_m: f64,
    /// Allowed deviation from the planned track, in metres.
    pub deviation_m: f64,
    /// Farthest slant range that the navigator renders or matches, in
    /// metres.
    pub max_range_m: f64,
    /// The imagery source.
    pub imagery: LayerLimits,
    /// The elevation source.
    pub elevation: LayerLimits,
    /// Zoom levels between the imagery and the elevation of one band.
    pub elevation_offset: u32,
}

/// The tiles of one layer at one zoom along a leg.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ZoomBand {
    /// The layer.
    pub layer: LayerKind,
    /// The zoom.
    pub zoom: u32,
    /// Ground distance ahead of the vehicle where the band starts, metres.
    pub near_m: f64,
    /// Ground distance ahead of the vehicle where the band ends, metres.
    pub far_m: f64,
    /// Half width of the strip about the track, metres.
    pub half_width_m: f64,
    /// Area of the strip in square kilometres.
    pub area_km2: f64,
    /// Estimated encoded size in bytes.
    pub bytes: f64,
}

/// What one leg needs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LegResources {
    /// Index of the leg.
    pub leg: usize,
    /// Length of the leg in metres.
    pub length_m: f64,
    /// Lowest height above ground on the leg, metres.
    pub agl_m: f64,
    /// Altitude band of `agl_m`.
    pub altitude_band: u32,
    /// The bands, imagery first, each layer from fine to coarse.
    pub bands: Vec<ZoomBand>,
    /// Widest strip half width of the leg, metres.
    pub corridor_half_width_m: f64,
    /// Estimated encoded size in bytes.
    pub bytes: f64,
}

/// What the flight needs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResourcePlan {
    /// One entry for each leg.
    pub legs: Vec<LegResources>,
    /// Estimated encoded size in bytes. Strips of one zoom that overlap
    /// between legs are counted once for each leg.
    pub bytes: f64,
}

/// The heights of the ground that the host knows before the fetch, for
/// example from a coarse elevation layer.
pub trait GroundHeight {
    /// The ground height at `[lat, lon]` in degrees, or `None` when it is
    /// unknown.
    fn ground_m(&self, point: [f64; 2]) -> Option<f64>;
}

/// Computes what each leg needs.
///
/// # Errors
///
/// Refuses a request with no leg or no camera, values that are not finite,
/// a ground height that is unknown on a leg, and a leg below the ground.
pub fn plan_resources<G: GroundHeight + ?Sized>(
    request: &ResourceRequest,
    ground: &G,
) -> Result<ResourcePlan, DataError> {
    check(request)?;
    let mut legs = Vec::with_capacity(request.legs.len());
    for (index, leg) in request.legs.iter().enumerate() {
        legs.push(leg_resources(request, ground, index, leg)?);
    }
    let bytes = legs.iter().map(|l| l.bytes).sum();
    Ok(ResourcePlan { legs, bytes })
}

fn invalid(field: &'static str, detail: alloc::string::String) -> DataError {
    DataError::Invalid { field, detail }
}

fn check(request: &ResourceRequest) -> Result<(), DataError> {
    if request.legs.is_empty() || request.cameras.is_empty() {
        return Err(invalid(
            "request",
            "a request needs legs and cameras".into(),
        ));
    }
    let finite = request
        .legs
        .iter()
        .flat_map(|l| l.from.iter().chain(l.to.iter()))
        .all(|v| v.is_finite())
        && request.cameras.iter().all(|c| {
            c.focal_px.is_finite()
                && c.focal_px > 0.0
                && c.depression_rad.is_finite()
                && c.width_px > 0
                && c.height_px > 0
        })
        && [
            request.search_radius_m,
            request.deviation_m,
            request.max_range_m,
        ]
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0)
        && request.max_range_m > 0.0;
    if !finite {
        return Err(invalid(
            "request",
            "values must be finite and positive".into(),
        ));
    }
    Ok(())
}

/// The lowest height above ground along a leg.
fn lowest_agl<G: GroundHeight + ?Sized>(
    ground: &G,
    leg: &PlannedLeg,
    length_m: f64,
) -> Result<f64, DataError> {
    let samples = ceil(length_m / GROUND_SAMPLE_M).max(1.0) as u32;
    let mut lowest = f64::INFINITY;
    for i in 0..=samples {
        let t = f64::from(i) / f64::from(samples);
        let point = point_along([leg.from[0], leg.from[1]], [leg.to[0], leg.to[1]], t);
        let altitude = leg.from[2] + (leg.to[2] - leg.from[2]) * t;
        let ground_m = ground
            .ground_m(point)
            .ok_or_else(|| invalid("ground", alloc::format!("unknown at {point:?}")))?;
        lowest = lowest.min(altitude - ground_m);
    }
    if lowest <= 0.0 {
        return Err(invalid("leg", alloc::format!("{lowest} m above ground")));
    }
    Ok(lowest)
}

fn leg_resources<G: GroundHeight + ?Sized>(
    request: &ResourceRequest,
    ground: &G,
    index: usize,
    leg: &PlannedLeg,
) -> Result<LegResources, DataError> {
    let length_m = great_circle_m([leg.from[0], leg.from[1]], [leg.to[0], leg.to[1]]);
    let agl_m = lowest_agl(ground, leg, length_m)?;
    let lat = leg.from[0].abs().min(leg.to[0].abs()).copysign(leg.from[0]);
    let margin = request.search_radius_m + request.deviation_m;
    let mut imagery: Vec<ZoomBand> = Vec::new();
    for camera in &request.cameras {
        for band in camera_bands(request, camera, agl_m, lat) {
            merge(&mut imagery, band);
        }
    }
    let mut elevation: Vec<ZoomBand> = Vec::new();
    for band in &imagery {
        let zoom = band
            .zoom
            .saturating_sub(request.elevation_offset)
            .clamp(request.elevation.min_zoom, request.elevation.max_zoom);
        merge(
            &mut elevation,
            ZoomBand {
                layer: LayerKind::Elevation,
                zoom,
                ..*band
            },
        );
    }
    let mut bands = Vec::new();
    for mut band in imagery.into_iter().chain(elevation) {
        band.half_width_m += margin;
        let strip_m = length_m + band.far_m;
        band.area_km2 = strip_m * 2.0 * band.half_width_m / 1e6;
        let limits = match band.layer {
            LayerKind::Elevation => &request.elevation,
            _ => &request.imagery,
        };
        band.bytes = band.area_km2 * limits.bytes_per_km2_at(band.zoom);
        bands.push(band);
    }
    let corridor_half_width_m = bands.iter().map(|b| b.half_width_m).fold(0.0, f64::max);
    let bytes = bands.iter().map(|b| b.bytes).sum();
    Ok(LegResources {
        leg: index,
        length_m,
        agl_m,
        altitude_band: altitude_band(agl_m),
        bands,
        corridor_half_width_m,
        bytes,
    })
}

/// Joins `band` into the band of the same layer and zoom: the union of the
/// ground ranges and the larger strip.
fn merge(bands: &mut Vec<ZoomBand>, band: ZoomBand) {
    if let Some(found) = bands
        .iter_mut()
        .find(|b| b.layer == band.layer && b.zoom == band.zoom)
    {
        found.near_m = found.near_m.min(band.near_m);
        found.far_m = found.far_m.max(band.far_m);
        found.half_width_m = found.half_width_m.max(band.half_width_m);
    } else {
        bands.push(band);
        bands.sort_by_key(|b| core::cmp::Reverse(b.zoom));
    }
}

/// The imagery bands that one camera sees at a height above ground.
fn camera_bands(
    request: &ResourceRequest,
    camera: &PlannedCamera,
    agl_m: f64,
    lat: f64,
) -> Vec<ZoomBand> {
    let limits = &request.imagery;
    let half_v = atan(f64::from(camera.height_px) / (2.0 * camera.focal_px));
    let half_h = atan(f64::from(camera.width_px) / (2.0 * camera.focal_px));
    let steep = (camera.depression_rad + half_v).min(FRAC_PI_2);
    let shallow = camera.depression_rad - half_v;
    let near_slant = agl_m / navigate_geodesy::float_math::sin(steep);
    let far_slant = if shallow > 0.0 {
        (agl_m / navigate_geodesy::float_math::sin(shallow)).min(request.max_range_m)
    } else {
        request.max_range_m
    }
    .max(near_slant);
    let ground_of = |slant: f64| sqrt((slant * slant - agl_m * agl_m).max(0.0));
    let fine = ceil(texel_zoom(
        camera.focal_px,
        near_slant,
        lat,
        limits.tile_pixels,
    ));
    let coarse = ceil(texel_zoom(
        camera.focal_px,
        far_slant,
        lat,
        limits.tile_pixels,
    ));
    let clamp = |z: f64| (z.max(0.0) as u32).clamp(limits.min_zoom, limits.max_zoom);
    let mut bands = Vec::new();
    let (fine, coarse) = (clamp(fine), clamp(coarse));
    for zoom in (coarse..=fine).rev() {
        let lo = if zoom == fine {
            near_slant
        } else {
            slant_at_zoom(camera.focal_px, zoom + 1, lat, limits.tile_pixels).max(near_slant)
        };
        let hi = if zoom == coarse {
            far_slant
        } else {
            slant_at_zoom(camera.focal_px, zoom, lat, limits.tile_pixels).min(far_slant)
        };
        if hi < lo {
            continue;
        }
        bands.push(ZoomBand {
            layer: LayerKind::Imagery,
            zoom,
            near_m: ground_of(lo),
            far_m: ground_of(hi),
            half_width_m: hi * tan(half_h),
            area_km2: 0.0,
            bytes: 0.0,
        });
    }
    bands
}

#[cfg(test)]
mod tests;
