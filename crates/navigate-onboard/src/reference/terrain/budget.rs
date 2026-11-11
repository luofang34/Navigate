//! The error of a terrain fix that image geometry cannot measure, composed
//! as a root sum square of independent terms.
//!
//! The terms are independent of each other: the registration of the
//! imagery, the height and the tilt of the elevation model, and the
//! calibration of the camera come from different processes. Their
//! covariances add. The fusion bridge adds the image-geometry covariance
//! of the match to the sum.
//!
//! The tilt term holds what a free 6-DoF solve does with an elevation
//! error that is not uniform under the footprint: it takes a tilt of the
//! elevation error as a turned camera, and the fix moves horizontally by
//! the height above the ground times the tilt. A uniform elevation error
//! moves the fix vertically only, and a mount angle error turns the solved
//! attitude and hardly moves the position. An elevation error of one-sigma
//! `σ` that is correlated over a length `L` has a slope of about `√2·σ/L`
//! on each axis. The solve fits the tilt over the whole footprint: over a
//! footprint of width `W`, the `(W/L)²` patches of the error average, and
//! the fitted slope is about `√12·σ·L/W²`. The term takes the smaller of
//! the two. A fix is taken as independent of the fixes of other map
//! cells. That holds while the footprints of the fixes of adjacent cells
//! do not overlap: cells of zoom 16 are about 470 m wide, and a footprint
//! of a camera with a field of 0.8 of the range is that wide from about
//! 600 m above the ground. From higher, adjacent fixes share part of
//! their tilt error, and the term does not hold that correlation.

use nalgebra::{Matrix3, Vector3};
use navigate_camera::Intrinsics;
use navigate_visual::{Estimate, ReferenceView};

/// Largest off-nadir angle of the height term. Past it the tangent grows
/// without bound, and a view that near the horizon does not match.
const MAX_OFF_NADIR_RAD: f64 = 1.5;

/// One-sigma errors of the map at the present day.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MapSigmas {
    /// Horizontal registration of the imagery, each axis, metres.
    pub horizontal_m: f64,
    /// Height of the terrain surface above the ellipsoid, metres: the
    /// elevation model and the bound of an unknown geoid undulation.
    pub height_m: f64,
    /// The elevation model alone, metres: the error whose tilt under a
    /// footprint moves a fix. An unknown geoid undulation is the same
    /// over a footprint and has no tilt.
    pub elevation_m: f64,
    /// Length over which the error of the elevation model is correlated,
    /// metres ([`super::TerrainConfig::dem_correlation_length_m`]).
    pub elevation_length_m: f64,
}

/// How the camera saw the ground in one fix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ViewGeometry {
    /// Angle of the optical axis from the local down, radians.
    pub off_nadir_rad: f64,
    /// Unit direction of the optical axis over the ground, `[north, east]`;
    /// zero for a view straight down.
    pub ground_direction: [f64; 2],
    /// Mean distance to the matched ground along the optical axis, metres.
    pub range_m: f64,
    /// The narrower extent of the view for each metre of range, the
    /// smaller of the width over `fx` and the height over `fy`: times the
    /// range, the narrower side of the footprint.
    pub field_ratio: f64,
}

impl ViewGeometry {
    /// The geometry of a solved estimate over its reference. The frame of
    /// an estimate is a tangent frame near the fix, so its axes are east,
    /// north, and up at the fix.
    pub(crate) fn of(estimate: &Estimate, reference: &ReferenceView, camera: &Intrinsics) -> Self {
        // The eye axes are right, up, and back: the camera looks along -Z.
        let axis = estimate.pose.orientation * -Vector3::z();
        let horizontal = axis.x.hypot(axis.y);
        let ground_direction = if horizontal > 1e-9 {
            [axis.y / horizontal, axis.x / horizontal]
        } else {
            [0.0, 0.0]
        };
        let (sum, count) = reference
            .depth_m
            .iter()
            .filter(|d| **d > 0.0)
            .fold((0.0_f64, 0_u32), |(s, n), d| {
                (s + f64::from(*d), n.saturating_add(1))
            });
        Self {
            off_nadir_rad: (-axis.z).clamp(-1.0, 1.0).acos(),
            ground_direction,
            range_m: if count == 0 {
                0.0
            } else {
                sum / f64::from(count)
            },
            field_ratio: (f64::from(camera.width) / camera.fx)
                .min(f64::from(camera.height) / camera.fy),
        }
    }
}

#[cfg(test)]
impl MapSigmas {
    /// The covariance of the map error in north, east, down axes, m²:
    /// the registration on each horizontal axis and the terrain height
    /// on the vertical. Every fix of one map cell shares this error.
    pub(crate) fn covariance_ned(&self) -> Matrix3<f64> {
        Matrix3::from_diagonal(&Vector3::new(
            self.horizontal_m.powi(2),
            self.horizontal_m.powi(2),
            self.height_m.powi(2),
        ))
    }
}

/// The external covariance of a fix in north, east, down axes, m²: the
/// map covariance ([`MapSigmas::covariance_ned`]) and the rest
/// ([`unshared_covariance_ned`]).
///
/// - Imagery registration: `horizontal_m²` on each horizontal axis.
/// - Terrain height: the camera height follows the terrain height, so
///   `height_m²` on the vertical axis. A height error also moves a ground
///   point seen at the off-nadir angle `θ` by `height_m · tan θ` along the
///   ground direction of the view, so `(height_m · tan θ)²` on that
///   direction. A view straight down has no such term.
/// - Calibration: an angular error `calibration_sigma_rad` moves the fix by
///   that angle times the range, on each axis.
/// - Terrain tilt ([`tilt_sigma_m`]): on each horizontal axis.
#[cfg(test)]
pub(crate) fn external_covariance_ned(
    map: &MapSigmas,
    calibration_sigma_rad: f64,
    view: &ViewGeometry,
) -> Matrix3<f64> {
    map.covariance_ned() + unshared_covariance_ned(map, calibration_sigma_rad, view)
}

/// The external covariance of a fix without the map error, in north,
/// east, down axes, m²: the relief, the calibration, and the tilt terms of
/// [`external_covariance_ned`].
pub(crate) fn unshared_covariance_ned(
    map: &MapSigmas,
    calibration_sigma_rad: f64,
    view: &ViewGeometry,
) -> Matrix3<f64> {
    let calibration_m2 = (calibration_sigma_rad * view.range_m).powi(2);
    let relief_m = map.height_m * view.off_nadir_rad.clamp(0.0, MAX_OFF_NADIR_RAD).tan();
    let [north, east] = view.ground_direction;
    let along = Vector3::new(north, east, 0.0) * relief_m;
    let tilt_m2 = tilt_sigma_m(map, view).powi(2);
    Matrix3::from_diagonal(&Vector3::new(
        calibration_m2 + tilt_m2,
        calibration_m2 + tilt_m2,
        calibration_m2,
    )) + along * along.transpose()
}

/// The horizontal one-sigma of a fix from the tilt of the elevation error
/// under its footprint, each axis, metres: the height above the ground
/// times the fitted slope of the error, the smaller of `√2·σ/L` and
/// `√12·σ·L/W²`, `W` being the narrower side of the footprint. The term
/// grows with the height above the ground while the footprint is narrower
/// than about `1.6·L`, and falls beyond, where the footprint averages more
/// of the error than the height multiplies.
pub(crate) fn tilt_sigma_m(map: &MapSigmas, view: &ViewGeometry) -> f64 {
    let off_nadir = view.off_nadir_rad.clamp(0.0, MAX_OFF_NADIR_RAD);
    let height_m = view.range_m * off_nadir.cos();
    let footprint_m = view.range_m * view.field_ratio;
    let (sigma, length) = (map.elevation_m, map.elevation_length_m);
    if !(length > 0.0 && length.is_finite()) {
        return 0.0;
    }
    let local = core::f64::consts::SQRT_2 * sigma / length;
    let fitted = if footprint_m > 0.0 {
        12.0_f64.sqrt() * sigma * length / (footprint_m * footprint_m)
    } else {
        local
    };
    height_m * local.min(fitted)
}

#[cfg(test)]
mod tests;
