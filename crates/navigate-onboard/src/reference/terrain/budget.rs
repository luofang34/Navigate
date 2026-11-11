//! The error of a terrain fix that image geometry cannot measure, composed
//! as a root sum square of independent terms.
//!
//! The terms are independent of each other: the registration of the
//! imagery, the height of the elevation model, and the calibration of the
//! camera come from three different processes. Their covariances add. The
//! fusion bridge adds the image-geometry covariance of the match to the
//! sum.

use nalgebra::{Matrix3, Vector3};
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
}

impl ViewGeometry {
    /// The geometry of a solved estimate over its reference. The frame of
    /// an estimate is a tangent frame near the fix, so its axes are east,
    /// north, and up at the fix.
    pub(crate) fn of(estimate: &Estimate, reference: &ReferenceView) -> Self {
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
        }
    }
}

/// The external covariance of a fix in north, east, down axes, m².
///
/// - Imagery registration: `horizontal_m²` on each horizontal axis.
/// - Terrain height: the camera height follows the terrain height, so
///   `height_m²` on the vertical axis. A height error also moves a ground
///   point seen at the off-nadir angle `θ` by `height_m · tan θ` along the
///   ground direction of the view, so `(height_m · tan θ)²` on that
///   direction. A view straight down has no such term.
/// - Calibration: an angular error `calibration_sigma_rad` moves the fix by
///   that angle times the range, on each axis.
pub(crate) fn external_covariance_ned(
    map: &MapSigmas,
    calibration_sigma_rad: f64,
    view: &ViewGeometry,
) -> Matrix3<f64> {
    let calibration_m2 = (calibration_sigma_rad * view.range_m).powi(2);
    let relief_m = map.height_m * view.off_nadir_rad.clamp(0.0, MAX_OFF_NADIR_RAD).tan();
    let [north, east] = view.ground_direction;
    let along = Vector3::new(north, east, 0.0) * relief_m;
    Matrix3::from_diagonal(&Vector3::new(
        map.horizontal_m.powi(2) + calibration_m2,
        map.horizontal_m.powi(2) + calibration_m2,
        map.height_m.powi(2) + calibration_m2,
    )) + along * along.transpose()
}

#[cfg(test)]
mod tests;
