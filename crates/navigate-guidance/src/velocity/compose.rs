//! Composing the commanded velocity from admitted leg geometry.
//!
//! Limits are enforced with `min`/`max` rather than `clamp`: a
//! misconfigured negative ceiling saturates to that ceiling instead of
//! panicking on inverted bounds.

use crate::admission::AdmittedLeg;
use crate::config::VelocityGuidanceConfig;

/// The horizontal half of a commanded velocity, local-level NED.
pub(crate) struct HorizontalMps {
    /// North component in meters per second.
    pub(crate) north_mps: f64,
    /// East component in meters per second.
    pub(crate) east_mps: f64,
}

/// Composes the horizontal velocity: along-track progress at the
/// arrival-scaled speed, plus the cross-track correction along the
/// course normal, capped by scaling the composed vector. Returns the
/// vector and the along-track speed it was built from, which also
/// scales any gradient-limited vertical rate (NAV-VC-002).
///
/// `remaining_m` is the great-circle distance from ownship to the target
/// waypoint; `cruise_mps` is the caller's cruise after any waypoint
/// speed constraint (NAV-VC-003).
pub(crate) fn horizontal_mps(
    leg: &AdmittedLeg,
    remaining_m: f64,
    cruise_mps: f64,
    config: &VelocityGuidanceConfig,
) -> (HorizontalMps, f64) {
    let (along_north, along_east) = (leg.course_rad.cos(), leg.course_rad.sin());
    // Right of course is the course rotated a quarter turn clockwise;
    // the identity avoids re-evaluating trigonometry a quarter turn on.
    let (right_north, right_east) = (-along_east, along_north);
    let speed_mps = approach_speed_mps(remaining_m, cruise_mps, config);
    let correction_mps = correction_along_right_mps(leg.cross_track_m, config);
    let vector = capped(
        HorizontalMps {
            north_mps: along_north * speed_mps + right_north * correction_mps,
            east_mps: along_east * speed_mps + right_east * correction_mps,
        },
        config.max_horizontal_mps,
    );
    (vector, speed_mps)
}

/// Commanded vertical rate from the profile deviation. Above the profile
/// (positive deviation) descends, so the down component is positive. A
/// declared leg gradient bounds the rate at `|gradient| ·
/// along_speed_mps` when that is tighter than the configured ceiling
/// (NAV-VC-002); the gradient's sign carries the plan's intent, not the
/// limit.
pub(crate) fn down_mps(
    deviation_m: f64,
    gradient: Option<f64>,
    along_speed_mps: f64,
    config: &VelocityGuidanceConfig,
) -> f64 {
    let mut limit = config.max_vertical_mps;
    if let Some(gradient) = gradient {
        limit = limit.min(gradient.abs() * along_speed_mps);
    }
    (config.vertical_gain_per_s * deviation_m)
        .min(limit)
        .max(-limit)
}

/// Along-track speed for the distance remaining: cruise outside the
/// slowdown radius, scaling linearly with the distance inside it, never
/// below the approach floor and never above cruise.
fn approach_speed_mps(remaining_m: f64, cruise_mps: f64, config: &VelocityGuidanceConfig) -> f64 {
    let scale = if config.arrival_slowdown_radius_m > 0.0 {
        (remaining_m / config.arrival_slowdown_radius_m).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let floor_mps = VelocityGuidanceConfig::MIN_APPROACH_SPEED_MPS.min(cruise_mps);
    (cruise_mps * scale).max(floor_mps)
}

/// Correction speed along the right-of-course normal: negative pushes
/// left. A deviation right of course yields a leftward correction —
/// toward the track, never away from it. A non-positive gain corrects
/// nothing rather than driving the vehicle off track.
fn correction_along_right_mps(cross_track_m: f64, config: &VelocityGuidanceConfig) -> f64 {
    let magnitude_mps = (config.cross_track_gain_per_s * cross_track_m.abs())
        .min(config.max_horizontal_mps)
        .max(0.0);
    if cross_track_m > 0.0 {
        -magnitude_mps
    } else {
        magnitude_mps
    }
}

/// The vector scaled to the speed ceiling when it exceeds it. Scaling
/// preserves the commanded direction, which per-axis clipping would
/// rotate.
fn capped(horizontal: HorizontalMps, max_mps: f64) -> HorizontalMps {
    let magnitude_mps = horizontal.north_mps.hypot(horizontal.east_mps);
    if magnitude_mps <= max_mps || magnitude_mps == 0.0 {
        return horizontal;
    }
    let scale = max_mps / magnitude_mps;
    HorizontalMps {
        north_mps: horizontal.north_mps * scale,
        east_mps: horizontal.east_mps * scale,
    }
}
