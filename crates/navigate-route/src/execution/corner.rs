//! Corner classification at a fix (NAV-TT-003).
//!
//! The decision depends only on the plan geometry and mandatory turn
//! directions, so it is fixed at activation and reported per fix by
//! [`super::PlanExecution::corners`].

use core::f64::consts::{FRAC_PI_3, PI, TAU};

use navigate_contract::{FlightPlan, GeodeticPosition, LegPath, TurnDirection, TurnType};
use navigate_geodesy::{BodyEllipsoid, along_track_m, distance_m, initial_bearing_rad};

use super::activation::resolve_path;
use crate::float_math::rem_euclid;
use crate::turn::fold_to_half_turn;

/// Minimum track separation in meters required to define a course.
const MIN_TRACK_SEPARATION_M: f64 = 1e-3;

/// Maximum nominal fly-by track change: 120°.
const FLY_BY_LIMIT_RAD: f64 = 2.0 * FRAC_PI_3;

/// Whole-degree and great-circle convergence allowance on
/// [`FLY_BY_LIMIT_RAD`]: 1°. A turn designed at 120° anticipates; past
/// 121° the fix sequences at the capture radius.
const COURSE_TOLERANCE_RAD: f64 = 1.0_f64.to_radians();

/// Rounding tolerance when comparing zero track change against a mandatory
/// turn direction.
const DIRECTION_EPS_RAD: f64 = 1e-9;

/// Sequencing behavior earned by the corner geometry at a fix.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum Corner {
    /// A fly-by fix between two track-to-fix legs whose turn stays within
    /// the 121° fly-by boundary. The fix sequences on the line abeam the
    /// distance of turn anticipation before it.
    Anticipated {
        /// Track change at the fix in radians, in `[0, π]`.
        track_change_rad: f64,
    },
    /// A fly-by fix between two track-to-fix legs whose turn exceeds the
    /// 121° fly-by boundary (including a mandatory turn opposite the
    /// shorter arc). The fix sequences inside the capture radius.
    PastFlyByLimit {
        /// Track change in the required turn direction, radians, in
        /// `(121°, 2π)`.
        track_change_rad: f64,
    },
    /// No turn to anticipate: a fly-over fix, the terminal fix, a fix
    /// whose inbound or outbound leg is not [`LegPath::TrackToFix`], or a
    /// degenerate track.
    Captured,
}

/// Computes the [`Corner`] at `index` in `plan`.
pub(super) fn corner(plan: &FlightPlan<'_>, index: usize, ellipsoid: BodyEllipsoid) -> Corner {
    let (Some(to), Some(previous)) = (
        plan.waypoints.get(index),
        index.checked_sub(1).and_then(|i| plan.waypoints.get(i)),
    ) else {
        return Corner::Captured;
    };
    let next_index = index.wrapping_add(1);
    let Some(next) = plan.waypoints.get(next_index) else {
        return Corner::Captured;
    };
    let tracks = resolve_path(index, to.path) == LegPath::TrackToFix
        && resolve_path(next_index, next.path) == LegPath::TrackToFix;
    if to.turn != TurnType::FlyBy || !tracks {
        return Corner::Captured;
    }
    let inbound_m = distance_m(&previous.position, &to.position, ellipsoid);
    let outbound_m = distance_m(&to.position, &next.position, ellipsoid);
    if inbound_m < MIN_TRACK_SEPARATION_M || outbound_m < MIN_TRACK_SEPARATION_M {
        return Corner::Captured;
    }
    let inbound = initial_bearing_rad(&to.position, &previous.position) + PI;
    let outbound = initial_bearing_rad(&to.position, &next.position);
    let track_change_rad = directed_track_change_rad(inbound, outbound, next.turn_direction);
    if track_change_rad <= FLY_BY_LIMIT_RAD + COURSE_TOLERANCE_RAD {
        Corner::Anticipated { track_change_rad }
    } else {
        Corner::PastFlyByLimit { track_change_rad }
    }
}

fn directed_track_change_rad(
    inbound_rad: f64,
    outbound_rad: f64,
    mandatory_turn: Option<TurnDirection>,
) -> f64 {
    let Some(direction) = mandatory_turn else {
        return fold_to_half_turn(outbound_rad - inbound_rad);
    };
    let signed = rem_euclid(outbound_rad - inbound_rad + PI, TAU) - PI;
    match direction {
        TurnDirection::Right if signed >= -DIRECTION_EPS_RAD => signed.max(0.0),
        TurnDirection::Right => TAU + signed,
        TurnDirection::Left if signed <= DIRECTION_EPS_RAD => (-signed).max(0.0),
        TurnDirection::Left => TAU - signed,
    }
}

/// Along-track distance to go to `fix` on the great circle from `origin`,
/// in meters. Positive short of the perpendicular through `fix`, negative
/// past it.
pub(super) fn distance_to_go_m(
    position: &GeodeticPosition,
    origin: &GeodeticPosition,
    fix: &GeodeticPosition,
    ellipsoid: BodyEllipsoid,
) -> Option<f64> {
    along_track_m(position, fix, origin, ellipsoid)
        .ok()
        .filter(|m| m.is_finite())
}
