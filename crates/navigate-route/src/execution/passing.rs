//! The passed-fix rule: a fixed inbound leg ends when the vehicle crosses
//! the perpendicular through its fix (NAV-TT-006).
//!
//! The capture rules of NAV-TT-003 and NAV-TT-004 need the vehicle inside
//! a disc around the fix. A vehicle that passes the fix outside that disc
//! cannot capture it any more, and guidance continues on the inbound
//! course away from the fix. This rule ends that leg.

use navigate_contract::{GeodeticPosition, LegPath};
use navigate_geodesy::{BodyEllipsoid, cross_track_from_course_m, initial_bearing_rad};

use super::Leg;
use crate::turn::fold_to_half_turn;

/// The signed cross-track miss in meters, positive right of the inbound
/// course, when `position` is beyond the perpendicular through the
/// fix of `leg`. `None` while the vehicle is short of the fix, and for a
/// leg that has no fixed inbound course.
///
/// The inbound course at the fix is the reciprocal of the bearing from
/// the fix back to the upstream fix on a track-to-fix leg, and the
/// published course on a course-to-fix leg. A direct-to-fix leg and an
/// initial fix have no fixed inbound course: guidance steers them toward
/// the fix from the present position, so they always end by capture.
///
/// The test is exact on the mean-radius sphere: the vehicle is beyond
/// the perpendicular when the bearing from the fix to the vehicle is
/// less than a quarter turn from the inbound course.
pub(super) fn passed_cross_track_m(
    leg: &Leg<'_>,
    position: &GeodeticPosition,
    ellipsoid: BodyEllipsoid,
) -> Option<f64> {
    let fix = &leg.to.position;
    let inbound_rad = match leg.path {
        LegPath::TrackToFix => {
            initial_bearing_rad(fix, &leg.from?.position) + core::f64::consts::PI
        }
        LegPath::CourseToFix { course_rad, .. } => course_rad,
        _ => return None,
    };
    let off_course_rad = fold_to_half_turn(initial_bearing_rad(fix, position) - inbound_rad);
    // A NaN comparison is false, so a non-finite input never passes.
    if off_course_rad < core::f64::consts::FRAC_PI_2 {
        Some(cross_track_from_course_m(
            position,
            fix,
            inbound_rad,
            ellipsoid,
        ))
    } else {
        None
    }
}
