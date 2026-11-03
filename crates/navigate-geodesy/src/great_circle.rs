//! Great-circle guidance geometry on the spherical mean-radius
//! approximation.
//!
//! All functions here treat the body as a sphere of radius
//! [`BodyEllipsoid::mean_radius_m`] and ignore altitude. The functions
//! without the `_on_` part use [`BodyEllipsoid::WGS84`], whose mean
//! radius is [`crate::wgs84::MEAN_RADIUS_M`]. On the Earth the spherical
//! model differs from the ellipsoid by at most ~0.5% in distance and a
//! comparable fraction in bearing — ample for guidance geometry, where
//! cross-track deviations feed a feedback loop that nulls them and
//! absolute distance drives sequencing thresholds measured in hundreds
//! of meters. The Mars flattening is approximately 1.8 times the Earth
//! flattening, so the error budget on Mars is approximately 1.8 times
//! larger. A lunar sphere has no flattening error. These figures are NOT
//! survey-grade; anything needing centimeter truth goes through the
//! body-fixed/ellipsoidal path instead.
//!
//! The track-deviation helpers ([`cross_track_m`], [`along_track_m`])
//! refuse tracks whose endpoints are closer than 1 mm
//! ([`GeodesyError::DegenerateTrack`]): a track needs direction, and
//! below that floor the bearing is numerically meaningless.
//! [`along_track_m`] is the reserved along-track half of the
//! track-deviation pair — leg-progress and abeam-capture extensions
//! consume it. [`cross_track_from_course_m`] takes a position and a
//! direction instead of two endpoints, which is what a published course
//! gives, and so has no degenerate case to refuse.

use core::f64::consts::TAU;

use navigate_contract::GeodeticPosition;

use crate::ellipsoid::BodyEllipsoid;
use crate::error::GeodesyError;

/// Track endpoints closer than this have no numerically meaningful
/// bearing: below 1 mm of separation the direction is rounding noise,
/// so the track-deviation helpers refuse the track as degenerate.
const MIN_TRACK_SEPARATION_M: f64 = 1e-3;

/// Great-circle distance between `a` and `b` in meters, on the WGS84
/// mean-radius sphere. See [`distance_on_m`].
#[must_use]
pub fn distance_m(a: &GeodeticPosition, b: &GeodeticPosition) -> f64 {
    distance_on_m(a, b, &BodyEllipsoid::WGS84)
}

/// Great-circle distance between `a` and `b` in meters, on the
/// mean-radius sphere of `ellipsoid`. Haversine form, numerically stable
/// for both nearby and antipodal pairs.
#[must_use]
pub fn distance_on_m(a: &GeodeticPosition, b: &GeodeticPosition, ellipsoid: &BodyEllipsoid) -> f64 {
    central_angle_rad(a, b) * ellipsoid.mean_radius_m()
}

/// Initial true bearing of the great circle from `a` toward `b`, in
/// radians in `[0, 2π)`. Coincident or antipodal points have no defined
/// bearing; the `atan2(0, 0)` convention reports 0. The bearing does not
/// depend on the radius of the sphere.
#[must_use]
pub fn initial_bearing_rad(a: &GeodeticPosition, b: &GeodeticPosition) -> f64 {
    let dlon = b.longitude_rad - a.longitude_rad;
    let y = dlon.sin() * b.latitude_rad.cos();
    let x = a.latitude_rad.cos() * b.latitude_rad.sin()
        - a.latitude_rad.sin() * b.latitude_rad.cos() * dlon.cos();
    normalize_bearing_rad(y.atan2(x))
}

/// Signed cross-track distance in meters on the WGS84 mean-radius
/// sphere. See [`cross_track_on_m`].
///
/// # Errors
///
/// [`GeodesyError::DegenerateTrack`] when the endpoints are closer than
/// the 1 mm floor below which the track bearing is numerically
/// meaningless (a track needs direction).
pub fn cross_track_m(
    point: &GeodeticPosition,
    track_start: &GeodeticPosition,
    track_end: &GeodeticPosition,
) -> Result<f64, GeodesyError> {
    cross_track_on_m(point, track_start, track_end, &BodyEllipsoid::WGS84)
}

/// Signed cross-track distance in meters from `point` to the great
/// circle through `track_start` and `track_end`, on the mean-radius
/// sphere of `ellipsoid`: positive when the point lies right of the
/// track (looking from start toward end), negative left. Sign
/// convention matches the contract's deviation-tracking setpoint
/// (`lateral_m` positive right of course).
///
/// Near-antipodal track endpoints make the great-circle direction
/// ill-conditioned and the deviation sign follows floating-point
/// rounding — callers keep legs far from antipodal.
///
/// # Errors
///
/// [`GeodesyError::DegenerateTrack`] when the endpoints are closer than
/// the 1 mm floor below which the track bearing is numerically
/// meaningless (a track needs direction).
pub fn cross_track_on_m(
    point: &GeodeticPosition,
    track_start: &GeodeticPosition,
    track_end: &GeodeticPosition,
    ellipsoid: &BodyEllipsoid,
) -> Result<f64, GeodesyError> {
    let (distance_angle, relative_bearing) =
        track_relative_angles(point, track_start, track_end, ellipsoid)?;
    // Right of track means the bearing to the point is clockwise of the
    // track bearing, making sin(relative bearing) — and the result —
    // positive.
    let sine = (distance_angle.sin() * relative_bearing.sin()).clamp(-1.0, 1.0);
    Ok(sine.asin() * ellipsoid.mean_radius_m())
}

/// Signed cross-track distance in meters from `point` to a course line,
/// on the WGS84 mean-radius sphere. See [`cross_track_from_course_on_m`].
#[must_use]
pub fn cross_track_from_course_m(
    point: &GeodeticPosition,
    anchor: &GeodeticPosition,
    course_rad: f64,
) -> f64 {
    cross_track_from_course_on_m(point, anchor, course_rad, &BodyEllipsoid::WGS84)
}

/// Signed cross-track distance in meters from `point` to the course
/// line, on the mean-radius sphere of `ellipsoid`: the great circle
/// through `anchor` whose true bearing at `anchor`, in the direction of
/// flight, is `course_rad`. Positive when the point lies right of the
/// course, matching [`cross_track_on_m`].
///
/// A course line needs one position and one direction, so — unlike a
/// two-endpoint track — it has no degenerate case to refuse: a point at
/// the anchor lies on the line and reads zero. A non-finite course or
/// position yields a non-finite result rather than a refusal, because
/// the caller screens those before it reaches guidance geometry.
#[must_use]
pub fn cross_track_from_course_on_m(
    point: &GeodeticPosition,
    anchor: &GeodeticPosition,
    course_rad: f64,
    ellipsoid: &BodyEllipsoid,
) -> f64 {
    let distance_angle = central_angle_rad(anchor, point);
    let relative_bearing = initial_bearing_rad(anchor, point) - course_rad;
    let sine = (distance_angle.sin() * relative_bearing.sin()).clamp(-1.0, 1.0);
    sine.asin() * ellipsoid.mean_radius_m()
}

/// Signed along-track distance in meters on the WGS84 mean-radius
/// sphere. See [`along_track_on_m`].
///
/// # Errors
///
/// [`GeodesyError::DegenerateTrack`] when the endpoints are closer than
/// the 1 mm floor below which the track bearing is numerically
/// meaningless (a track needs direction).
pub fn along_track_m(
    point: &GeodeticPosition,
    track_start: &GeodeticPosition,
    track_end: &GeodeticPosition,
) -> Result<f64, GeodesyError> {
    along_track_on_m(point, track_start, track_end, &BodyEllipsoid::WGS84)
}

/// Signed along-track distance in meters, on the mean-radius sphere of
/// `ellipsoid`: how far along the great circle from `track_start` toward
/// `track_end` the foot of the perpendicular from `point` lies. Negative
/// when the foot is behind the start.
///
/// Near-antipodal track endpoints make the great-circle direction
/// ill-conditioned and the deviation sign follows floating-point
/// rounding — callers keep legs far from antipodal.
///
/// # Errors
///
/// [`GeodesyError::DegenerateTrack`] when the endpoints are closer than
/// the 1 mm floor below which the track bearing is numerically
/// meaningless (a track needs direction).
pub fn along_track_on_m(
    point: &GeodeticPosition,
    track_start: &GeodeticPosition,
    track_end: &GeodeticPosition,
    ellipsoid: &BodyEllipsoid,
) -> Result<f64, GeodesyError> {
    let (distance_angle, relative_bearing) =
        track_relative_angles(point, track_start, track_end, ellipsoid)?;
    // Napier's rule for the right spherical triangle:
    // tan(along) = tan(distance)·cos(relative bearing). The atan2 form
    // is total (no division) and yields the sign directly.
    let along_angle = (distance_angle.sin() * relative_bearing.cos()).atan2(distance_angle.cos());
    Ok(along_angle * ellipsoid.mean_radius_m())
}

/// Central angle start→point and the bearing to the point relative to
/// the track bearing, shared by the track-deviation helpers. Refuses
/// tracks too short to define a direction on the sphere of `ellipsoid`.
fn track_relative_angles(
    point: &GeodeticPosition,
    track_start: &GeodeticPosition,
    track_end: &GeodeticPosition,
    ellipsoid: &BodyEllipsoid,
) -> Result<(f64, f64), GeodesyError> {
    let separation_m = distance_on_m(track_start, track_end, ellipsoid);
    if separation_m < MIN_TRACK_SEPARATION_M {
        return Err(GeodesyError::DegenerateTrack { separation_m });
    }
    let distance_angle = central_angle_rad(track_start, point);
    let bearing_to_point = initial_bearing_rad(track_start, point);
    let track_bearing = initial_bearing_rad(track_start, track_end);
    Ok((distance_angle, bearing_to_point - track_bearing))
}

/// Central angle between two positions on the sphere, haversine form.
fn central_angle_rad(a: &GeodeticPosition, b: &GeodeticPosition) -> f64 {
    let half_dlat = (b.latitude_rad - a.latitude_rad) / 2.0;
    let half_dlon = (b.longitude_rad - a.longitude_rad) / 2.0;
    let s = half_dlat.sin() * half_dlat.sin()
        + a.latitude_rad.cos() * b.latitude_rad.cos() * half_dlon.sin() * half_dlon.sin();
    // Rounding can push s a hair past 1 near antipodal pairs.
    2.0 * s.sqrt().min(1.0).asin()
}

/// Folds a bearing into `[0, 2π)`. `rem_euclid` alone can round a tiny
/// negative input up to exactly 2π, so that endpoint folds to 0.
fn normalize_bearing_rad(bearing: f64) -> f64 {
    let folded = bearing.rem_euclid(TAU);
    if folded >= TAU { 0.0 } else { folded }
}

#[cfg(test)]
mod tests;
