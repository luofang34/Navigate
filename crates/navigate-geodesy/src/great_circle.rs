//! Great-circle guidance geometry on the spherical mean-radius
//! approximation.
//!
//! All functions here treat the Earth as a sphere of radius
//! [`crate::wgs84::MEAN_RADIUS_M`] and ignore altitude. The spherical
//! model differs from the ellipsoid by at most ~0.5% in distance and a
//! comparable fraction in bearing — ample for guidance geometry, where
//! cross-track deviations feed a feedback loop that nulls them and
//! absolute distance drives sequencing thresholds measured in hundreds
//! of meters. These figures are NOT survey-grade; anything needing
//! centimeter truth goes through the ECEF/ellipsoidal path instead.
//!
//! The track-deviation helpers ([`cross_track_m`], [`along_track_m`])
//! refuse tracks whose endpoints are closer than 1 mm
//! ([`GeodesyError::DegenerateTrack`]): a track needs direction, and
//! below that floor the bearing is numerically meaningless.
//! [`along_track_m`] is the reserved along-track half of the
//! track-deviation pair — leg-progress and abeam-capture extensions
//! consume it.

use core::f64::consts::TAU;

use navigate_contract::GeodeticPosition;

use crate::error::GeodesyError;
use crate::wgs84;

/// Track endpoints closer than this have no numerically meaningful
/// bearing: below 1 mm of separation the direction is rounding noise,
/// so the track-deviation helpers refuse the track as degenerate.
const MIN_TRACK_SEPARATION_M: f64 = 1e-3;

/// Great-circle distance between `a` and `b` in meters, on the
/// mean-radius sphere. Haversine form, numerically stable for both
/// nearby and antipodal pairs.
#[must_use]
pub fn distance_m(a: &GeodeticPosition, b: &GeodeticPosition) -> f64 {
    central_angle_rad(a, b) * wgs84::MEAN_RADIUS_M
}

/// Initial true bearing of the great circle from `a` toward `b`, in
/// radians in `[0, 2π)`. Coincident or antipodal points have no defined
/// bearing; the `atan2(0, 0)` convention reports 0.
#[must_use]
pub fn initial_bearing_rad(a: &GeodeticPosition, b: &GeodeticPosition) -> f64 {
    let dlon = b.longitude_rad - a.longitude_rad;
    let y = dlon.sin() * b.latitude_rad.cos();
    let x = a.latitude_rad.cos() * b.latitude_rad.sin()
        - a.latitude_rad.sin() * b.latitude_rad.cos() * dlon.cos();
    normalize_bearing_rad(y.atan2(x))
}

/// Signed cross-track distance in meters from `point` to the great
/// circle through `track_start` and `track_end`: positive when the
/// point lies right of the track (looking from start toward end),
/// negative left. Sign convention matches the contract's
/// deviation-tracking setpoint (`lateral_m` positive right of course).
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
pub fn cross_track_m(
    point: &GeodeticPosition,
    track_start: &GeodeticPosition,
    track_end: &GeodeticPosition,
) -> Result<f64, GeodesyError> {
    let (distance_angle, relative_bearing) = track_relative_angles(point, track_start, track_end)?;
    // Right of track means the bearing to the point is clockwise of the
    // track bearing, making sin(relative bearing) — and the result —
    // positive.
    let sine = (distance_angle.sin() * relative_bearing.sin()).clamp(-1.0, 1.0);
    Ok(sine.asin() * wgs84::MEAN_RADIUS_M)
}

/// Signed along-track distance in meters: how far along the great
/// circle from `track_start` toward `track_end` the foot of the
/// perpendicular from `point` lies. Negative when the foot is behind
/// the start.
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
pub fn along_track_m(
    point: &GeodeticPosition,
    track_start: &GeodeticPosition,
    track_end: &GeodeticPosition,
) -> Result<f64, GeodesyError> {
    let (distance_angle, relative_bearing) = track_relative_angles(point, track_start, track_end)?;
    // Napier's rule for the right spherical triangle:
    // tan(along) = tan(distance)·cos(relative bearing). The atan2 form
    // is total (no division) and yields the sign directly.
    let along_angle = (distance_angle.sin() * relative_bearing.cos()).atan2(distance_angle.cos());
    Ok(along_angle * wgs84::MEAN_RADIUS_M)
}

/// Central angle start→point and the bearing to the point relative to
/// the track bearing, shared by the track-deviation helpers. Refuses
/// tracks too short to define a direction.
fn track_relative_angles(
    point: &GeodeticPosition,
    track_start: &GeodeticPosition,
    track_end: &GeodeticPosition,
) -> Result<(f64, f64), GeodesyError> {
    let separation_m = distance_m(track_start, track_end);
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
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;

    fn deg(latitude_deg: f64, longitude_deg: f64) -> GeodeticPosition {
        GeodeticPosition::new(latitude_deg.to_radians(), longitude_deg.to_radians(), 0.0)
    }

    #[test]
    fn lax_to_jfk_matches_the_formulary_within_half_a_percent() {
        // Aviation Formulary (Ed Williams) worked example: LAX
        // (33°57'N, 118°24'W) to JFK (40°38'N, 73°47'W) is 0.623585 rad
        // ≈ 2144 nm ≈ 3_970_688 m.
        let lax = deg(33.95, -118.4);
        let jfk = deg(40.0 + 38.0 / 60.0, -(73.0 + 47.0 / 60.0));
        let reference_m = 3_970_688.0;
        let measured_m = distance_m(&lax, &jfk);
        assert!(
            ((measured_m - reference_m) / reference_m).abs() < 0.005,
            "distance {measured_m} m vs reference {reference_m} m"
        );
        // Same worked example: the initial true course is ≈ 66°.
        let bearing_deg = initial_bearing_rad(&lax, &jfk).to_degrees();
        assert!(
            (bearing_deg - 65.89).abs() < 0.5,
            "bearing {bearing_deg}° vs reference 66°"
        );
    }

    #[test]
    fn bearing_stays_in_zero_to_two_pi() {
        let north = initial_bearing_rad(&deg(0.0, 0.0), &deg(1.0, 0.0));
        let west = initial_bearing_rad(&deg(0.0, 0.0), &deg(0.0, -1.0));
        let south = initial_bearing_rad(&deg(1.0, 0.0), &deg(0.0, 0.0));
        assert!(north.abs() < 1e-12);
        assert!((west - 270.0_f64.to_radians()).abs() < 1e-12);
        assert!((south - 180.0_f64.to_radians()).abs() < 1e-9);
        let coincident = initial_bearing_rad(&deg(10.0, 10.0), &deg(10.0, 10.0));
        assert!(coincident.abs() < 1e-15);
    }

    #[test]
    fn cross_track_sign_is_positive_right_of_track() {
        // Eastbound track along the equator; right of track is south.
        let start = deg(0.0, 0.0);
        let end = deg(0.0, 1.0);
        let south = cross_track_m(&deg(-0.1, 0.5), &start, &end).expect("valid track");
        let north = cross_track_m(&deg(0.1, 0.5), &start, &end).expect("valid track");
        // 0.1° of arc on the mean sphere is ≈ 11_119.5 m.
        assert!(
            (south - 11_119.5).abs() < 1.0,
            "south of eastbound track must be positive: {south}"
        );
        assert!(
            (north + 11_119.5).abs() < 1.0,
            "north of eastbound track must be negative: {north}"
        );
    }

    #[test]
    fn along_track_is_signed_from_the_start() {
        let start = deg(0.0, 0.0);
        let end = deg(0.0, 1.0);
        let ahead = along_track_m(&deg(0.1, 0.5), &start, &end).expect("valid track");
        let behind = along_track_m(&deg(0.0, -0.1), &start, &end).expect("valid track");
        // 0.5° along the equator ≈ 55_597.5 m; 0.1° behind ≈ −11_119.5 m.
        assert!((ahead - 55_597.5).abs() < 1.0, "ahead: {ahead}");
        assert!((behind + 11_119.5).abs() < 1.0, "behind: {behind}");
    }

    #[test]
    fn point_on_track_has_zero_cross_track() {
        let start = deg(0.0, 0.0);
        let end = deg(0.0, 1.0);
        let on_track = cross_track_m(&deg(0.0, 0.4), &start, &end).expect("valid track");
        assert!(on_track.abs() < 1e-6, "on-track point: {on_track}");
    }

    #[test]
    fn zero_length_track_is_refused_as_degenerate() {
        let start = deg(10.0, 20.0);
        let point = deg(10.5, 20.5);
        let cross = cross_track_m(&point, &start, &start);
        let along = along_track_m(&point, &start, &start);
        assert!(
            matches!(cross, Err(GeodesyError::DegenerateTrack { separation_m }) if separation_m < 1e-3),
            "cross-track on a zero-length track: {cross:?}"
        );
        assert!(
            matches!(along, Err(GeodesyError::DegenerateTrack { separation_m }) if separation_m < 1e-3),
            "along-track on a zero-length track: {along:?}"
        );
    }
}
