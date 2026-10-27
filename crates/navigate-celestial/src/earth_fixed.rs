use crate::{
    AnalyticEphemeris, AnalyticError, Correction, Epoch, FrameId, ObservationError, State,
    earth_rotation, observe_body,
};
use nalgebra::Vector3;

const SUN: i32 = 10;
const MOON: i32 = 301;
/// IAU 2015 nominal solar radius, in metres.
const SUN_RADIUS_M: f64 = 695_700_000.0;
/// IAU mean lunar radius, in metres.
const MOON_RADIUS_M: f64 = 1_737_400.0;

/// The apparent Sun or Moon for an observer, in Earth-fixed axes.
///
/// The axes are ECEF: x toward 0° N 0° E, y toward 0° N 90° E, and z toward
/// the north pole. They come from the model in [`sun_ecef`], not from
/// ITRF data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EarthFixedSight {
    /// Unit direction from the observer toward the body centre, with light
    /// time and aberration applied. No refraction is applied.
    pub direction: [f64; 3],
    /// Distance from the observer to the body centre at emission, in metres.
    pub distance_m: f64,
    /// Apparent angular radius of the body disc, in radians.
    pub angular_radius_rad: f64,
}

/// The point on the Earth where the Sun is overhead.
///
/// The latitude is geodetic: the ellipsoid normal at this point is parallel
/// to the Sun direction from the Earth's centre. No ellipsoid parameter is
/// necessary for this. The Sun direction from the point itself differs by the
/// solar parallax, less than 9 arcseconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubsolarPoint {
    /// Geodetic latitude in degrees, positive north.
    pub latitude_deg: f64,
    /// Longitude in degrees, positive east, in `[-180, 180]`.
    pub longitude_deg: f64,
}

/// The apparent Sun from an Earth-fixed observer, without kernel files.
///
/// `observer_ecef_m` is a position in ECEF axes, in metres, fixed to the
/// Earth's surface. `None` is the Earth's centre. The calculation uses
/// [`AnalyticEphemeris`] and [`Correction::VacuumReception`].
///
/// The Earth rotation model applies IAU 1976 precession, the main IAU 1980
/// nutation terms, and GAST. It takes UT1 as equal to UTC and omits polar
/// motion. From 1972, leap seconds keep the UT1 difference within 0.9 s, which
/// rotates directions by up to 13.5 arcseconds about the pole. Before 1972,
/// the model reads the UTC clock reading as UT1, because civil time (GMT)
/// then followed the Earth's rotation. From 2000 to 2026, the tests bound the
/// direction error relative to JPL DE421 with ITRF93 Earth orientation at
/// 15 arcseconds for the Sun and 22 arcseconds for the Moon, for observers at
/// the Earth's centre and on its surface. From 1900 to 1971, the tests bound
/// it relative to Skyfield with historical Delta T at 15 arcseconds for the
/// Sun and 30 arcseconds for the Moon.
///
/// # Errors
/// Returns an error for an epoch outside 1900 to 2100 TDB, or for a
/// non-finite observer position.
pub fn sun_ecef(
    epoch: Epoch,
    observer_ecef_m: Option<[f64; 3]>,
) -> Result<EarthFixedSight, ObservationError<AnalyticError>> {
    sight(SUN, SUN_RADIUS_M, epoch, observer_ecef_m)
}

/// The apparent Moon from an Earth-fixed observer, without kernel files.
///
/// An observer on the surface sees the Moon up to about 1° from its
/// geocentric direction. Give the observer position for a topocentric view.
/// Axes, model, and bounds are the same as for [`sun_ecef`].
///
/// # Errors
/// Returns an error for an epoch outside 1900 to 2100 TDB, or for a
/// non-finite observer position.
pub fn moon_ecef(
    epoch: Epoch,
    observer_ecef_m: Option<[f64; 3]>,
) -> Result<EarthFixedSight, ObservationError<AnalyticError>> {
    sight(MOON, MOON_RADIUS_M, epoch, observer_ecef_m)
}

/// The subsolar point from the geocentric apparent Sun.
///
/// # Errors
/// Returns an error for an epoch outside 1900 to 2100 TDB.
pub fn subsolar_point(epoch: Epoch) -> Result<SubsolarPoint, ObservationError<AnalyticError>> {
    let [x, y, z] = sun_ecef(epoch, None)?.direction;
    Ok(SubsolarPoint {
        latitude_deg: z.clamp(-1.0, 1.0).asin().to_degrees(),
        longitude_deg: y.atan2(x).to_degrees(),
    })
}

fn sight(
    body: i32,
    radius_m: f64,
    epoch: Epoch,
    observer_ecef_m: Option<[f64; 3]>,
) -> Result<EarthFixedSight, ObservationError<AnalyticError>> {
    let observer = State {
        position_m: observer_ecef_m.unwrap_or([0.0; 3]),
        velocity_m_s: [0.0; 3],
        epoch,
        frame: FrameId::EARTH_ITRF93,
    };
    let sight = observe_body(
        &AnalyticEphemeris,
        observer,
        body,
        Correction::VacuumReception,
    )?;
    let to_earth_fixed = earth_rotation::j2000_to_earth_fixed(sight.received_at);
    let direction = to_earth_fixed * Vector3::from(sight.direction);
    Ok(EarthFixedSight {
        direction: direction.normalize().into(),
        distance_m: sight.coordinate_range_m,
        angular_radius_rad: (radius_m / sight.coordinate_range_m).asin(),
    })
}

#[cfg(test)]
mod tests;
