//! Fusion observations from an attitude solution.

use nalgebra::UnitQuaternion;
use navigate_contract::{
    AttitudeQuaternion, FrameId, GeodeticPosition, ObservationStamp, SensorClass,
    SourceComposition, SymmetricCov3,
};
use navigate_fusion::inertial::ned_to_body_fixed_dcm;
use navigate_fusion::{LineOfSightReference, Observation, ObservationValue};

use crate::{AttitudeSolution, StarTrackError};

/// One `ObservationValue::LineOfSight` for each star of `solution`.
///
/// The measured direction is the centroid direction in the body axes. The
/// reference is `LineOfSightReference::InertialDirection` with the
/// catalogue direction in the ECEF axes at the acquisition time. The
/// observations do not use the solved attitude or a vehicle position, so
/// the star camera stays a source that is independent of the filter.
///
/// Observation `i` has the stamp `stamp` with the sequence advanced `i`
/// times, because the filter admits each sequence of a source only once.
/// All observations have the acquisition time of `stamp`. That time must
/// be the middle of the exposure: under a body rotation each centroid is
/// the star direction at the middle of the exposure. `frame` is the
/// ECEF frame of the filter, for example the frame of its WGS84 ellipsoid.
///
/// The angular variance of star `i` is its
/// [`crate::StarMatch::direction_sigma_rad`] squared plus
/// `mount_variance_rad2`. The camera mount alignment error is common to all
/// stars of one image; the caller gives its variance there, or zero when
/// the filter estimates it.
///
/// # Errors
/// Refuses a mount variance that is not finite and non-negative, and a star
/// whose variance is not finite and positive.
pub fn line_of_sight_observations(
    solution: &AttitudeSolution,
    stamp: ObservationStamp,
    frame: FrameId,
    mount_variance_rad2: f64,
) -> Result<Vec<Observation>, StarTrackError> {
    if !(mount_variance_rad2.is_finite() && mount_variance_rad2 >= 0.0) {
        return Err(StarTrackError::Invalid {
            field: "mount variance",
        });
    }
    let mut stamp = stamp;
    let mut observations = Vec::with_capacity(solution.matches.len());
    for star in &solution.matches {
        let angular_variance_rad2 = star.direction_sigma_rad.powi(2) + mount_variance_rad2;
        if !(angular_variance_rad2.is_finite() && angular_variance_rad2 > 0.0) {
            return Err(StarTrackError::Invalid {
                field: "star direction variance",
            });
        }
        let value = ObservationValue::LineOfSight {
            measured_body_unit: star.body_unit.normalize().into(),
            reference: LineOfSightReference::InertialDirection {
                direction_in_frame_unit: star.ecef_unit.normalize().into(),
            },
            angular_variance_rad2,
        };
        observations.push(Observation::in_frame(stamp, value, celestial(), frame));
        stamp.sequence = stamp.sequence.next();
    }
    Ok(observations)
}

/// An `ObservationValue::AttitudeFix` from the body axes into the local
/// NED axes at `position`.
///
/// The NED axes come from `position`, not from the stars. A horizontal
/// position error of `d` metres tilts the NED axes by about `d / R`
/// radians, with `R` the Earth radius: 1 km gives about 32 arcseconds.
/// Thus this fix shares error with the source of `position`. When the
/// filter also uses that source, the two errors are correlated. Use
/// [`line_of_sight_observations`] when the filter can take them.
///
/// The covariance is [`AttitudeSolution::covariance_body_rad2`]. It does
/// not include the position error.
#[must_use]
pub fn attitude_fix_local_ned(
    solution: &AttitudeSolution,
    position: GeodeticPosition,
    stamp: ObservationStamp,
    frame: FrameId,
) -> Observation {
    let ned_to_ecef = ned_to_body_fixed_dcm(position.latitude_rad, position.longitude_rad);
    let ecef_from_ned = UnitQuaternion::from_matrix(&ned_to_ecef);
    let ned_from_body = ecef_from_ned.inverse() * solution.ecef_from_body;
    let q = ned_from_body.quaternion();
    let c = &solution.covariance_body_rad2;
    let covariance = SymmetricCov3::from_upper_triangle([
        c[(0, 0)],
        c[(0, 1)],
        c[(0, 2)],
        c[(1, 1)],
        c[(1, 2)],
        c[(2, 2)],
    ]);
    let value = ObservationValue::AttitudeFix {
        attitude: AttitudeQuaternion::new(q.w, q.i, q.j, q.k),
        covariance,
    };
    Observation::in_frame(stamp, value, celestial(), frame)
}

fn celestial() -> SourceComposition {
    SourceComposition::of(SensorClass::Celestial)
}

#[cfg(test)]
mod tests;
