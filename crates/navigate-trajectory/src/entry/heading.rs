//! The heading of the velocity and the azimuth to the landing site in
//! the local horizontal plane of the vehicle.

use nalgebra::Vector3;

use crate::error::TrajectoryError;
use crate::state::{StateVector, unit};

use super::{azimuth_rad, wrap_pi};

/// The heading and the course of one evaluation, in radians, clockwise
/// from north.
pub(super) struct Heading {
    /// The azimuth of the horizontal velocity.
    pub(super) heading_rad: f64,
    /// The azimuth of the great circle from the vehicle to the site.
    pub(super) course_rad: f64,
}

impl Heading {
    /// The heading of `vehicle` and the course to `site_m`, both in one
    /// body-fixed frame whose `z` axis is the rotation axis.
    pub(super) fn new(
        vehicle: &StateVector,
        site_m: &Vector3<f64>,
    ) -> Result<Self, TrajectoryError> {
        let up = unit(&vehicle.position_m).ok_or(TrajectoryError::DegenerateOrbit)?;
        let east = unit(&Vector3::z().cross(&up)).ok_or(TrajectoryError::UndefinedHeading)?;
        let north = up.cross(&east);
        let horizontal = |v: &Vector3<f64>| {
            unit(&(v - up * v.dot(&up))).ok_or(TrajectoryError::UndefinedHeading)
        };
        let velocity = horizontal(&vehicle.velocity_mps)?;
        let to_site = horizontal(site_m)?;
        Ok(Self {
            heading_rad: azimuth_rad(&velocity, &east, &north),
            course_rad: azimuth_rad(&to_site, &east, &north),
        })
    }

    /// The heading minus the course, in `[-π, π)`.
    pub(super) fn error_rad(&self) -> f64 {
        wrap_pi(self.heading_rad, self.course_rad)
    }
}
