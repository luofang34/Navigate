//! Published navigation solutions: a geodetic solution on a reference
//! ellipsoid, and a Cartesian solution in a named frame (ADR-0013).

use crate::composition::SourceComposition;
use crate::frame::{Ellipsoid, FrameId};
use crate::integrity::IntegrityAssessment;
use crate::kinematics::{
    AttitudeQuaternion, CartesianPosition, CartesianVelocity, GeodeticPosition, NedVelocity,
    SymmetricCov3,
};
use crate::stamp::SolutionStamp;

/// One published ownship navigation solution on the surface frame of a
/// body.
///
/// Everything a consumer needs travels together: state, covariance,
/// integrity, provenance, the reference ellipsoid, and the stamp that
/// makes staleness the consumer's honest judgment.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct NavigationSolution {
    /// Identity, ordering, and reference time of this solution.
    pub stamp: SolutionStamp,
    /// Ownship position on [`Self::ellipsoid`].
    pub position: GeodeticPosition,
    /// Ownship velocity, local-level NED.
    pub velocity: NedVelocity,
    /// Ownship attitude when an attitude-bearing source contributes;
    /// absent is absent, never an identity quaternion.
    pub attitude: Option<AttitudeQuaternion>,
    /// Position covariance in local NED meters².
    pub position_cov: SymmetricCov3,
    /// Velocity covariance in NED (m/s)².
    pub velocity_cov: SymmetricCov3,
    /// Attitude covariance in radians² about the NED axes. It is absent
    /// when the attitude is absent or has no covariance.
    pub attitude_cov: Option<SymmetricCov3>,
    /// The integrity assessment this solution rests on.
    pub integrity: IntegrityAssessment,
    /// Declared provenance of the solution.
    pub composition: SourceComposition,
    /// The reference ellipsoid of [`Self::position`]. Its frame is the
    /// body-fixed frame of the solution.
    pub ellipsoid: Ellipsoid,
}

impl NavigationSolution {
    /// Builds a solution from its parts on [`Ellipsoid::WGS84`].
    /// Attitude starts absent. [`Self::with_ellipsoid`] sets a different
    /// ellipsoid, and [`Self::with_attitude`] adds an attitude.
    #[must_use]
    pub const fn new(
        stamp: SolutionStamp,
        position: GeodeticPosition,
        velocity: NedVelocity,
        position_cov: SymmetricCov3,
        velocity_cov: SymmetricCov3,
        integrity: IntegrityAssessment,
        composition: SourceComposition,
    ) -> Self {
        Self {
            stamp,
            position,
            velocity,
            attitude: None,
            position_cov,
            velocity_cov,
            attitude_cov: None,
            integrity,
            composition,
            ellipsoid: Ellipsoid::WGS84,
        }
    }

    /// This solution with an attitude attached.
    #[must_use]
    pub const fn with_attitude(mut self, attitude: AttitudeQuaternion) -> Self {
        self.attitude = Some(attitude);
        self
    }

    /// This solution with an attitude covariance attached.
    #[must_use]
    pub const fn with_attitude_cov(mut self, cov: SymmetricCov3) -> Self {
        self.attitude_cov = Some(cov);
        self
    }

    /// This solution on `ellipsoid`.
    #[must_use]
    pub const fn with_ellipsoid(mut self, ellipsoid: Ellipsoid) -> Self {
        self.ellipsoid = ellipsoid;
        self
    }
}

/// One published ownship solution in a body-centered Cartesian frame,
/// for example an inertial frame during a transfer orbit.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct CartesianSolution {
    /// Identity, ordering, and reference time of this solution.
    pub stamp: SolutionStamp,
    /// The frame of the position, the velocity, and the covariances.
    pub frame: FrameId,
    /// Ownship position in [`Self::frame`].
    pub position: CartesianPosition,
    /// Ownship velocity in the axes of [`Self::frame`].
    pub velocity: CartesianVelocity,
    /// Ownship attitude relative to the axes of [`Self::frame`], when an
    /// attitude-bearing source contributes.
    pub attitude: Option<AttitudeQuaternion>,
    /// Position covariance in meters², in the axes of [`Self::frame`].
    pub position_cov: SymmetricCov3,
    /// Velocity covariance in (m/s)², in the axes of [`Self::frame`].
    pub velocity_cov: SymmetricCov3,
    /// Attitude covariance in radians², when it is known.
    pub attitude_cov: Option<SymmetricCov3>,
    /// The integrity assessment this solution rests on.
    pub integrity: IntegrityAssessment,
    /// Declared provenance of the solution.
    pub composition: SourceComposition,
}

impl CartesianSolution {
    /// Builds a solution from its parts. Attitude starts absent.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "each argument is one required field of the published solution"
    )]
    pub const fn new(
        stamp: SolutionStamp,
        frame: FrameId,
        position: CartesianPosition,
        velocity: CartesianVelocity,
        position_cov: SymmetricCov3,
        velocity_cov: SymmetricCov3,
        integrity: IntegrityAssessment,
        composition: SourceComposition,
    ) -> Self {
        Self {
            stamp,
            frame,
            position,
            velocity,
            attitude: None,
            position_cov,
            velocity_cov,
            attitude_cov: None,
            integrity,
            composition,
        }
    }

    /// This solution with an attitude attached.
    #[must_use]
    pub const fn with_attitude(mut self, attitude: AttitudeQuaternion) -> Self {
        self.attitude = Some(attitude);
        self
    }

    /// This solution with an attitude covariance attached.
    #[must_use]
    pub const fn with_attitude_cov(mut self, cov: SymmetricCov3) -> Self {
        self.attitude_cov = Some(cov);
        self
    }
}

#[cfg(test)]
mod tests;
