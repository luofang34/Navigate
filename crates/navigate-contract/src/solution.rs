//! The published navigation solution.

use crate::composition::SourceComposition;
use crate::integrity::IntegrityAssessment;
use crate::kinematics::{AttitudeQuaternion, GeodeticPosition, NedVelocity, SymmetricCov3};
use crate::stamp::SolutionStamp;

/// One published ownship navigation solution.
///
/// Everything a consumer needs travels together: state, covariance,
/// integrity, provenance, and the stamp that makes staleness the
/// consumer's honest judgment.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct NavigationSolution {
    /// Identity, ordering, and reference time of this solution.
    pub stamp: SolutionStamp,
    /// Ownship position, WGS84.
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
    /// The integrity assessment this solution rests on.
    pub integrity: IntegrityAssessment,
    /// Declared provenance of the solution.
    pub composition: SourceComposition,
}

impl NavigationSolution {
    /// Builds a solution from its parts. Attitude starts absent;
    /// [`Self::with_attitude`] adds it.
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
}
