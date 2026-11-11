//! The navigator trait.

use navigate_contract::{
    AttitudeQuaternion, GeodeticPosition, MonotonicNanos, NavigationSolution, NedVelocity,
    SymmetricCov3,
};

use crate::{AcceptanceSink, OnboardError, SensorInput, SensorSuite};

/// The initial state of a navigator, with its covariances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Alignment {
    /// Position.
    pub position: GeodeticPosition,
    /// Velocity, NED.
    pub velocity: NedVelocity,
    /// Body-to-NED attitude.
    pub attitude: AttitudeQuaternion,
    /// Position covariance, NED, m².
    pub position_cov: SymmetricCov3,
    /// Velocity covariance, NED, (m/s)².
    pub velocity_cov: SymmetricCov3,
    /// Attitude covariance, rad².
    pub attitude_cov: SymmetricCov3,
    /// When the state is valid, on the navigator clock.
    pub at: MonotonicNanos,
}

/// A navigator: sensor inputs in, a navigation solution out.
pub trait Navigator {
    /// The declared sensors.
    fn suite(&self) -> &SensorSuite;

    /// Starts the state.
    ///
    /// # Errors
    ///
    /// Refuses an alignment that the filter does not accept.
    fn align(&mut self, alignment: &Alignment) -> Result<(), OnboardError>;

    /// Takes one input and writes what it did with each measurement into
    /// `sink`. A refused aiding measurement is a record, not an error.
    ///
    /// # Errors
    ///
    /// Refuses an input from an undeclared sensor, data of the wrong kind,
    /// and an IMU increment that the filter cannot take.
    fn ingest(
        &mut self,
        input: &SensorInput,
        sink: &mut dyn AcceptanceSink,
    ) -> Result<(), OnboardError>;

    /// The solution at `now`, or `None` before the alignment.
    fn solution(&mut self, now: MonotonicNanos) -> Option<NavigationSolution>;
}
