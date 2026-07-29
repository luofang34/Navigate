//! Typed boundary vocabulary for the Navigate system.
//!
//! This crate is the contract external consumers pin (ADR-0005): pure
//! types with no I/O, no behavioral traits, and no math dependencies.
//! It defines navigation solutions, integrity assessments, stamps and
//! source compositions, flight plans, guidance setpoints, and a DRAFT
//! aiding-observation module whose schema is owed to a joint RFC with
//! the flight-controller side.
//!
//! Canonical units throughout: radians for angles, meters for distance,
//! meters per second for speed, nanoseconds for monotonic time.

pub mod aiding;
pub mod composition;
pub mod guidance;
pub mod integrity;
pub mod kinematics;
pub mod plan;
pub mod solution;
pub mod stamp;
pub mod time;

pub use composition::{SensorClass, SourceComposition};
pub use guidance::{GuidanceCommand, GuidanceSetpoint};
pub use integrity::{
    FaultDetection, IntegrityAssessment, ProtectionLevels, Redundancy, SolutionQuality,
};
pub use kinematics::{AttitudeQuaternion, GeodeticPosition, NedVelocity, SymmetricCov3};
pub use plan::{AltitudeConstraint, FlightPlan, PlanRole, PlanValidationError, Waypoint};
pub use solution::NavigationSolution;
pub use stamp::{ObservationStamp, SolutionStamp, SourceEpoch, SourceId, WrappingSequence};
pub use time::{ClockDomainId, DurationNanos, MonotonicNanos};

/// Version of this contract vocabulary. Additive growth does not bump it;
/// a breaking change is a new major decision and a new version.
pub const CONTRACT_VERSION: u32 = 1;
