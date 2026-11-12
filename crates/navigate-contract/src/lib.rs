//! Typed boundary vocabulary for the Navigate system.
//!
//! This crate is the contract external consumers pin (ADR-0005): pure
//! types with no I/O, no behavioral traits, and no math dependencies.
//! It defines navigation solutions, reference frames and ellipsoids,
//! integrity assessments, stamps and source compositions, physical time,
//! flight plans, guidance setpoints, and a DRAFT aiding-observation
//! module whose schema is owed to a joint RFC with the flight-controller
//! side.
//!
//! The crate is `#![no_std]` in every build and does not allocate
//! (ADR-0015). Thus a microcontroller and a ground tool use the same
//! types.
//!
//! Canonical units throughout: radians for angles, meters for distance,
//! meters per second for speed, nanoseconds for monotonic time, and
//! seconds plus attoseconds for physical time.

#![no_std]

#[cfg(test)]
extern crate std;

pub mod aiding;
pub mod composition;
pub mod frame;
pub mod guidance;
pub mod integrity;
pub mod kinematics;
pub mod plan;
pub mod solution;
pub mod stamp;
pub mod time;

pub use composition::{SensorClass, SourceComposition};
pub use frame::{Ellipsoid, FrameId};
pub use guidance::{GuidanceCommand, GuidanceSetpoint, LateralReference};
pub use integrity::{
    FaultDetection, IntegrityAssessment, ProtectionLevels, Redundancy, SolutionQuality,
    SpaceIntegrityAssessment, SpaceUncertainty,
};
pub use kinematics::{
    AttitudeQuaternion, CartesianPosition, CartesianVelocity, GeodeticPosition, NedVelocity,
    SymmetricCov3,
};
pub use plan::{
    AltitudeConstraint, ConstraintField, CourseReference, FlightPlan, HoldGeometry, HoldLegLength,
    Ident, IdentError, LegPath, MAX_IDENT_BYTES, PlanRole, PlanValidationError, TurnDirection,
    TurnType, Waypoint,
};
pub use solution::{CartesianSolution, NavigationSolution};
pub use stamp::{
    ObservationStamp, RadiometricStamps, SolutionStamp, SourceEpoch, SourceId, WrappingSequence,
};
pub use time::{
    ATTOSECONDS_PER_SECOND, ClockAnchor, ClockDomainId, DurationNanos, MonotonicNanos,
    PhysicalInstant, SignedDuration, TimeScale,
};

/// Version of this contract vocabulary. Additive growth does not bump it;
/// a breaking change is a new major decision and a new version.
pub const CONTRACT_VERSION: u32 = 1;
