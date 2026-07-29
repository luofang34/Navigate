//! DRAFT aiding-observation vocabulary.
//!
//! **This module is a draft.** The schema an FC actually consumes —
//! wire encoding, covariance representation, integrity terms, source-
//! composition encoding — is owed to a joint RFC with the FC side
//! (ADR-0005). These types exist so the shape is reserved and testable;
//! nothing may feed them to a real controller before that RFC lifts the
//! draft status via a superseding decision record.
//!
//! The boundary rules the RFC must preserve (platform ADR on the
//! navigation authority split): observations are timestamped and
//! bounded, carry covariance and integrity metadata, declare their
//! source composition, and the FC independently validates, fuses, or
//! rejects each one. Fused outputs derived from measurements the FC
//! also consumes are declared correlated, never presented as
//! independent.

use crate::composition::SourceComposition;
use crate::integrity::SolutionQuality;
use crate::kinematics::{GeodeticPosition, NedVelocity, SymmetricCov3};
use crate::stamp::ObservationStamp;

/// DRAFT (ADR-0005) — schema not ratified; see the module docs. The
/// measured value of an aiding observation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum AidingValue {
    /// An absolute position fix.
    PositionFix {
        /// Measured position, WGS84.
        position: GeodeticPosition,
        /// Measurement covariance, local NED meters².
        covariance: SymmetricCov3,
    },
    /// A velocity measurement.
    VelocityFix {
        /// Measured velocity, local-level NED.
        velocity: NedVelocity,
        /// Measurement covariance, NED (m/s)².
        covariance: SymmetricCov3,
    },
}

/// DRAFT (ADR-0005) — schema not ratified; see the module docs. One
/// aiding observation offered to a flight controller.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct AidingObservation {
    /// Identity, ordering, and acquisition time.
    pub stamp: ObservationStamp,
    /// The measured value with its covariance.
    pub value: AidingValue,
    /// Declared provenance; the consumer applies its own correlation
    /// policy against it.
    pub composition: SourceComposition,
    /// Producer's quality claim; advisory to the consumer's own
    /// validation, never binding.
    pub quality: SolutionQuality,
}

impl AidingObservation {
    /// DRAFT (ADR-0005) — schema not ratified; see the module docs.
    /// Builds an observation from its parts.
    #[must_use]
    pub const fn new(
        stamp: ObservationStamp,
        value: AidingValue,
        composition: SourceComposition,
        quality: SolutionQuality,
    ) -> Self {
        Self {
            stamp,
            value,
            composition,
            quality,
        }
    }
}
