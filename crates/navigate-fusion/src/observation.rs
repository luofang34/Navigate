//! Typed observations offered to the navigation filter (ADR-0003).
//!
//! Producing observations from real hardware is outside this crate: a
//! source adapter builds an [`Observation`] from whatever it measures and
//! the filter judges it at admission. The value enum is
//! `#[non_exhaustive]`: celestial bearing/elevation and visual-odometry
//! increments are reserved future variants, added without breaking
//! consumers of the existing ones.

use navigate_contract::{
    GeodeticPosition, NedVelocity, ObservationStamp, SourceComposition, SymmetricCov3,
};

/// The measured value of one observation, with its covariance.
///
/// Reserved future variants: celestial bearing/elevation and visual
/// odometry (ADR-0003 names them as designed extensions; a new variant
/// is a new measurement model, never a change to existing ones).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ObservationValue {
    /// An absolute position fix.
    PositionFix {
        /// Measured position, WGS84.
        position: GeodeticPosition,
        /// Measurement covariance in local NED meters².
        covariance: SymmetricCov3,
    },
    /// A velocity measurement.
    VelocityFix {
        /// Measured velocity, local-level NED.
        velocity: NedVelocity,
        /// Measurement covariance in NED (m/s)².
        covariance: SymmetricCov3,
    },
}

/// One observation offered to the filter: a stamped, covariance-bearing
/// value with its declared provenance.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Observation {
    /// Identity, ordering, and acquisition time.
    pub stamp: ObservationStamp,
    /// The measured value with its covariance.
    pub value: ObservationValue,
    /// Declared provenance; empty or estimator-derived compositions are
    /// inadmissible (ADR-0003).
    pub composition: SourceComposition,
}

impl Observation {
    /// Builds an observation from its parts.
    #[must_use]
    pub const fn new(
        stamp: ObservationStamp,
        value: ObservationValue,
        composition: SourceComposition,
    ) -> Self {
        Self {
            stamp,
            value,
            composition,
        }
    }
}
