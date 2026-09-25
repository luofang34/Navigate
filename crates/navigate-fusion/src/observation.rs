//! Typed observations offered to the navigation filter (ADR-0003).
//!
//! Producing observations from real hardware is outside this crate: a
//! source adapter builds an [`Observation`] from whatever it measures and
//! the filter judges it at admission. The value enum is
//! `#[non_exhaustive]`. Reserved variants name measurements that this
//! build cannot use yet (ADR-0008). The filter refuses them by name with
//! [`crate::RejectionReason::UnsupportedMeasurement`] and counts them.
//! A reserved variant becomes usable when its measurement model lands.

use navigate_contract::{
    AttitudeQuaternion, GeodeticPosition, NedVelocity, ObservationStamp, SourceComposition,
    SymmetricCov3,
};

/// The measured value of one observation, with its covariance.
///
/// Each variant is one measurement model (ADR-0003, ADR-0008). A new
/// variant is a new model, never a change to an existing one.
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
    /// Slant range to a transmitter at a known position, such as a DME.
    /// Reserved: refused until the range model lands (ADR-0008).
    Range {
        /// Transmitter position, WGS84.
        station: GeodeticPosition,
        /// Measured slant range in meters.
        range_m: f64,
        /// Range variance in meters².
        variance_m2: f64,
    },
    /// GNSS pseudorange to one satellite, before any position solution.
    /// Reserved: refused until the pseudorange model lands (ADR-0008).
    Pseudorange {
        /// Satellite position at transmission, Earth-centered Earth-fixed meters.
        satellite_ecef_m: [f64; 3],
        /// Measured pseudorange in meters, satellite clock corrected.
        pseudorange_m: f64,
        /// Pseudorange variance in meters².
        variance_m2: f64,
    },
    /// Camera position and attitude from a map reference.
    /// Reserved: refused until the filter carries attitude (ADR-0008).
    VisualPose {
        /// Camera position, WGS84.
        position: GeodeticPosition,
        /// Position covariance in local NED meters².
        position_covariance: SymmetricCov3,
        /// Camera attitude.
        attitude: AttitudeQuaternion,
        /// Attitude covariance in radians² about NED axes.
        attitude_covariance: SymmetricCov3,
    },
}

/// The measurement model that an [`ObservationValue`] needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MeasurementKind {
    /// [`ObservationValue::PositionFix`].
    PositionFix,
    /// [`ObservationValue::VelocityFix`].
    VelocityFix,
    /// [`ObservationValue::Range`].
    Range,
    /// [`ObservationValue::Pseudorange`].
    Pseudorange,
    /// [`ObservationValue::VisualPose`].
    VisualPose,
}

impl ObservationValue {
    /// The measurement model this value needs.
    #[must_use]
    pub const fn kind(&self) -> MeasurementKind {
        match self {
            Self::PositionFix { .. } => MeasurementKind::PositionFix,
            Self::VelocityFix { .. } => MeasurementKind::VelocityFix,
            Self::Range { .. } => MeasurementKind::Range,
            Self::Pseudorange { .. } => MeasurementKind::Pseudorange,
            Self::VisualPose { .. } => MeasurementKind::VisualPose,
        }
    }

    /// Whether this build has the measurement model for this value.
    #[must_use]
    pub const fn is_supported(&self) -> bool {
        matches!(self, Self::PositionFix { .. } | Self::VelocityFix { .. })
    }
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
