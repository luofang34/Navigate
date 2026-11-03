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
    /// Slant range to a transmitter at a known position, such as a DME
    /// (ADR-0008). Requires an initialized filter.
    Range {
        /// Transmitter position, WGS84.
        station: GeodeticPosition,
        /// Measured slant range in meters.
        range_m: f64,
        /// Range variance in meters².
        variance_m2: f64,
    },
    /// Height above the WGS84 ellipsoid, such as a corrected barometric
    /// altitude (ADR-0012). Requires an initialized filter.
    Altitude {
        /// Height above the WGS84 ellipsoid in meters.
        altitude_m: f64,
        /// Altitude variance in meters².
        variance_m2: f64,
    },
    /// Scalar magnetic anomaly, linearized against an anomaly map by the
    /// source adapter (ADR-0012). All magnetic values are SI: tesla,
    /// tesla per meter, and tesla². Requires an initialized filter.
    MagneticAnomaly {
        /// Measured anomaly in tesla, after platform compensation.
        measured_t: f64,
        /// Map anomaly at the linearization point in tesla.
        map_t_at_lin: f64,
        /// Map gradient in tesla per meter. The axes are the NED axes of
        /// the tangent plane at `linearization_point`.
        gradient_ned_t_per_m: [f64; 3],
        /// The position where the adapter linearized the map, WGS84.
        linearization_point: GeodeticPosition,
        /// Radius around `linearization_point` in meters where the linear
        /// map model is valid.
        valid_radius_m: f64,
        /// Total variance in tesla². It includes the sensor noise and
        /// the map error.
        variance_t2: f64,
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
    /// [`ObservationValue::Altitude`].
    Altitude,
    /// [`ObservationValue::MagneticAnomaly`].
    MagneticAnomaly,
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
            Self::Altitude { .. } => MeasurementKind::Altitude,
            Self::MagneticAnomaly { .. } => MeasurementKind::MagneticAnomaly,
            Self::Pseudorange { .. } => MeasurementKind::Pseudorange,
            Self::VisualPose { .. } => MeasurementKind::VisualPose,
        }
    }

    /// Whether this build has the measurement model for this value.
    #[must_use]
    pub const fn is_supported(&self) -> bool {
        matches!(
            self,
            Self::PositionFix { .. }
                | Self::VelocityFix { .. }
                | Self::Range { .. }
                | Self::Altitude { .. }
                | Self::MagneticAnomaly { .. }
        )
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
