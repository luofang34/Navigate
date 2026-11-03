//! Typed observations offered to the navigation filter (ADR-0003).
//!
//! Producing observations from real hardware is outside this crate: a
//! source adapter builds an [`Observation`] from whatever it measures and
//! the filter judges it at admission. The value enum is
//! `#[non_exhaustive]`. A filter that has no model for a value refuses
//! it by name with [`crate::RejectionReason::UnsupportedMeasurement`]
//! and counts it (ADR-0008). [`ObservationValue::Pseudorange`] is
//! reserved: no filter in this crate has its model.

use navigate_contract::{
    AttitudeQuaternion, CartesianPosition, CartesianVelocity, FrameId, GeodeticPosition,
    NedVelocity, ObservationStamp, SourceComposition, SymmetricCov3,
};

/// The measured value of one observation, with its covariance.
///
/// Each variant is one measurement model (ADR-0003, ADR-0008). A new
/// variant is a new model, never a change to an existing one. A geodetic
/// position or an altitude in a value is on the reference ellipsoid of
/// the filter (ADR-0013).
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ObservationValue {
    /// An absolute position fix.
    PositionFix {
        /// Measured position on the reference ellipsoid.
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
        /// Transmitter position on the reference ellipsoid.
        station: GeodeticPosition,
        /// Measured slant range in meters.
        range_m: f64,
        /// Range variance in meters².
        variance_m2: f64,
    },
    /// Height above the reference ellipsoid, such as a corrected
    /// barometric altitude (ADR-0012). Requires an initialized filter.
    Altitude {
        /// Height above the reference ellipsoid in meters.
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
        /// The position where the adapter linearized the map, on the
        /// reference ellipsoid.
        linearization_point: GeodeticPosition,
        /// Radius around `linearization_point` in meters where the linear
        /// map model is valid.
        valid_radius_m: f64,
        /// Total variance in tesla². It includes the sensor noise and
        /// the map error.
        variance_t2: f64,
    },
    /// GNSS pseudorange to one satellite, before any position solution.
    /// Reserved: every filter refuses it with
    /// [`crate::RejectionReason::UnsupportedMeasurement`] (ADR-0008).
    Pseudorange {
        /// Satellite position at transmission, Earth-centered Earth-fixed meters.
        satellite_ecef_m: [f64; 3],
        /// Measured pseudorange in meters, satellite clock corrected.
        pseudorange_m: f64,
        /// Pseudorange variance in meters².
        variance_m2: f64,
    },
    /// Camera position and attitude from a map reference (ADR-0014).
    /// [`crate::InertialNavigationFilter`] applies it on a body surface as
    /// one update of dimension six with
    /// [`crate::FusionConfig::pose_gate_chi2`].
    /// [`crate::NavigationFilter`] has no attitude state and refuses it
    /// with [`crate::RejectionReason::UnsupportedMeasurement`].
    VisualPose {
        /// Camera position on the reference ellipsoid.
        position: GeodeticPosition,
        /// Position covariance in local NED meters².
        position_covariance: SymmetricCov3,
        /// Unit attitude quaternion from the body axes to the local NED
        /// axes at `position`.
        attitude: AttitudeQuaternion,
        /// Attitude covariance in radians² about the local NED axes at
        /// `position`. The inertial filter rotates it into the body
        /// axes.
        attitude_covariance: SymmetricCov3,
    },
    /// Slant range from the vehicle to the terrain along a body-fixed
    /// beam, such as a radar altimeter range (ADR-0014). The source
    /// adapter reads the terrain height and the terrain slope at the
    /// nominal horizontal position. Requires an initialized filter.
    TerrainClearance {
        /// Measured slant range along the beam in meters.
        clearance_m: f64,
        /// Terrain height above the reference ellipsoid in meters.
        terrain_height_m: f64,
        /// Terrain slope `[∂h/∂north, ∂h/∂east]`. It has no unit.
        terrain_slope_north_east: [f64; 2],
        /// Unit beam direction in the body axes.
        beam_body_unit: [f64; 3],
        /// Range variance in meters².
        variance_m2: f64,
    },
    /// An attitude measurement, such as a star tracker attitude
    /// (ADR-0014). On a body surface the quaternion rotates the body
    /// axes into the local NED axes at the vehicle. In a Cartesian frame
    /// it rotates the body axes into the frame axes.
    AttitudeFix {
        /// Unit attitude quaternion.
        attitude: AttitudeQuaternion,
        /// Covariance of the body-axis rotation vector in radians².
        covariance: SymmetricCov3,
    },
    /// A position in the Cartesian frame of the filter (ADR-0014).
    CartesianPositionFix {
        /// Measured position in meters.
        position: CartesianPosition,
        /// Covariance in meters², in the frame axes.
        covariance: SymmetricCov3,
    },
    /// A velocity relative to the Cartesian frame of the filter, in the
    /// frame axes (ADR-0014).
    CartesianVelocityFix {
        /// Measured velocity in meters per second.
        velocity: CartesianVelocity,
        /// Covariance in (m/s)², in the frame axes.
        covariance: SymmetricCov3,
    },
    /// Range to a beacon at a known position in the Cartesian frame of
    /// the filter (ADR-0014).
    CartesianRange {
        /// Beacon position in meters.
        beacon: CartesianPosition,
        /// Measured range in meters.
        range_m: f64,
        /// Range variance in meters².
        variance_m2: f64,
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
    /// [`ObservationValue::TerrainClearance`].
    TerrainClearance,
    /// [`ObservationValue::AttitudeFix`].
    AttitudeFix,
    /// [`ObservationValue::CartesianPositionFix`].
    CartesianPositionFix,
    /// [`ObservationValue::CartesianVelocityFix`].
    CartesianVelocityFix,
    /// [`ObservationValue::CartesianRange`].
    CartesianRange,
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
            Self::TerrainClearance { .. } => MeasurementKind::TerrainClearance,
            Self::AttitudeFix { .. } => MeasurementKind::AttitudeFix,
            Self::CartesianPositionFix { .. } => MeasurementKind::CartesianPositionFix,
            Self::CartesianVelocityFix { .. } => MeasurementKind::CartesianVelocityFix,
            Self::CartesianRange { .. } => MeasurementKind::CartesianRange,
        }
    }

    /// Whether the six-state [`crate::NavigationFilter`] has the
    /// measurement model for this value. [`crate::InertialNavigationFilter`]
    /// applies a larger set (ADR-0014).
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
/// value with its declared provenance and its declared frame.
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
    /// Declared body-fixed frame of the value (ADR-0013). The filter
    /// refuses a frame that is not the frame of its ellipsoid. `None`
    /// declares no frame. A filter with
    /// [`crate::FusionConfig::require_explicit_frame`] refuses `None`.
    pub frame: Option<FrameId>,
    /// Revision of the map that the source adapter read, when the value
    /// comes from a map (ADR-0014). The inertial filter starts a new map
    /// bias when the source or the revision changes. `None` declares no
    /// revision.
    pub map_revision: Option<u64>,
}

impl Observation {
    /// Builds an observation from its parts without a declared frame.
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
            frame: None,
            map_revision: None,
        }
    }

    /// Builds an observation from its parts with a declared frame.
    #[must_use]
    pub const fn in_frame(
        stamp: ObservationStamp,
        value: ObservationValue,
        composition: SourceComposition,
        frame: FrameId,
    ) -> Self {
        Self::new(stamp, value, composition).with_frame(frame)
    }

    /// This observation with `frame` declared.
    #[must_use]
    pub const fn with_frame(mut self, frame: FrameId) -> Self {
        self.frame = Some(frame);
        self
    }

    /// This observation with `map_revision` declared.
    #[must_use]
    pub const fn with_map_revision(mut self, map_revision: u64) -> Self {
        self.map_revision = Some(map_revision);
        self
    }
}
