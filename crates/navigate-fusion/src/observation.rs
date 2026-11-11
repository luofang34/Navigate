//! Typed observations offered to the navigation filter (ADR-0003).
//!
//! Producing observations from real hardware is outside this crate: a
//! source adapter builds an [`Observation`] from whatever it measures and
//! the filter judges it at admission. The value enum is
//! `#[non_exhaustive]`. A filter that has no model for a value refuses
//! it by name with [`crate::RejectionReason::UnsupportedMeasurement`]
//! and counts it (ADR-0008). The radiometric and line-of-sight values
//! of ADR-0018 are in the Cartesian frame of
//! [`crate::InertialNavigationFilter`], which has their models.

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
    ///
    /// In the inertial filter, the vertical error correlates with the
    /// horizontal error through the Coriolis and tilt terms of the
    /// error dynamics. Thus an altitude update also moves the horizontal
    /// estimate. When the horizontal uncertainty is large, the movement
    /// is large: tens of meters for each update at an 800 m horizontal
    /// 1-sigma. This is correct estimation, and on average it decreases
    /// the horizontal error. A consumer of the solution, such as
    /// guidance or a display, must tolerate this movement and must not
    /// treat it as a fault.
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
    /// One-way range from an emitter with a corrected clock, such as a
    /// GNSS satellite or a lunar surface beacon (ADR-0018). The model is
    /// `|p − p_e| + c·δt + Δρ`, with the receiver clock bias `c·δt` as a
    /// state. [`crate::InertialNavigationFilter`] refuses it with
    /// [`crate::RejectionReason::ClockStateDisabled`] without a clock
    /// model. [`crate::NavigationFilter`] refuses it with
    /// [`crate::RejectionReason::UnsupportedMeasurement`].
    Pseudorange {
        /// Emitter position at the transmission time in meters, in the
        /// frame of the filter.
        emitter_position_m: [f64; 3],
        /// Measured pseudorange in meters, with the emitter clock
        /// corrected.
        pseudorange_m: f64,
        /// Sum `Δρ` of the Sagnac, ionosphere, and troposphere terms in
        /// meters. The model adds it to the geometric range.
        sagnac_and_media_correction_m: f64,
        /// Pseudorange variance in meters².
        variance_m2: f64,
    },
    /// One-way range rate from an emitter with a corrected clock, such
    /// as a one-way Doppler (ADR-0018). The model is
    /// `u·(v − v_e) + c·δṫ`, with `u` the unit vector from the emitter to
    /// the vehicle and the receiver clock drift `c·δṫ` as a state. The
    /// filter refuses it without a clock model.
    PseudorangeRate {
        /// Emitter position in meters, in the frame of the filter.
        emitter_position_m: [f64; 3],
        /// Emitter velocity relative to the frame in m/s, in the frame
        /// axes.
        emitter_velocity_mps: [f64; 3],
        /// Measured range rate in m/s, with the emitter clock drift
        /// corrected.
        pseudorange_rate_mps: f64,
        /// Range-rate variance in (m/s)².
        variance_m2_per_s2: f64,
    },
    /// Coherent two-way range from a ground station through a
    /// transponder (ADR-0018). The model is the sum of the uplink and the
    /// downlink distances, the general-relativistic delay of each leg,
    /// and the turnaround delay. The round trip cancels the vehicle
    /// clock, so the model does not use the clock states. The
    /// acquisition time is the time of the turnaround. The caller gives
    /// each station at its own transmission or reception time, which
    /// applies the light time.
    TwoWayRange {
        /// Transmitting station position in meters, in the frame of the
        /// filter, at the uplink transmission time.
        tx_station_pos_m: [f64; 3],
        /// Receiving station position in meters, in the frame of the
        /// filter, at the downlink reception time.
        rx_station_pos_m: [f64; 3],
        /// Gravitational parameter `μ` in m³/s² and position in meters of
        /// the body of the Shapiro delay, such as the Sun. Each leg adds
        /// `2μ/c²·ln((r₁ + r₂ + r₁₂)/(r₁ + r₂ − r₁₂))`. `None` adds no
        /// delay.
        gravitation_gm_m3_s2: Option<(f64, [f64; 3])>,
        /// Measured round-trip light distance in meters.
        round_trip_range_m: f64,
        /// Turnaround and equipment delay of the transponder and the
        /// stations, as a distance in meters.
        turnaround_delay_m: f64,
        /// Round-trip range variance in meters².
        variance_m2: f64,
    },
    /// Coherent two-way range rate, such as a two-way Doppler
    /// (ADR-0018). The model is the sum of the uplink and the downlink
    /// range rates. The model does not use the clock states.
    TwoWayRangeRate {
        /// Transmitting station position in meters, in the frame of the
        /// filter.
        tx_station_pos_m: [f64; 3],
        /// Transmitting station velocity relative to the frame in m/s.
        tx_station_vel_mps: [f64; 3],
        /// Receiving station position in meters, in the frame of the
        /// filter.
        rx_station_pos_m: [f64; 3],
        /// Receiving station velocity relative to the frame in m/s.
        rx_station_vel_mps: [f64; 3],
        /// Measured round-trip range rate in m/s.
        round_trip_range_rate_mps: f64,
        /// Range-rate variance in (m/s)².
        variance_m2_per_s2: f64,
    },
    /// A unit direction measured in the body axes, such as a star
    /// tracker direction or an optical navigation bearing (ADR-0018).
    /// The update has two degrees of freedom, in the plane that is
    /// orthogonal to the predicted direction.
    LineOfSight {
        /// Measured unit direction in the body axes.
        measured_body_unit: [f64; 3],
        /// The reference of the direction.
        reference: LineOfSightReference,
        /// Variance of each angle of the plane in radians².
        angular_variance_rad2: f64,
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

/// The reference of a [`ObservationValue::LineOfSight`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LineOfSightReference {
    /// A direction to a far source, such as a star or the Sun, in the
    /// frame axes at the acquisition time. The caller applies the
    /// aberration and the parallax. The update corrects the attitude.
    InertialDirection {
        /// Unit direction in the frame axes.
        direction_in_frame_unit: [f64; 3],
    },
    /// A target at a known position, such as a moon, a planet, an
    /// asteroid, or a beacon. The update corrects the position and the
    /// attitude.
    TargetPosition {
        /// Target position in meters, in the frame of the filter.
        position_in_frame_m: [f64; 3],
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
    /// [`ObservationValue::PseudorangeRate`].
    PseudorangeRate,
    /// [`ObservationValue::TwoWayRange`].
    TwoWayRange,
    /// [`ObservationValue::TwoWayRangeRate`].
    TwoWayRangeRate,
    /// [`ObservationValue::LineOfSight`].
    LineOfSight,
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
            Self::PseudorangeRate { .. } => MeasurementKind::PseudorangeRate,
            Self::TwoWayRange { .. } => MeasurementKind::TwoWayRange,
            Self::TwoWayRangeRate { .. } => MeasurementKind::TwoWayRangeRate,
            Self::LineOfSight { .. } => MeasurementKind::LineOfSight,
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

    /// Which state components (`HorizontalPosition`, `VerticalPosition`,
    /// `Velocity`, `Attitude`) this measurement directly constrains.
    pub(crate) const fn constrained_state(&self) -> crate::provenance::ConstrainedState {
        use crate::provenance::ConstrainedState;
        match self {
            Self::PositionFix { .. }
            | Self::Range { .. }
            | Self::MagneticAnomaly { .. }
            | Self::Pseudorange { .. }
            | Self::TwoWayRange { .. }
            | Self::CartesianPositionFix { .. }
            | Self::CartesianRange { .. } => ConstrainedState::POSITION,
            Self::Altitude { .. } | Self::TerrainClearance { .. } => {
                ConstrainedState::VERTICAL_POSITION
            }
            Self::VelocityFix { .. }
            | Self::PseudorangeRate { .. }
            | Self::TwoWayRangeRate { .. }
            | Self::CartesianVelocityFix { .. } => ConstrainedState::VELOCITY,
            Self::AttitudeFix { .. }
            | Self::LineOfSight {
                reference: LineOfSightReference::InertialDirection { .. },
                ..
            } => ConstrainedState::ATTITUDE,
            Self::VisualPose { .. }
            | Self::LineOfSight {
                reference: LineOfSightReference::TargetPosition { .. },
                ..
            } => ConstrainedState::POSE,
        }
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
