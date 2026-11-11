//! How a navigator gets its first state, and what it states about it.
//!
//! A navigator aligns itself from its declared sensors, or it takes a
//! transfer alignment from another estimator or a survey
//! ([`crate::Navigator::align`]). The first state never comes from a value
//! that only a simulator knows.

use navigate_contract::MonotonicNanos;

/// What gave the first state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AlignmentSource {
    /// The navigator aligned itself from its sensors.
    Sensors,
    /// The host gave the state of another estimator or of a survey.
    Transfer,
    /// An evaluation gave the true state. A flight never has this source.
    Truth,
    /// An operator declared position, height, heading and speed; the
    /// navigator levelled from its own inertial data.
    Declared,
}

/// What gave the heading of a self-alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HeadingSource {
    /// The magnetometer at rest, with the field that the host states.
    Magnetometer,
    /// The course over the ground of a vehicle that moves along its nose,
    /// to the configured allowance for its crab angle.
    GnssCourse,
    /// The heading that an operator declared.
    Declared,
}

/// What gave the position of a self-alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PositionSource {
    /// A GNSS fix.
    Gnss,
    /// A surveyed point that the host measured before the flight.
    SurveyedPoint,
    /// A position that the host states, with no statement of rest.
    KnownPosition,
    /// The position that an operator declared with the start.
    Declared,
}

/// The part of a self-alignment that is not complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AlignmentPhase {
    /// Roll and pitch from the specific force at rest.
    Levelling,
    /// The heading.
    Heading,
    /// The position.
    Position,
}

/// What an alignment in progress lacks. A host shows and counts these by
/// [`AlignmentNeed::name`], which does not change; a new case gets a new
/// variant and a new name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AlignmentNeed {
    /// No inertial increment has come.
    InertialIncrements,
    /// The rest window is not complete.
    RestWindow,
    /// The window holds too few inertial increments to judge rest by
    /// their scatter.
    RestSamples,
    /// The specific force, the body rate, or the GNSS velocity of the
    /// window is not that of a vehicle at rest.
    Rest,
    /// No sensor says that the vehicle is at rest: a GNSS fix with a
    /// velocity or a surveyed point.
    RestEvidence,
    /// No magnetometer is declared, or it gave no usable field, or the
    /// host stated no magnetic reference.
    HeadingSource,
    /// The strength of the measured field is not that of the stated
    /// reference: the field is disturbed.
    FieldStrength,
    /// The heading is less certain than the alignment accepts.
    HeadingUncertain,
    /// No GNSS fix with a covariance, no surveyed point, and no known
    /// position has come.
    PositionSource,
    /// The vehicle moves, and the alignment has not two GNSS fixes with a
    /// velocity a usable time apart.
    VelocityFixes,
    /// A GNSS velocity of a moving vehicle has no stated or declared
    /// one-sigma.
    VelocityAccuracy,
    /// The inertial increments do not cover the span between two GNSS
    /// fixes of a moving vehicle.
    InertialGap,
    /// The specific force of the inertial increments between two fixes and
    /// that of the change of the GNSS velocity differ in size by more than
    /// their declared errors explain.
    ForceDisagreement,
    /// The vehicle moves too slowly for its course to be a heading as well
    /// as the alignment needs.
    CourseSpeed,
    /// The specific force before a declared start is not that of a vehicle
    /// that does not accelerate, within what the inertial errors explain.
    SteadyMotion,
    /// A declared height has no known reference.
    HeightDatum,
}

impl AlignmentNeed {
    /// The name of the need, for a host to show and count.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::InertialIncrements => "inertial_increments",
            Self::RestWindow => "rest_window",
            Self::RestSamples => "rest_samples",
            Self::Rest => "rest",
            Self::RestEvidence => "rest_evidence",
            Self::HeadingSource => "heading_source",
            Self::FieldStrength => "field_strength",
            Self::HeadingUncertain => "heading_uncertain",
            Self::PositionSource => "position_source",
            Self::VelocityFixes => "velocity_fixes",
            Self::VelocityAccuracy => "velocity_accuracy",
            Self::InertialGap => "inertial_gap",
            Self::ForceDisagreement => "force_disagreement",
            Self::CourseSpeed => "course_speed",
            Self::SteadyMotion => "steady_motion",
            Self::HeightDatum => "height_datum",
        }
    }
}

/// What a completed alignment used and how well it knows each part.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct AlignmentRecord {
    /// When the state is valid, on the navigator clock.
    pub at: MonotonicNanos,
    /// What gave the state.
    pub source: AlignmentSource,
    /// What gave the heading; `None` for a state that the host gave.
    pub heading: Option<HeadingSource>,
    /// What gave the position; `None` for a state that the host gave.
    pub position: Option<PositionSource>,
    /// One-sigma of roll and of pitch, radians.
    pub tilt_sigma_rad: f64,
    /// One-sigma of the heading, radians.
    pub heading_sigma_rad: f64,
    /// The gyroscope bias that the rest window gave, rad/s, in the body
    /// axes: the mean rate less the Earth rate. The filter starts its
    /// gyroscope bias from it; `None` for a state that the host gave,
    /// after which the filter starts the biases at zero.
    pub gyro_bias_rad_s: Option<[f64; 3]>,
    /// One-sigma of that bias on each axis, rad/s: the gyroscope noise
    /// over the window and the Earth rate turned by the attitude error.
    pub gyro_bias_sigma_rad_s: Option<f64>,
}

/// Where a navigator is on the way to its first state.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum AlignmentStatus {
    /// No state yet. A solution does not come before the alignment.
    Aligning {
        /// The part that is not complete.
        phase: AlignmentPhase,
        /// What that part lacks.
        waiting_for: AlignmentNeed,
    },
    /// The state is set.
    Aligned {
        /// What the alignment used.
        record: AlignmentRecord,
        /// The filter refused several GNSS positions in a row at its
        /// innovation gate after the alignment. The navigator does not
        /// align again by itself: the host calls
        /// [`crate::Navigator::realign`].
        suspect: bool,
    },
}

/// The Earth's field at the place, as the host states it from a field
/// model, for the heading of a magnetometer. The host states it again when
/// the place or the epoch changes enough to change it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MagneticReference {
    /// Declination, radians, east positive.
    pub declination_rad: f64,
    /// One-sigma of the declination, radians: the error of the model and
    /// the local anomaly.
    pub declination_sigma_rad: f64,
    /// Total intensity of the field, tesla.
    pub intensity_t: f64,
    /// One-sigma of the intensity, tesla.
    pub intensity_sigma_t: f64,
}

impl MagneticReference {
    /// Whether each value is finite, the one-sigmas and the intensity
    /// positive.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        self.declination_rad.is_finite()
            && [
                self.declination_sigma_rad,
                self.intensity_t,
                self.intensity_sigma_t,
            ]
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
    }
}

/// How a navigator aligns itself.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct AlignmentConfig {
    /// Length of one rest window, seconds. The tests of rest and the
    /// velocity one-sigma of a state of rest come from the declared
    /// budgets of the sensors, not from this configuration.
    pub window_s: f64,
    /// Largest heading one-sigma that an alignment accepts, radians. A
    /// larger error is outside the linear range of the filter. The
    /// default is an engineering value, not a measured limit of the
    /// filter.
    pub max_heading_sigma_rad: f64,
    /// Count of GNSS positions in a row that the filter refuses at its
    /// innovation gate before the alignment is suspect.
    pub suspect_after: u32,
    /// One-sigma of the angle between the course over the ground and the
    /// heading of a moving vehicle, radians: the crab angle in wind and
    /// the sideslip. `None` says that the course is not a heading, as for
    /// a multirotor, which moves in each direction at each heading; a
    /// moving vehicle then waits for rest. A vehicle on its wheels in a
    /// ground roll has a small value; a fixed wing in the air has the
    /// crab angle of the wind that it can meet. The value is the host's
    /// engineering choice for its vehicle.
    pub course_heading_sigma_rad: Option<f64>,
    /// One-sigma of the wind on each horizontal axis, m/s, for a declared
    /// start whose speed is an airspeed: the statement does not say the
    /// wind. An engineering choice for the vehicle's weather limits.
    pub declared_wind_sigma_m_s: f64,
    /// One-sigma of the vertical speed of a declared start, m/s: the
    /// statement does not say it. An engineering choice for the vehicle.
    pub declared_vertical_speed_sigma_m_s: f64,
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            window_s: 2.0,
            max_heading_sigma_rad: 0.2,
            suspect_after: 5,
            course_heading_sigma_rad: None,
            declared_wind_sigma_m_s: 10.0,
            declared_vertical_speed_sigma_m_s: 3.0,
        }
    }
}

impl AlignmentConfig {
    /// Whether each value is finite and positive.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        [
            self.window_s,
            self.max_heading_sigma_rad,
            self.declared_wind_sigma_m_s,
            self.declared_vertical_speed_sigma_m_s,
        ]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0)
            && self.suspect_after > 0
            && self
                .course_heading_sigma_rad
                .is_none_or(|v| v.is_finite() && v >= 0.0)
    }
}
