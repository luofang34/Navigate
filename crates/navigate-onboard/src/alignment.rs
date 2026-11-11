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
}

/// What gave the heading of a self-alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HeadingSource {
    /// The magnetometer at rest, with the declared declination.
    Magnetometer,
}

/// What gave the position of a self-alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PositionSource {
    /// A GNSS fix.
    Gnss,
    /// A surveyed point that the host measured before the flight.
    SurveyedPoint,
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
        /// The name of what that part lacks, for example `rest_evidence`.
        /// The names do not change.
        waiting_for: &'static str,
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
}

impl Default for AlignmentConfig {
    fn default() -> Self {
        Self {
            window_s: 2.0,
            max_heading_sigma_rad: 0.2,
            suspect_after: 5,
        }
    }
}

impl AlignmentConfig {
    /// Whether each value is finite and positive.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        [self.window_s, self.max_heading_sigma_rad]
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
            && self.suspect_after > 0
    }
}
