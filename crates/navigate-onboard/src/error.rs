//! Errors of the onboard boundary.

use navigate_contract::SourceId;
use navigate_fusion::RejectionReason;

/// A refusal of the onboard boundary.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum OnboardError {
    /// The sensor suite or a configuration value is not usable.
    #[error("invalid {field}: {detail}")]
    Config {
        /// The input.
        field: &'static str,
        /// Why it is not usable.
        detail: String,
    },
    /// An input names a sensor that the suite does not declare.
    #[error("sensor {0:?} is not in the declared suite")]
    UnknownSensor(SourceId),
    /// An input does not match the kind of its sensor.
    #[error("sensor {sensor:?} is a {kind}, not a source of {data}")]
    WrongData {
        /// The sensor.
        sensor: SourceId,
        /// Its declared kind.
        kind: &'static str,
        /// The data that came.
        data: &'static str,
    },
    /// The filter refused a step that the navigation cannot continue
    /// without: the alignment or an IMU increment.
    #[error("the filter refused the {stage}: {reason:?}")]
    Refused {
        /// What was refused.
        stage: &'static str,
        /// Navigate's reason.
        reason: RejectionReason,
    },
    /// The star tracker failed outside identification.
    #[error("star tracker: {0}")]
    StarTrack(String),
}
