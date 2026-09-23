//! Refusals of the visual fusion adapter.

use thiserror::Error;

/// A visual estimate cannot become a fusion observation.
#[derive(Debug, Error, PartialEq)]
pub enum VisualFusionError {
    /// A budget term is missing, zero, negative, or not finite.
    #[error("visual error budget term {field} must be finite and positive")]
    InvalidBudget {
        /// Name of the budget term.
        field: &'static str,
    },
    /// The estimate converts to a position outside geodetic range.
    #[error("visual estimate position is outside geodetic range")]
    ImplausiblePosition,
    /// The combined covariance is not finite and positive semidefinite.
    #[error("visual estimate covariance is not usable")]
    InvalidCovariance,
    /// The same frame evidence was already converted once.
    #[error("frame evidence {observation_sha256} was already admitted as a fix")]
    RepeatedEvidence {
        /// Evidence digest of the repeated frame.
        observation_sha256: String,
    },
    /// The frame capture time does not follow the last converted frame.
    #[error("frame capture time {received_ns} does not follow {previous_ns}")]
    FrameOrder {
        /// Capture time of the last converted frame.
        previous_ns: u64,
        /// Capture time of the refused frame.
        received_ns: u64,
    },
}
