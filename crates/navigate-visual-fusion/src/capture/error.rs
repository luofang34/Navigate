//! Invalid capture alignment inputs.

/// Capture time and attitude cannot be aligned under the supplied limits.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum CaptureError {
    /// A quaternion, timing interval, or motion limit is invalid.
    #[error("invalid capture alignment: {field}")]
    Invalid {
        /// Input that failed validation.
        field: &'static str,
    },
    /// A required error allowance is unknown.
    #[error("unknown capture error: {field}")]
    UnknownError {
        /// Missing error allowance.
        field: &'static str,
    },
    /// Measurements do not belong to the same clock and boot session.
    #[error("capture alignment clock domains differ")]
    ClockDomain,
    /// The measured offset has no error bound at this source timestamp.
    #[error("clock offset distance {distance_ns} ns exceeds its {maximum_ns} ns validity")]
    ClockWindow {
        /// Distance from the clock correspondence.
        distance_ns: u64,
        /// Valid distance supplied by the host.
        maximum_ns: u64,
    },
    /// The capture time error interval is outside the attitude bracket.
    #[error(
        "capture {capture_ns} ns is not inside attitude bracket {first_ns}..{last_ns} ns with timing errors"
    )]
    NotBracketed {
        /// Camera timestamp.
        capture_ns: u64,
        /// Earlier attitude timestamp.
        first_ns: u64,
        /// Later attitude timestamp.
        last_ns: u64,
    },
}
