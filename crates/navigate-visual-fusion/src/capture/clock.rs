//! Apply a host-validated clock offset over its stated validity interval.
use super::CaptureError;
use navigate_contract::{ClockDomainId, DurationNanos, MonotonicNanos};

/// A timestamp with an explicit absolute error bound.
#[derive(Clone, Copy, Debug)]
pub struct TimedCapture {
    /// Estimated measurement time, not packet arrival time.
    pub at: MonotonicNanos,
    /// Domain and boot session of this timestamp.
    pub clock: ClockDomainId,
    /// Known maximum timestamp error. Zero is an explicit exact-clock claim.
    pub error_bound: DurationNanos,
}

/// A measured clock correspondence. It does not estimate an offset from arrivals.
///
/// A host can use a TIMESYNC filter or a shared hardware clock. The error bound
/// must cover clock drift throughout `maximum_span`, offset error, quantization,
/// and measurement timestamp error. Re-estimate it when this interval expires.
#[derive(Clone, Copy, Debug)]
pub struct ClockAlignment {
    /// Source clock and boot session. Reboots require a new domain.
    pub source_clock: ClockDomainId,
    /// Source timestamp at the measured correspondence.
    pub source_anchor: MonotonicNanos,
    /// Corresponding target timestamp and total error bound.
    pub target_anchor: TimedCapture,
    /// Largest source-time distance for which the error bound is valid.
    pub maximum_span: DurationNanos,
}
impl ClockAlignment {
    /// Map an extended source timestamp to the target domain.
    ///
    /// # Errors
    /// Refuses a foreign boot/domain, an expired mapping, and integer overflow.
    pub fn map(
        &self,
        at: MonotonicNanos,
        clock: ClockDomainId,
    ) -> Result<TimedCapture, CaptureError> {
        if clock != self.source_clock {
            return Err(CaptureError::ClockDomain);
        }
        let distance = at.as_nanos().abs_diff(self.source_anchor.as_nanos());
        if distance > self.maximum_span.as_nanos() || self.maximum_span.as_nanos() == 0 {
            return Err(CaptureError::ClockWindow {
                distance_ns: distance,
                maximum_ns: self.maximum_span.as_nanos(),
            });
        }
        let delta = i128::from(at.as_nanos()) - i128::from(self.source_anchor.as_nanos());
        let mapped = i128::from(self.target_anchor.at.as_nanos()) + delta;
        let mapped = u64::try_from(mapped).map_err(|_| CaptureError::Invalid {
            field: "mapped timestamp range",
        })?;
        Ok(TimedCapture {
            at: MonotonicNanos::from_nanos(mapped),
            ..self.target_anchor
        })
    }
}
