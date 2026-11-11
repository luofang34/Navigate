//! What the navigator did with each measurement.

use navigate_contract::{MonotonicNanos, SourceId};
use navigate_fusion::RejectionReason;

/// The fate of one measurement.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// The filter accepted it.
    Accepted,
    /// The filter refused it.
    Rejected(RejectionReason),
    /// The navigator did not offer it to the filter, for the stated reason,
    /// for example a star frame that did not identify.
    Unused(&'static str),
}

/// One acceptance record.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptanceRecord {
    /// The sensor.
    pub sensor: SourceId,
    /// Acquisition time of the measurement.
    pub acquired_at: MonotonicNanos,
    /// What the measurement was, for example `gnss_position`.
    pub measurement: &'static str,
    /// What the navigator did with it.
    pub outcome: Outcome,
}

/// Where a navigator writes its acceptance records. The caller owns the
/// storage, so a step of the navigator allocates nothing for them.
pub trait AcceptanceSink {
    /// Takes one record.
    fn record(&mut self, record: AcceptanceRecord);
}

impl AcceptanceSink for Vec<AcceptanceRecord> {
    fn record(&mut self, record: AcceptanceRecord) {
        self.push(record);
    }
}

/// A sink that keeps nothing.
impl AcceptanceSink for () {
    fn record(&mut self, _record: AcceptanceRecord) {}
}
