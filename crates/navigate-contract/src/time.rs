//! Monotonic time vocabulary. Time is always supplied by the caller;
//! nothing in this workspace reads a clock (ADR-0002).

/// Monotonic time in nanoseconds since an unspecified per-process origin.
///
/// Values from different processes (or different [`ClockDomainId`]s) are
/// not comparable; consumers compare only within one domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MonotonicNanos(u64);

impl MonotonicNanos {
    /// Wraps a raw nanosecond reading.
    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Raw nanosecond value.
    #[must_use]
    pub const fn as_nanos(self) -> u64 {
        self.0
    }

    /// Time elapsed since `earlier`, or `None` when `earlier` is not in
    /// the past of `self` (clock misuse is surfaced, never saturated
    /// into a fake zero age).
    #[must_use]
    pub const fn elapsed_since(self, earlier: Self) -> Option<DurationNanos> {
        if self.0 >= earlier.0 {
            Some(DurationNanos(self.0 - earlier.0))
        } else {
            None
        }
    }
}

/// A non-negative span between two [`MonotonicNanos`] readings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DurationNanos(u64);

impl DurationNanos {
    /// Wraps a raw nanosecond span.
    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Builds a span from whole milliseconds.
    #[must_use]
    pub const fn from_millis(millis: u64) -> Self {
        Self(millis.saturating_mul(1_000_000))
    }

    /// Raw nanosecond value.
    #[must_use]
    pub const fn as_nanos(self) -> u64 {
        self.0
    }
}

/// Identifies the clock a timestamp was read from. Stamps carrying
/// different domains are never subtracted from each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ClockDomainId(u32);

impl ClockDomainId {
    /// Wraps a raw domain identifier.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Raw identifier value.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}
