//! The map from a hardware clock reading to a physical instant.

use super::physical::{PhysicalInstant, SignedDuration};
use super::{ClockDomainId, MonotonicNanos};

/// The largest magnitude of [`ClockAnchor::fractional_rate_offset`] that
/// [`ClockAnchor::is_plausible`] accepts: 10^-3, or 1000 parts per
/// million. A quartz oscillator stays well inside this bound. A larger
/// value is a unit error, not a clock property.
const MAX_FRACTIONAL_RATE_OFFSET: f64 = 1e-3;

/// One calibration of a hardware clock against a time scale.
///
/// The anchor maps a reading of the clock `clock` to a
/// [`PhysicalInstant`]: the reference pair plus the elapsed monotonic
/// time, corrected for the rate offset. The anchor holds data only. The
/// estimation of the offset and the rate is the job of the producer.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ClockAnchor {
    /// The clock domain that the anchor calibrates.
    pub clock: ClockDomainId,
    /// The monotonic reading of the reference pair.
    pub reference_monotonic: MonotonicNanos,
    /// The physical instant of the reference pair.
    pub reference_physical: PhysicalInstant,
    /// `(d_physical / d_monotonic) − 1`. It has no unit. It is 0 for a
    /// clock at the nominal rate.
    pub fractional_rate_offset: f64,
    /// One-sigma uncertainty of a mapped instant in nanoseconds, 0 or
    /// more.
    pub uncertainty_ns: f64,
}

impl ClockAnchor {
    /// Builds an anchor from its parts. [`Self::is_plausible`] screens
    /// the values.
    #[must_use]
    pub const fn new(
        clock: ClockDomainId,
        reference_monotonic: MonotonicNanos,
        reference_physical: PhysicalInstant,
        fractional_rate_offset: f64,
        uncertainty_ns: f64,
    ) -> Self {
        Self {
            clock,
            reference_monotonic,
            reference_physical,
            fractional_rate_offset,
            uncertainty_ns,
        }
    }

    /// Whether the anchor can map a reading. The rate offset must be
    /// finite with a magnitude of 10^-3 or less. The uncertainty must be
    /// finite and 0 or more.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        self.fractional_rate_offset.is_finite()
            && self.fractional_rate_offset.abs() <= MAX_FRACTIONAL_RATE_OFFSET
            && self.uncertainty_ns.is_finite()
            && self.uncertainty_ns >= 0.0
    }

    /// Maps `reading` from `reading_clock` to a physical instant on the
    /// scale of [`Self::reference_physical`]. The reading can be before
    /// or after the reference reading.
    ///
    /// The result is `None` when `reading_clock` is not [`Self::clock`],
    /// when the anchor is not plausible, or on overflow. A reading from a
    /// different clock has no relation to this anchor.
    #[must_use]
    pub fn map_monotonic(
        &self,
        reading: MonotonicNanos,
        reading_clock: ClockDomainId,
    ) -> Option<PhysicalInstant> {
        if reading_clock != self.clock || !self.is_plausible() {
            return None;
        }
        let elapsed_ns =
            i128::from(reading.as_nanos()) - i128::from(self.reference_monotonic.as_nanos());
        let elapsed = SignedDuration::from_nanos(elapsed_ns)?;
        // The rate correction is small, so `f64` holds it to far better
        // than one nanosecond for any span a `u64` reading can give.
        let correction_s = elapsed_ns as f64 * self.fractional_rate_offset * 1e-9;
        let correction = SignedDuration::from_seconds_f64(correction_s)?;
        self.reference_physical
            .checked_add(elapsed)?
            .checked_add(correction)
    }
}
