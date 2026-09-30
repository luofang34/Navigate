//! Bounded frame admission from measured processing cost and a host time budget.
use crate::VisualError;
use std::time::Duration;

/// Timing requirements supplied by the host. These are not device utilization telemetry.
#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig {
    /// Smallest permitted interval between processing starts.
    pub minimum_interval: Duration,
    /// Largest useful interval. A smaller compute budget can make this unattainable.
    pub maximum_interval: Duration,
    /// Fraction of elapsed time available for this serialized processing lane, from zero to one.
    pub utilization: f64,
}

/// Latest timing decision, independent of the image matcher and its device.
#[derive(Clone, Copy, Debug)]
pub struct SamplingStatus {
    /// Smoothed processing cost. A cost increase applies immediately.
    pub estimated_cost: Duration,
    /// Host processing-time fraction. This is not measured device utilization.
    pub utilization: f64,
    /// Requested interval at the current host budget.
    pub interval: Duration,
    /// The budget cannot meet the host's largest useful interval.
    pub deadline_unattainable: bool,
    /// Admission is paused by the host.
    pub paused: bool,
}

/// Admit at most one observation at a time without a work queue.
///
/// The host supplies a monotonic clock and the fraction of time that it can spare.
/// Complete every admitted observation, including a rejection. Measured cost must
/// include preprocessing, inference, rendering and verification. This controller
/// cannot preempt a running inference call or promise a hard real-time deadline.
pub struct AdaptiveSampler {
    config: SamplingConfig,
    estimated_seconds: f64,
    active: Option<Duration>,
    previous_start: Option<Duration>,
    last_clock: Option<Duration>,
}
impl AdaptiveSampler {
    /// Create a controller with a conservative first-observation cost estimate.
    ///
    /// # Errors
    /// Rejects invalid intervals, budgets, or a zero initial cost.
    pub fn new(config: SamplingConfig, initial_cost: Duration) -> Result<Self, VisualError> {
        if config.minimum_interval.is_zero()
            || config.maximum_interval < config.minimum_interval
            || initial_cost.is_zero()
        {
            return Err(VisualError::Invalid {
                field: "sampling intervals",
            });
        }
        let mut sampler = Self {
            config,
            estimated_seconds: initial_cost.as_secs_f64(),
            active: None,
            previous_start: None,
            last_clock: None,
        };
        sampler.set_utilization(config.utilization)?;
        Ok(sampler)
    }
    /// Change the available host budget. Zero pauses admission without cancelling active work.
    ///
    /// # Errors
    /// Rejects non-finite values or fractions outside zero through one.
    pub fn set_utilization(&mut self, fraction: f64) -> Result<(), VisualError> {
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
            return Err(VisualError::Invalid {
                field: "sampling utilization",
            });
        }
        self.config.utilization = fraction;
        Ok(())
    }
    /// Admit the newest available frame when due. The caller discards unadmitted frames.
    ///
    /// # Errors
    /// Rejects a clock value older than a preceding admission attempt.
    pub fn begin(&mut self, now: Duration) -> Result<bool, VisualError> {
        if self.last_clock.is_some_and(|last| now < last) {
            return Err(VisualError::Invalid {
                field: "sampling clock order",
            });
        }
        self.last_clock = Some(now);
        let status = self.status();
        if self.active.is_some()
            || status.paused
            || self
                .previous_start
                .is_some_and(|last| now.saturating_sub(last) < status.interval)
        {
            return Ok(false);
        }
        self.active = Some(now);
        self.previous_start = Some(now);
        Ok(true)
    }
    /// Record observed wall time for the admitted frame, even if geometry rejects it.
    ///
    /// # Errors
    /// Rejects completion without an active observation.
    pub fn complete(&mut self, elapsed: Duration) -> Result<SamplingStatus, VisualError> {
        if self.active.take().is_none() {
            return Err(VisualError::Invalid {
                field: "sampling completion without admission",
            });
        }
        let seconds = elapsed.as_secs_f64().max(0.000001);
        self.estimated_seconds = seconds.max(0.8 * self.estimated_seconds + 0.2 * seconds);
        Ok(self.status())
    }
    /// Return the current request, including a budget shortfall instead of hiding it.
    pub fn status(&self) -> SamplingStatus {
        let paused = self.config.utilization == 0.0;
        let seconds = self.estimated_seconds / self.config.utilization.max(f64::MIN_POSITIVE);
        let interval = duration(seconds).max(self.config.minimum_interval);
        SamplingStatus {
            estimated_cost: duration(self.estimated_seconds),
            utilization: self.config.utilization,
            interval,
            deadline_unattainable: paused || interval > self.config.maximum_interval,
            paused,
        }
    }
}
fn duration(seconds: f64) -> Duration {
    Duration::try_from_secs_f64(seconds).unwrap_or(Duration::MAX)
}
#[cfg(test)]
mod tests;
