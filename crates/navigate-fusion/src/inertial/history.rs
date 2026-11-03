//! Bounded event history and out-of-sequence replay (ADR-0012,
//! ADR-0014).
//!
//! The history keeps the core before each event and the event. The
//! events are IMU samples, observations, frame changes, and environment
//! updates. The rules of time are these:
//!
//! - An IMU sample at or after the state time propagates the state to its
//!   end time with its own rates.
//! - An observation at or after the state time propagates the state with
//!   the most recent IMU rates.
//! - An IMU sample or an observation before the state time rewinds to
//!   the core before the first later event. The propagation to the
//!   acquisition time uses the rates of the first later IMU sample when
//!   its interval holds the acquisition time. Then the replay of that
//!   sample propagates the rest of its interval with the same rates, so
//!   the split of the interval is exact. The later events replay in time
//!   order without the gates.
//! - A frame change and an environment update are events at or after the
//!   state time. A replay that starts before them applies them again.

use std::collections::VecDeque;

use navigate_contract::{DurationNanos, MonotonicNanos};

use super::config::InertialConfig;
use super::estimate::{InertialCore, Mode};
use super::mechanization::ImuRates;
use super::update::{Event, apply_event};
use crate::rejection::RejectionReason;

/// One event and the core before it.
#[derive(Debug, Clone)]
struct HistoryEntry {
    at: MonotonicNanos,
    before: InertialCore,
    event: Event,
}

/// The retained events of one filter epoch, oldest first, and the
/// initialization time.
#[derive(Debug, Clone, Default)]
pub(crate) struct History {
    entries: VecDeque<HistoryEntry>,
    anchor: Option<MonotonicNanos>,
}

impl History {
    /// Forgets every entry and the initialization time.
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.anchor = None;
    }

    /// Starts the history at the initialization time.
    pub(crate) fn initialize(&mut self, at: MonotonicNanos) {
        self.entries.clear();
        self.anchor = Some(at);
    }

    /// Keeps an in-order event. The oldest entries go when the history
    /// holds more than `capacity` entries.
    pub(crate) fn record(
        &mut self,
        before: InertialCore,
        event: Event,
        at: MonotonicNanos,
        capacity: usize,
    ) {
        self.entries.push_back(HistoryEntry { at, before, event });
        while self.entries.len() > capacity {
            self.entries.pop_front();
        }
    }

    /// Applies `event` at `at`, before the state time of `current`. On
    /// success the result is the new core and the chi-square statistic of
    /// an observation, and the history holds the replayed entries. On an
    /// error the history does not change.
    pub(crate) fn apply_delayed(
        &mut self,
        current: &InertialCore,
        event: &Event,
        at: MonotonicNanos,
        config: &InertialConfig,
    ) -> Result<(InertialCore, Option<f64>), RejectionReason> {
        let capacity = config.fusion.history_capacity;
        let outside = || RejectionReason::OutsideHistory {
            age: current
                .state_at
                .elapsed_since(at)
                .unwrap_or(DurationNanos::from_nanos(0)),
            depth: self.depth(current, capacity),
        };
        if capacity == 0 || self.anchor.is_none_or(|anchor| at < anchor) {
            return Err(outside());
        }
        // Entries at the same time as the delayed event stay before it.
        let index = self.entries.partition_point(|e| e.at <= at);
        let Some(base) = self.entries.get(index).map(|e| e.before.clone()) else {
            return Err(outside());
        };
        if base.state_at > at {
            return Err(outside());
        }
        let mut scratch = base.clone();
        let covering = self.covering_rates(index, at);
        let chi2 = apply_event(&mut scratch, event, at, config, Mode::Gated, covering)?;
        let mut rebuilt: VecDeque<HistoryEntry> =
            self.entries.iter().take(index).cloned().collect();
        rebuilt.push_back(HistoryEntry {
            at,
            before: base,
            event: event.clone(),
        });
        for (offset, entry) in self.entries.iter().enumerate().skip(index) {
            let before = scratch.clone();
            let covering = self.covering_rates(offset.wrapping_add(1), entry.at);
            apply_event(
                &mut scratch,
                &entry.event,
                entry.at,
                config,
                Mode::Replay,
                covering,
            )?;
            rebuilt.push_back(HistoryEntry {
                before,
                ..entry.clone()
            });
        }
        while rebuilt.len() > capacity {
            rebuilt.pop_front();
        }
        self.entries = rebuilt;
        Ok((scratch, chi2))
    }

    /// The rates of the first IMU sample at or after `index` when its
    /// interval holds `at`.
    fn covering_rates(&self, index: usize, at: MonotonicNanos) -> Option<ImuRates> {
        let sample = self
            .entries
            .iter()
            .skip(index)
            .find_map(|entry| match &entry.event {
                Event::Imu(sample) => Some(*sample),
                _ => None,
            })?;
        let interval_ns = (sample.dt_s * 1e9).round() as u64;
        let start = sample
            .stamp
            .acquired_at
            .as_nanos()
            .saturating_sub(interval_ns);
        (start <= at.as_nanos()).then(|| ImuRates::of(&sample))
    }

    /// Time from the oldest rewind point to the state time of `current`.
    /// Zero when replay is disabled.
    fn depth(&self, current: &InertialCore, capacity: usize) -> DurationNanos {
        let zero = DurationNanos::from_nanos(0);
        if capacity == 0 {
            return zero;
        }
        let oldest = self
            .entries
            .front()
            .map(|e| e.before.state_at)
            .or(self.anchor);
        oldest
            .and_then(|at| current.state_at.elapsed_since(at))
            .unwrap_or(zero)
    }
}
