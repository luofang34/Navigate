//! Bounded event history and out-of-sequence replay (ADR-0012,
//! ADR-0014, ADR-0017).
//!
//! The history keeps the core before each event and the event. The
//! events are IMU samples, IMU increments, observations, frame changes,
//! and regime updates. The rules of time are these:
//!
//! - An IMU sample or an IMU increment at or after the state time
//!   propagates the state to its end time.
//! - An observation at or after the state time propagates the state with
//!   the most recent IMU rates.
//! - An IMU event or an observation before the state time rewinds to the
//!   core before the first later event. The propagation to the
//!   acquisition time uses the rates of the first later IMU event when
//!   its interval holds the acquisition time. For an increment these
//!   rates are its mean rates. Then the replay of that event propagates
//!   the rest of its interval with the same rates, so the split of the
//!   interval is exact. The later events replay in time order without
//!   the gates.
//! - A frame change and a regime update are events at or after the state
//!   time. A replay that starts before them applies them again.

use navigate_contract::{DurationNanos, MonotonicNanos};

use super::config::InertialConfig;
use super::estimate::{InertialCore, Mode};
use super::mechanization::ImuRates;
use super::update::{Event, apply_event};
use crate::float_math::round;
use crate::history_ring::HistoryRing;
use crate::rejection::RejectionReason;

/// Largest number of events that the inertial history keeps. A larger
/// `FusionConfig::history_capacity` keeps this number.
pub const MAX_INERTIAL_HISTORY: usize = 32;

/// One event and the core before it.
#[derive(Debug, Clone, Copy)]
struct HistoryEntry {
    at: MonotonicNanos,
    before: InertialCore,
    event: Event,
}

/// The retained events of one filter epoch, oldest first, and the
/// initialization time.
#[derive(Debug, Clone, Copy)]
pub(crate) struct History {
    entries: HistoryRing<HistoryEntry, MAX_INERTIAL_HISTORY>,
    anchor: Option<MonotonicNanos>,
}

impl Default for History {
    fn default() -> Self {
        Self {
            entries: HistoryRing::new(),
            anchor: None,
        }
    }
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
        self.entries
            .push_back(HistoryEntry { at, before, event }, capacity);
    }

    /// Applies `event` at `at`, before the state time of `current`. On
    /// success the result is the new core and the chi-square statistic of
    /// an observation, and the history holds the replayed entries. On an
    /// error the history does not change.
    ///
    /// The replay runs two times. The first pass checks that every event
    /// applies. The second pass writes the new core before each later
    /// entry. The replay is deterministic, so the two passes give the
    /// same cores, and the history needs no second buffer.
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
        let Some(base) = self.entries.get(index).map(|e| e.before) else {
            return Err(outside());
        };
        if base.state_at > at {
            return Err(outside());
        }
        let (scratch, chi2) = self.replay(base, event, at, index, config)?;
        let (mut rewrite, covering) = (base, self.covering_rates(index, at));
        apply_event(&mut rewrite, event, at, config, Mode::Gated, covering)?;
        self.replay_rewrite(rewrite, index, config)?;
        let delayed = HistoryEntry {
            at,
            before: base,
            event: *event,
        };
        self.entries.insert(index, delayed, capacity);
        Ok((scratch, chi2))
    }

    /// Applies `event` at `at` to `base`, then replays the entries from
    /// `index` without a change to the history.
    fn replay(
        &self,
        base: InertialCore,
        event: &Event,
        at: MonotonicNanos,
        index: usize,
        config: &InertialConfig,
    ) -> Result<(InertialCore, Option<f64>), RejectionReason> {
        let mut scratch = base;
        let covering = self.covering_rates(index, at);
        let chi2 = apply_event(&mut scratch, event, at, config, Mode::Gated, covering)?;
        for (offset, entry) in self.entries.iter().enumerate().skip(index) {
            let covering = self.covering_rates(offset.wrapping_add(1), entry.at);
            apply_event(
                &mut scratch,
                &entry.event,
                entry.at,
                config,
                Mode::Replay,
                covering,
            )?;
        }
        Ok((scratch, chi2))
    }

    /// Writes `core` as the core before the entry at `index`, replays
    /// that entry, and goes on to the newest entry.
    fn replay_rewrite(
        &mut self,
        mut core: InertialCore,
        index: usize,
        config: &InertialConfig,
    ) -> Result<(), RejectionReason> {
        for position in index..self.entries.len() {
            let covering = self
                .entries
                .get(position)
                .and_then(|entry| self.covering_rates(position.wrapping_add(1), entry.at));
            let Some(entry) = self.entries.get_mut(position) else {
                break;
            };
            entry.before = core;
            let (event, at) = (entry.event, entry.at);
            apply_event(&mut core, &event, at, config, Mode::Replay, covering)?;
        }
        Ok(())
    }

    /// The rates of the first IMU event at or after `index` when its
    /// interval holds `at`.
    fn covering_rates(&self, index: usize, at: MonotonicNanos) -> Option<ImuRates> {
        let (end, dt_s, rates) = self
            .entries
            .iter()
            .skip(index)
            .find_map(|entry| match &entry.event {
                Event::Imu(sample) => {
                    Some((sample.stamp.acquired_at, sample.dt_s, ImuRates::of(sample)))
                }
                Event::Increment(increment) => Some((
                    increment.stamp.acquired_at,
                    increment.dt_s,
                    ImuRates::mean_of(increment),
                )),
                _ => None,
            })?;
        let interval_ns = round(dt_s * 1e9) as u64;
        let start = end.as_nanos().saturating_sub(interval_ns);
        (start <= at.as_nanos()).then_some(rates)
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
