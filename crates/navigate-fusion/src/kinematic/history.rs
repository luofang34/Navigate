//! Bounded delayed-measurement history and out-of-sequence replay
//! (ADR-0012).
//!
//! The history keeps the core before each admitted update and the
//! observation value of that update. A delayed observation rewinds to
//! the core before the first later update. The filter applies the
//! delayed observation at its acquisition time with every check. Then it
//! replays the later updates in time order without the checks. The
//! result is the same as when the observations arrive in time order.

use navigate_contract::{DurationNanos, MonotonicNanos};

use super::Core;
use super::update::{self, Mode};
use crate::config::FusionConfig;
use crate::history_ring::HistoryRing;
use crate::observation::ObservationValue;
use crate::rejection::RejectionReason;

/// Largest number of updates that the kinematic history keeps. A larger
/// `FusionConfig::history_capacity` keeps this number.
pub const MAX_KINEMATIC_HISTORY: usize = 64;

/// One admitted update: the core before it and the observation value.
#[derive(Debug, Clone, Copy)]
struct HistoryEntry {
    acquired_at: MonotonicNanos,
    before: Core,
    value: ObservationValue,
}

/// The retained updates of one filter epoch, oldest first, and the core
/// at initialization.
#[derive(Debug, Clone, Copy)]
pub(super) struct History {
    entries: HistoryRing<HistoryEntry, MAX_KINEMATIC_HISTORY>,
    anchor: Option<Core>,
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
    /// Forgets every entry and the initialization anchor.
    pub(super) fn clear(&mut self) {
        self.entries.clear();
        self.anchor = None;
    }

    /// Starts the history at the core that the first position fix made.
    pub(super) fn initialize(&mut self, core: Core) {
        self.entries.clear();
        self.anchor = Some(core);
    }

    /// Keeps an in-order update. The oldest entries go when the history
    /// holds more than `capacity` entries.
    pub(super) fn record(
        &mut self,
        before: Core,
        value: ObservationValue,
        acquired_at: MonotonicNanos,
        capacity: usize,
    ) {
        let entry = HistoryEntry {
            acquired_at,
            before,
            value,
        };
        self.entries.push_back(entry, capacity);
    }

    /// Number of retained entries.
    #[cfg(test)]
    pub(super) const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Applies a delayed observation at `acquired_at`, before the state
    /// time of `current`. On success, the result is the new core and the
    /// chi-square statistic of the delayed observation, and the history
    /// holds the replayed entries. On an error, the history does not
    /// change.
    ///
    /// The replay runs two times. The first pass checks that every
    /// update applies. The second pass writes the new core before each
    /// later entry. The replay is deterministic, so the two passes give
    /// the same cores, and the history needs no second buffer.
    pub(super) fn apply_delayed(
        &mut self,
        current: &Core,
        value: &ObservationValue,
        acquired_at: MonotonicNanos,
        config: &FusionConfig,
    ) -> Result<(Core, f64), RejectionReason> {
        let outside = || RejectionReason::OutsideHistory {
            age: current
                .state_at
                .elapsed_since(acquired_at)
                .unwrap_or(DurationNanos::from_nanos(0)),
            depth: self.depth(current, config),
        };
        let before_init = self.anchor.is_none_or(|a| acquired_at < a.state_at);
        if config.history_capacity == 0 || before_init {
            return Err(outside());
        }
        // Entries at the same time as the delayed observation stay before
        // it. Thus a delayed observation never moves ahead of an update
        // that the filter already applied at that time.
        let index = self
            .entries
            .partition_point(|e| e.acquired_at <= acquired_at);
        let Some(base) = self.entries.get(index).map(|e| e.before) else {
            return Err(outside());
        };
        if base.state_at > acquired_at {
            // The history evicted the updates before `acquired_at`.
            return Err(outside());
        }
        let mut scratch = base;
        let chi2 = update::apply_value(&mut scratch, value, acquired_at, config, Mode::Gated)?;
        for entry in self.entries.iter().skip(index) {
            update::apply_value(
                &mut scratch,
                &entry.value,
                entry.acquired_at,
                config,
                Mode::Replay,
            )?;
        }
        let mut replay = base;
        update::apply_value(&mut replay, value, acquired_at, config, Mode::Gated)?;
        for position in index..self.entries.len() {
            let Some(entry) = self.entries.get_mut(position) else {
                break;
            };
            entry.before = replay;
            let (event_value, at) = (entry.value, entry.acquired_at);
            update::apply_value(&mut replay, &event_value, at, config, Mode::Replay)?;
        }
        let delayed = HistoryEntry {
            acquired_at,
            before: base,
            value: *value,
        };
        self.entries.insert(index, delayed, config.history_capacity);
        Ok((scratch, chi2))
    }

    /// Time from the oldest rewind point to the state time of `current`.
    /// The oldest rewind point is the core before the oldest entry, or
    /// the anchor when no entry is retained. Zero when replay is
    /// disabled.
    fn depth(&self, current: &Core, config: &FusionConfig) -> DurationNanos {
        let zero = DurationNanos::from_nanos(0);
        if config.history_capacity == 0 {
            return zero;
        }
        let oldest = self
            .entries
            .front()
            .map(|e| e.before.state_at)
            .or_else(|| self.anchor.map(|a| a.state_at));
        oldest
            .and_then(|at| current.state_at.elapsed_since(at))
            .unwrap_or(zero)
    }
}
