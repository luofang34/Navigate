//! Bounded event history with checkpoints, and out-of-sequence replay
//! (ADR-0012, ADR-0014, ADR-0017).
//!
//! The history keeps the events in time order: IMU samples, IMU
//! increments, observations, frame changes, regime updates, mode
//! changes, burns, and coasts. It keeps the full core only at a
//! checkpoint, one for every [`CHECKPOINT_STRIDE`] in-order events. The
//! core before any other entry comes from the replay of the entries
//! after the checkpoint before it. Each entry keeps the IMU rates that
//! its last application used, so this replay gives the same core as the
//! application, bit for bit. The oldest entry is always a checkpoint, so
//! the history can give a core at any time from the oldest checkpoint to
//! the state time.
//!
//! The rules of time are these:
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

mod store;

use navigate_contract::{DurationNanos, MonotonicNanos};

use super::config::InertialConfig;
use super::estimate::{InertialCore, Mode};
use super::mechanization::ImuRates;
use super::update::{Event, apply_event};
use crate::float_math::round;
use crate::history_ring::HistoryRing;
use crate::rejection::RejectionReason;
use store::{CorePool, EventPools, StoredEvent};

/// Largest number of events that the inertial history keeps. A larger
/// `FusionConfig::history_capacity` keeps this number. At 500 Hz IMU
/// data and 100 Hz aiding the history spans more than one second.
pub const MAX_INERTIAL_HISTORY: usize = 704;

/// In-order events between two checkpoints. A query or a delayed event
/// replays at most this number of events before its time.
pub(crate) const CHECKPOINT_STRIDE: usize = 64;

/// Slots of the checkpoint pool: one for each stride of the history and
/// one for the stride that a delayed insertion opens.
const MAX_CHECKPOINTS: usize = MAX_INERTIAL_HISTORY / CHECKPOINT_STRIDE + 2;

/// One event, and the pool slot of the core before it when it is a
/// checkpoint.
#[derive(Debug, Clone, Copy)]
struct HistoryEntry {
    at: MonotonicNanos,
    event: StoredEvent,
    checkpoint: Option<u16>,
}

/// The retained events of one filter epoch, oldest first, their
/// checkpoint cores, and the initialization time.
#[derive(Debug, Clone, Copy)]
pub(crate) struct History {
    entries: HistoryRing<HistoryEntry, { MAX_INERTIAL_HISTORY + 1 }>,
    cores: CorePool<MAX_CHECKPOINTS>,
    pools: EventPools,
    anchor: Option<MonotonicNanos>,
}

impl Default for History {
    fn default() -> Self {
        Self {
            entries: HistoryRing::new(),
            cores: CorePool::new(),
            pools: EventPools::new(),
            anchor: None,
        }
    }
}

impl History {
    /// Forgets every entry and the initialization time.
    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.cores.clear();
        self.pools.clear();
        self.anchor = None;
    }

    /// Starts the history at the initialization time.
    pub(crate) fn initialize(&mut self, at: MonotonicNanos) {
        self.clear();
        self.anchor = Some(at);
    }

    /// Keeps an in-order event that `before` was the core before. The
    /// oldest checkpoint and the entries after it go when the history
    /// holds `capacity` entries, or when a pool has no free slot.
    pub(crate) fn record(
        &mut self,
        before: InertialCore,
        event: Event,
        at: MonotonicNanos,
        capacity: usize,
    ) {
        if capacity == 0 {
            return;
        }
        let checkpoint_due = self.events_since_checkpoint() >= CHECKPOINT_STRIDE;
        while !self.entries.is_empty()
            && (self.entries.len() >= capacity
                || self.pools.is_full_for(&event)
                || (checkpoint_due && self.cores.is_full()))
        {
            self.evict_front();
        }
        // An empty history starts with a checkpoint.
        let Some(event) = self.pools.store(&event, None) else {
            return;
        };
        let checkpoint = if checkpoint_due || self.entries.is_empty() {
            self.cores
                .put(before)
                .and_then(|slot| u16::try_from(slot).ok())
        } else {
            None
        };
        let entry = HistoryEntry {
            at,
            event,
            checkpoint,
        };
        self.entries.push_back(entry, usize::MAX);
    }

    /// In-order events since the newest checkpoint, or
    /// [`CHECKPOINT_STRIDE`] when the history has none.
    fn events_since_checkpoint(&self) -> usize {
        let len = self.entries.len();
        (0..len)
            .rev()
            .position(|index| {
                self.entries
                    .get(index)
                    .is_some_and(|e| e.checkpoint.is_some())
            })
            .map_or(CHECKPOINT_STRIDE, |back| back.wrapping_add(1))
    }

    /// Removes the oldest checkpoint and the entries up to the next one,
    /// so the oldest entry is a checkpoint again.
    fn evict_front(&mut self) {
        let mut first = true;
        while let Some(entry) = self.entries.front().copied() {
            if !first && entry.checkpoint.is_some() {
                break;
            }
            first = false;
            if let Some(slot) = entry.checkpoint {
                self.cores.free(usize::from(slot));
            }
            self.pools.free(&entry.event);
            self.entries.pop_front();
        }
    }

    /// The core before the entry at `position`: the checkpoint at or
    /// before it, and the replay of the entries between. `None` when the
    /// history has no checkpoint at or before `position`.
    fn core_before(
        &self,
        position: usize,
        config: &InertialConfig,
    ) -> Option<Result<InertialCore, RejectionReason>> {
        let (start, slot) = (0..=position).rev().find_map(|index| {
            let slot = self.entries.get(index)?.checkpoint?;
            Some((index, slot))
        })?;
        let mut core = *self.cores.get(usize::from(slot))?;
        for index in start..position {
            let entry = self.entries.get(index)?;
            let (event, covering) = self.pools.load(&entry.event)?;
            if let Err(reason) =
                apply_event(&mut core, &event, entry.at, config, Mode::Replay, covering)
            {
                return Some(Err(reason));
            }
        }
        Some(Ok(core))
    }

    /// The core that the replay of a delayed observation at `at` starts
    /// from, and the rates of the IMU event whose interval holds `at`:
    /// the core before the first entry later than `at`, or `current`
    /// when no entry is later. The error is the oldest time the history
    /// can give a core for.
    pub(crate) fn base_at(
        &self,
        current: &InertialCore,
        at: MonotonicNanos,
        config: &InertialConfig,
    ) -> Result<(InertialCore, Option<ImuRates>), MonotonicNanos> {
        let capacity = config.fusion.history_capacity;
        let oldest = self.oldest(current, capacity);
        if capacity == 0 || at < oldest {
            return Err(oldest);
        }
        let index = self.entries.partition_point(|e| e.at <= at);
        let base = if index >= self.entries.len() {
            *current
        } else {
            match self.core_before(index, config) {
                Some(Ok(core)) => core,
                _ => return Err(oldest),
            }
        };
        if base.state_at > at {
            return Err(oldest);
        }
        Ok((base, self.covering_rates(index, at)))
    }

    /// The oldest time the history can give a core for: the state time
    /// of the oldest checkpoint, the initialization time without
    /// entries, and the state time of `current` when replay is disabled.
    pub(crate) fn oldest(&self, current: &InertialCore, capacity: usize) -> MonotonicNanos {
        if capacity == 0 {
            return current.state_at;
        }
        self.entries
            .front()
            .and_then(|entry| self.cores.get(usize::from(entry.checkpoint?)))
            .map(|core| core.state_at)
            .or(self.anchor)
            .unwrap_or(current.state_at)
    }

    /// Applies `event` at `at`, before the state time of `current`. On
    /// success the result is the new core and the chi-square statistic of
    /// an observation, and the history holds the replayed entries. On an
    /// error the history does not change.
    ///
    /// The replay runs two times. The first pass checks that every event
    /// applies. The second pass writes the new checkpoint cores and the
    /// rates of each replayed entry. The replay is deterministic, so the
    /// two passes give the same cores.
    pub(crate) fn apply_delayed(
        &mut self,
        current: &InertialCore,
        event: &Event,
        at: MonotonicNanos,
        config: &InertialConfig,
    ) -> Result<(InertialCore, Option<f64>), RejectionReason> {
        let capacity = config.fusion.history_capacity;
        let oldest = self.oldest(current, capacity);
        let outside = || RejectionReason::OutsideHistory {
            age: current
                .state_at
                .elapsed_since(at)
                .unwrap_or(DurationNanos::from_nanos(0)),
            depth: current
                .state_at
                .elapsed_since(oldest)
                .unwrap_or(DurationNanos::from_nanos(0)),
        };
        let delayable = matches!(
            event,
            Event::Imu(_) | Event::Increment(_) | Event::Observation(_)
        );
        if capacity == 0 || self.anchor.is_none_or(|anchor| at < anchor) || !delayable {
            return Err(outside());
        }
        // Entries at the same time as the delayed event stay before it.
        let index = self.entries.partition_point(|e| e.at <= at);
        if index >= self.entries.len() {
            return Err(outside());
        }
        let base = self.core_before(index, config).ok_or_else(outside)??;
        if base.state_at > at {
            return Err(outside());
        }
        let covering = self.covering_rates(index, at);
        let mut scratch = base;
        let chi2 = apply_event(&mut scratch, event, at, config, Mode::Gated, covering)?;
        for (offset, entry) in self.entries.iter().enumerate().skip(index) {
            let rates = self.covering_rates(offset.wrapping_add(1), entry.at);
            let (later, _) = self.pools.load(&entry.event).ok_or_else(outside)?;
            apply_event(&mut scratch, &later, entry.at, config, Mode::Replay, rates)?;
        }
        let stored = self.pools.store(event, covering).ok_or_else(outside)?;
        // The core before the old entry at `index` is `base`, so its
        // checkpoint moves to the delayed entry unchanged.
        let checkpoint = self
            .entries
            .get_mut(index)
            .and_then(|entry| entry.checkpoint.take());
        let delayed = HistoryEntry {
            at,
            event: stored,
            checkpoint,
        };
        self.entries.insert(index, delayed, usize::MAX);
        self.rewrite(base, index, config)?;
        while self.entries.len() > capacity {
            self.evict_front();
        }
        Ok((scratch, chi2))
    }

    /// Replays the entries from `index` with `core` as the core before
    /// the first of them, and writes the checkpoint cores and the rates of
    /// each entry.
    fn rewrite(
        &mut self,
        mut core: InertialCore,
        index: usize,
        config: &InertialConfig,
    ) -> Result<(), RejectionReason> {
        for position in index..self.entries.len() {
            let Some(entry) = self.entries.get(position).copied() else {
                break;
            };
            let Some((event, stored_covering)) = self.pools.load(&entry.event) else {
                break;
            };
            let covering = if position == index {
                stored_covering
            } else {
                self.covering_rates(position.wrapping_add(1), entry.at)
            };
            if let Some(slot) = entry.checkpoint {
                self.cores.set(usize::from(slot), core);
            }
            if let Some(stored) = self.entries.get_mut(position) {
                self.pools.set_covering(&mut stored.event, covering);
            }
            apply_event(&mut core, &event, entry.at, config, Mode::Replay, covering)?;
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
                StoredEvent::Imu(sample) => {
                    Some((sample.stamp.acquired_at, sample.dt_s, ImuRates::of(sample)))
                }
                StoredEvent::Increment(increment) => Some((
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
}
