//! Compact storage of the history: events without their large rare
//! payloads, and fixed pools for those payloads and for the checkpoint
//! cores (ADR-0016).
//!
//! The ring holds IMU events in place, 120 bytes each. An observation
//! is 240 bytes, and a regime update or a frame change carries a full
//! environment, about 0.8 KB. They are less frequent than IMU events, so
//! pools hold them and the entry holds a pool index. Only these pooled
//! events use the IMU rates of their last application, so the pool
//! keeps the rates with the event.

use super::super::estimate::InertialCore;
use super::super::imu::{ImuIncrement, ImuSample};
use super::super::mechanization::ImuRates;
use super::super::propagation::{ImpulsiveBurn, PropagationMode};
use super::super::update::Event;
use crate::observation::Observation;

/// A small event in place, or the pool index of a large one. An IMU
/// event propagates with its own rates, so it keeps no other rates. A
/// mode change, a burn, and a coast keep the rates of their last
/// application in place.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum StoredEvent {
    Imu(ImuSample),
    Increment(ImuIncrement),
    Mode(PropagationMode, Option<ImuRates>),
    Burn(ImpulsiveBurn, Option<ImuRates>),
    Coast(Option<ImuRates>),
    /// An observation, by its index in the observation pool.
    Observation(usize),
    /// A frame change or a regime update, by its index in the rare pool.
    Rare(usize),
}

/// A pooled event and the IMU rates of its last application.
#[derive(Debug, Clone, Copy)]
pub(super) struct Pooled<T: Copy> {
    pub(super) value: T,
    pub(super) covering: Option<ImuRates>,
}

/// A pool of `N` slots with the free-slot search of a fixed array.
#[derive(Debug, Clone, Copy)]
pub(super) struct Pool<T: Copy, const N: usize> {
    slots: [Option<T>; N],
}

impl<T: Copy, const N: usize> Pool<T, N> {
    pub(super) const fn new() -> Self {
        Self {
            slots: [const { None }; N],
        }
    }

    pub(super) fn clear(&mut self) {
        self.slots = [const { None }; N];
    }

    /// Puts `value` in a free slot. `None` when every slot is in use.
    pub(super) fn put(&mut self, value: T) -> Option<usize> {
        let index = self.slots.iter().position(Option::is_none)?;
        let slot = self.slots.get_mut(index)?;
        *slot = Some(value);
        Some(index)
    }

    pub(super) fn get(&self, index: usize) -> Option<&T> {
        self.slots.get(index)?.as_ref()
    }

    pub(super) fn set(&mut self, index: usize, value: T) {
        if let Some(slot) = self.slots.get_mut(index) {
            *slot = Some(value);
        }
    }

    pub(super) fn free(&mut self, index: usize) {
        if let Some(slot) = self.slots.get_mut(index) {
            *slot = None;
        }
    }

    pub(super) fn is_full(&self) -> bool {
        self.slots.iter().all(Option::is_some)
    }

    fn free_slots(&self) -> usize {
        self.slots.iter().filter(|slot| slot.is_none()).count()
    }
}

/// The pools of the events that the ring does not hold in place.
#[derive(Debug, Clone, Copy)]
pub(super) struct EventPools {
    observations: Pool<Pooled<Observation>, MAX_OBSERVATIONS>,
    rare: Pool<Pooled<Event>, MAX_RARE>,
}

impl EventPools {
    pub(super) const fn new() -> Self {
        Self {
            observations: Pool::new(),
            rare: Pool::new(),
        }
    }

    pub(super) fn clear(&mut self) {
        self.observations.clear();
        self.rare.clear();
    }

    /// Whether the pool that `event` needs has too few free slots for
    /// an in-order event. The observation pool keeps one slot free for a
    /// delayed observation.
    pub(super) fn is_full_for(&self, event: &Event) -> bool {
        match event {
            Event::Observation(_) => self.observations.free_slots() < 2,
            Event::FrameChange(_) | Event::Regime(_) => self.rare.is_full(),
            _ => false,
        }
    }

    /// Stores `event` with `covering`. `None` when its pool is full.
    pub(super) fn store(
        &mut self,
        event: &Event,
        covering: Option<ImuRates>,
    ) -> Option<StoredEvent> {
        Some(match event {
            Event::Imu(sample) => StoredEvent::Imu(*sample),
            Event::Increment(increment) => StoredEvent::Increment(*increment),
            Event::Mode(mode) => StoredEvent::Mode(*mode, covering),
            Event::Burn(burn) => StoredEvent::Burn(*burn, covering),
            Event::Coast => StoredEvent::Coast(covering),
            Event::Observation(obs) => StoredEvent::Observation(self.observations.put(Pooled {
                value: *obs,
                covering,
            })?),
            Event::FrameChange(_) | Event::Regime(_) => {
                StoredEvent::Rare(self.rare.put(Pooled {
                    value: *event,
                    covering,
                })?)
            }
        })
    }

    /// The event of a stored form and the rates of its last
    /// application. `None` for an index without a value, which the
    /// history never makes.
    pub(super) fn load(&self, stored: &StoredEvent) -> Option<(Event, Option<ImuRates>)> {
        Some(match stored {
            StoredEvent::Imu(sample) => (Event::Imu(*sample), None),
            StoredEvent::Increment(increment) => (Event::Increment(*increment), None),
            StoredEvent::Mode(mode, covering) => (Event::Mode(*mode), *covering),
            StoredEvent::Burn(burn, covering) => (Event::Burn(*burn), *covering),
            StoredEvent::Coast(covering) => (Event::Coast, *covering),
            StoredEvent::Observation(index) => {
                let pooled = self.observations.get(*index)?;
                (Event::Observation(pooled.value), pooled.covering)
            }
            StoredEvent::Rare(index) => {
                let pooled = self.rare.get(*index)?;
                (pooled.value, pooled.covering)
            }
        })
    }

    /// Writes the rates of the last application of an event.
    pub(super) fn set_covering(&mut self, stored: &mut StoredEvent, covering: Option<ImuRates>) {
        match stored {
            StoredEvent::Mode(_, rates)
            | StoredEvent::Burn(_, rates)
            | StoredEvent::Coast(rates) => {
                *rates = covering;
            }
            StoredEvent::Observation(index) => {
                if let Some(mut pooled) = self.observations.get(*index).copied() {
                    pooled.covering = covering;
                    self.observations.set(*index, pooled);
                }
            }
            StoredEvent::Rare(index) => {
                if let Some(mut pooled) = self.rare.get(*index).copied() {
                    pooled.covering = covering;
                    self.rare.set(*index, pooled);
                }
            }
            StoredEvent::Imu(_) | StoredEvent::Increment(_) => {}
        }
    }

    /// Frees the pool slot of `stored`.
    pub(super) fn free(&mut self, stored: &StoredEvent) {
        match stored {
            StoredEvent::Observation(index) => self.observations.free(*index),
            StoredEvent::Rare(index) => self.rare.free(*index),
            _ => {}
        }
    }
}

/// Slots of the observation pool. More observations in the history
/// shorten it instead.
const MAX_OBSERVATIONS: usize = 160;

/// Slots of the rare pool: frame changes and regime updates. More of
/// them shorten the history instead.
const MAX_RARE: usize = 4;

/// The checkpoint cores of the history.
pub(super) type CorePool<const N: usize, const SLOTS: usize> = Pool<InertialCore<N>, SLOTS>;
