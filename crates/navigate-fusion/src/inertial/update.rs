//! One event applied to an initialized core (ADR-0014, ADR-0017).
//!
//! The in-order path, the delayed path, and the replay of the history
//! use this module. The frame gate compares the declared frame with the
//! frame of the core that the event updates. Thus a delayed observation
//! is judged against the frame that was active at its acquisition time.

use navigate_contract::MonotonicNanos;

use super::config::InertialConfig;
use super::estimate::{InertialCore, Mode};
use super::frame_change::{FrameChange, apply_frame_change};
use super::imu::{ImuIncrement, ImuSample};
use super::mechanization::ImuRates;
use super::models::apply_observation;
use super::regime::Regime;
use crate::admission::check_frame;
use crate::observation::Observation;
use crate::rejection::RejectionReason;

/// One event of the history.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(
    clippy::large_enum_variant,
    reason = "each event is inline so that the history needs no heap"
)]
pub(crate) enum Event {
    /// An IMU sample that propagates the state to its end time.
    Imu(ImuSample),
    /// An IMU increment that propagates the state to its end time.
    Increment(ImuIncrement),
    /// An aiding observation.
    Observation(Observation),
    /// A change of the navigation frame.
    FrameChange(FrameChange),
    /// A new environment or reference ellipsoid in the same frame.
    Regime(Regime),
}

/// Applies `event` at `at`. `covering` gives the rates of the IMU sample
/// or increment whose interval holds `at`, when the history knows it.
/// Without it the propagation holds the most recent rates. The result is
/// the chi-square statistic of an observation. On an error the caller
/// drops `core`.
pub(crate) fn apply_event(
    core: &mut InertialCore,
    event: &Event,
    at: MonotonicNanos,
    config: &InertialConfig,
    mode: Mode,
    covering: Option<ImuRates>,
) -> Result<Option<f64>, RejectionReason> {
    let explicit = config.fusion.require_explicit_frame || core.regime.requires_explicit_frame();
    match event {
        Event::Imu(sample) => {
            check_frame(core.regime.frame(), sample.frame, explicit)?;
            let rates = ImuRates::of(sample);
            core.propagate_to(at, &rates, config)?;
            core.last_rates = Some(rates);
            Ok(None)
        }
        Event::Increment(increment) => {
            check_frame(core.regime.frame(), increment.frame, explicit)?;
            core.propagate_increment(at, increment, config)?;
            core.last_rates = Some(ImuRates::mean_of(increment));
            Ok(None)
        }
        Event::Observation(obs) => {
            check_frame(core.regime.frame(), obs.frame, explicit)?;
            core.propagate_with(at, covering, config)?;
            apply_observation(core, obs, config, mode).map(Some)
        }
        Event::FrameChange(change) => {
            core.propagate_with(at, covering, config)?;
            apply_frame_change(core, change);
            Ok(None)
        }
        Event::Regime(regime) => {
            core.regime = *regime;
            Ok(None)
        }
    }
}
