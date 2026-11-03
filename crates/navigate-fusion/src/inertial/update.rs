//! One event applied to an initialized core (ADR-0014).
//!
//! The in-order path, the delayed path, and the replay of the history
//! use this module. The frame gate compares the declared frame with the
//! frame of the core that the event updates. Thus a delayed observation
//! is judged against the frame that was active at its acquisition time.

use std::sync::Arc;

use nalgebra::{UnitQuaternion, Vector3};
use navigate_contract::MonotonicNanos;

use super::config::InertialConfig;
use super::estimate::{InertialCore, Mode};
use super::imu::ImuSample;
use super::mechanization::ImuRates;
use super::models::apply_observation;
use super::nominal::{Mat17, POS, VEL, renormalized};
use super::regime::Regime;
use crate::admission::check_frame;
use crate::environment::SpaceEnvironment;
use crate::observation::Observation;
use crate::rejection::RejectionReason;

/// A change of the space frame at one time: the target environment, the
/// position and the velocity of its center in the active frame, and the
/// rotation from the active axes to the target axes.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Recenter {
    pub(crate) env: Arc<SpaceEnvironment>,
    pub(crate) origin_shift: Vector3<f64>,
    pub(crate) velocity_shift: Vector3<f64>,
    pub(crate) rotation: UnitQuaternion<f64>,
}

/// One event of the history.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Event {
    /// An IMU sample that propagates the state to its end time.
    Imu(ImuSample),
    /// An aiding observation.
    Observation(Observation),
    /// A change of the space frame.
    Recenter(Recenter),
    /// New third-body positions in the same space frame.
    Environment(Arc<SpaceEnvironment>),
}

/// Applies `event` at `at`. `covering` gives the rates of the IMU sample
/// whose interval holds `at`, when the history knows it. Without it the
/// propagation holds the most recent rates. The result is the
/// chi-square statistic of an observation. On an error the caller drops
/// `core`.
pub(crate) fn apply_event(
    core: &mut InertialCore,
    event: &Event,
    at: MonotonicNanos,
    config: &InertialConfig,
    mode: Mode,
    covering: Option<ImuRates>,
) -> Result<Option<f64>, RejectionReason> {
    let explicit = config.fusion.require_explicit_frame;
    match event {
        Event::Imu(sample) => {
            check_frame(core.regime.frame(), sample.frame, explicit)?;
            let rates = ImuRates::of(sample);
            core.propagate_to(at, &rates, config)?;
            core.last_rates = Some(rates);
            Ok(None)
        }
        Event::Observation(obs) => {
            check_frame(core.regime.frame(), obs.frame, explicit)?;
            core.propagate_with(at, covering, config)?;
            apply_observation(core, obs, config, mode).map(Some)
        }
        Event::Recenter(recenter) => {
            core.propagate_with(at, covering, config)?;
            apply_recenter(core, recenter);
            Ok(None)
        }
        Event::Environment(env) => {
            core.regime = Regime::CartesianSpace(Arc::clone(env));
            Ok(None)
        }
    }
}

/// Moves the state into the new frame: `p' = R·(p − s_p)`,
/// `v' = R·(v − s_v)`, and `q' = q_R ⊗ q`. The position and velocity
/// blocks of the covariance rotate with `R`, with their cross terms. The
/// attitude error is in the body axes and the bias errors are in the
/// body or sensor axes, so their blocks do not change.
fn apply_recenter(core: &mut InertialCore, recenter: &Recenter) {
    let rotation = recenter.rotation.to_rotation_matrix().into_inner();
    let nominal = &mut core.nominal;
    nominal.p = rotation * (nominal.p - recenter.origin_shift);
    nominal.v = rotation * (nominal.v - recenter.velocity_shift);
    nominal.q = renormalized(recenter.rotation * nominal.q);
    let mut j = Mat17::identity();
    j.fixed_view_mut::<3, 3>(POS, POS).copy_from(&rotation);
    j.fixed_view_mut::<3, 3>(VEL, VEL).copy_from(&rotation);
    let rotated = j * core.p * j.transpose();
    core.p = (rotated + rotated.transpose()) * 0.5;
    core.regime = Regime::CartesianSpace(Arc::clone(&recenter.env));
}
