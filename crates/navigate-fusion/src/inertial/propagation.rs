//! The propagation modes, the impulsive burn, and the coast of the
//! inertial filter (ADR-0018).
//!
//! [`PropagationMode::SensedImu`] propagates with IMU samples and IMU
//! increments, for a powered ascent, a burn, an entry, an atmospheric
//! flight, or a powered descent. [`PropagationMode::ForceModel`]
//! propagates with the orbital force model of the environment, for a
//! coast without IMU data. A change of mode, a burn, and a coast are
//! history events, so a delayed observation replays through them.

use navigate_contract::{MonotonicNanos, SymmetricCov3};

use super::InertialNavigationFilter;
use super::estimate::InertialCore;
use super::nominal::VEL;
use super::update::Event;
use crate::admission::check_cov3;
use crate::kalman::cov3_to_matrix;
use crate::rejection::RejectionReason;

/// Default longest Runge–Kutta step of a coast in seconds.
const DEFAULT_MAX_RK4_STEP_S: f64 = 10.0;

/// The settings of a force-model coast.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForceModelCoast {
    /// Longest Runge–Kutta step in seconds. A longer interval is split
    /// into equal steps.
    pub max_rk4_step_s: f64,
    /// Constant inertial angular rate of the body in rad/s, in the body
    /// axes. Zero holds the attitude fixed in inertial space.
    pub body_rate_rad_s: [f64; 3],
    /// White noise density of the forces that the model does not hold,
    /// in (m/s²)/√Hz. It adds to the empirical acceleration states.
    pub acceleration_noise_sigma_mps2_sqrt_hz: f64,
    /// White noise density of the body rate in (rad/s)/√Hz.
    pub body_rate_noise_sigma_rad_s_sqrt_hz: f64,
}

impl ForceModelCoast {
    /// A coast with the longest step `max_rk4_step_s`, an attitude fixed
    /// in inertial space, and no white noise.
    #[must_use]
    pub const fn new(max_rk4_step_s: f64) -> Self {
        Self {
            max_rk4_step_s,
            body_rate_rad_s: [0.0; 3],
            acceleration_noise_sigma_mps2_sqrt_hz: 0.0,
            body_rate_noise_sigma_rad_s_sqrt_hz: 0.0,
        }
    }

    /// Whether the step is finite and positive, the rate is finite, and
    /// each noise density is finite and not negative.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        let density = |value: f64| value.is_finite() && value >= 0.0;
        self.max_rk4_step_s.is_finite()
            && self.max_rk4_step_s > 0.0
            && self.body_rate_rad_s.iter().all(|w| w.is_finite())
            && density(self.acceleration_noise_sigma_mps2_sqrt_hz)
            && density(self.body_rate_noise_sigma_rad_s_sqrt_hz)
    }
}

impl Default for ForceModelCoast {
    /// A coast with ten-second steps.
    fn default() -> Self {
        Self::new(DEFAULT_MAX_RK4_STEP_S)
    }
}

/// How the filter propagates the nominal state and the covariance.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum PropagationMode {
    /// IMU samples and IMU increments drive the state.
    #[default]
    SensedImu,
    /// The orbital force model of the environment and the empirical
    /// acceleration drive the state. The filter refuses IMU events.
    ForceModel(ForceModelCoast),
}

impl PropagationMode {
    /// Whether the settings of the mode are plausible.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        match self {
            Self::SensedImu => true,
            Self::ForceModel(coast) => coast.is_plausible(),
        }
    }
}

/// A velocity change that no sensor measures.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ImpulsiveBurn {
    /// The velocity change in m/s, in the frame axes.
    pub(crate) delta_v_frame_mps: [f64; 3],
    /// The covariance of the velocity change in (m/s)², in the frame
    /// axes.
    pub(crate) covariance: SymmetricCov3,
}

impl ImpulsiveBurn {
    /// Adds the velocity change to `core`, and its covariance to the
    /// velocity block.
    pub(crate) fn apply(&self, core: &mut InertialCore) {
        core.nominal.v += nalgebra::Vector3::from(self.delta_v_frame_mps);
        let mut block = core.p.fixed_view_mut::<3, 3>(VEL, VEL);
        block += cov3_to_matrix(&self.covariance);
    }
}

impl InertialNavigationFilter {
    /// The active propagation mode, or the mode of the next
    /// initialization.
    #[must_use]
    pub fn propagation_mode(&self) -> PropagationMode {
        self.core.as_ref().map_or(self.setup_mode, |core| core.mode)
    }

    /// Propagates the state to `at` in the active mode and then changes
    /// the mode to `mode`. A change of mode forgets the most recent IMU
    /// rates, so in the sensed-IMU mode the propagation waits for a new
    /// IMU sample. On an uninitialized filter, `mode` is the mode of the
    /// next initialization and `at` has no effect. A reset keeps the
    /// active mode.
    ///
    /// # Errors
    ///
    /// [`RejectionReason::NonFiniteValue`] for implausible settings,
    /// [`RejectionReason::AcquisitionTimeRegression`] for an `at` before
    /// the state time, and [`RejectionReason::NoInertialData`] for an
    /// `at` after the state time in the sensed-IMU mode before the first
    /// IMU sample.
    pub fn set_propagation_mode(
        &mut self,
        mode: PropagationMode,
        at: MonotonicNanos,
    ) -> Result<(), RejectionReason> {
        let result = (|| {
            if !mode.is_plausible() {
                return Err(RejectionReason::NonFiniteValue);
            }
            if self.core.is_none() {
                self.setup_mode = mode;
                return Ok(());
            }
            self.apply_state_event(Event::Mode(mode), at)
        })();
        self.record_failure(result)
    }

    /// Propagates the state to `at` with the force model and adds the
    /// velocity change `delta_v_frame_mps` in the frame axes. The
    /// covariance `delta_v_cov` of the change in (m/s)² adds to the
    /// velocity block.
    ///
    /// # Errors
    ///
    /// [`RejectionReason::NotInitialized`],
    /// [`RejectionReason::NonFiniteValue`] for a change that is not
    /// finite, [`RejectionReason::ImplausibleCovariance`],
    /// [`RejectionReason::AcquisitionTimeRegression`] for an `at` before
    /// the state time, and [`RejectionReason::PropagationModeMismatch`]
    /// in the sensed-IMU mode.
    pub fn apply_impulsive_burn(
        &mut self,
        delta_v_frame_mps: [f64; 3],
        delta_v_cov: SymmetricCov3,
        at: MonotonicNanos,
    ) -> Result<(), RejectionReason> {
        let result = (|| {
            if !delta_v_frame_mps.iter().all(|c| c.is_finite()) {
                return Err(RejectionReason::NonFiniteValue);
            }
            check_cov3(&delta_v_cov)?;
            let burn = ImpulsiveBurn {
                delta_v_frame_mps,
                covariance: delta_v_cov,
            };
            self.apply_state_event(Event::Burn(burn), at)
        })();
        self.record_failure(result)
    }

    /// Propagates the state to `at` with the force model. Each
    /// observation and each publication also propagates the state, so a
    /// coast is necessary only to move the state time without them.
    ///
    /// # Errors
    ///
    /// [`RejectionReason::NotInitialized`],
    /// [`RejectionReason::AcquisitionTimeRegression`] for an `at` before
    /// the state time, and [`RejectionReason::PropagationModeMismatch`]
    /// in the sensed-IMU mode.
    pub fn coast_to(&mut self, at: MonotonicNanos) -> Result<(), RejectionReason> {
        let result = self.apply_state_event(Event::Coast, at);
        self.record_failure(result)
    }

    /// Applies a state event at or after the state time.
    fn apply_state_event(
        &mut self,
        event: Event,
        at: MonotonicNanos,
    ) -> Result<(), RejectionReason> {
        let core = self.core.as_ref().ok_or(RejectionReason::NotInitialized)?;
        if at < core.state_at {
            return Err(RejectionReason::AcquisitionTimeRegression {
                last: core.state_at,
                got: at,
            });
        }
        self.apply_event(event, at).map(|_| ())
    }
}
