//! Changes of the space frame and of the space environment (ADR-0014).

use std::sync::Arc;

use nalgebra::{UnitQuaternion, Vector3};
use navigate_contract::{AttitudeQuaternion, MonotonicNanos};

use super::InertialNavigationFilter;
use super::nominal::to_unit_quaternion;
use super::regime::{Regime, Setup};
use super::update::{Event, Recenter};
use crate::admission::is_unit_quaternion;
use crate::environment::SpaceEnvironment;
use crate::rejection::RejectionReason;

impl InertialNavigationFilter {
    /// Moves the space state into the frame of `new_env` at `at`, for
    /// example from `EARTH_J2000` to `MOON_J2000` at the sphere of
    /// influence. `origin_shift_m` and `velocity_shift_mps` are the
    /// position and the velocity of the center of `new_env` in the active
    /// frame. `orientation_old_to_new` rotates the active axes into the
    /// axes of `new_env`; `None` keeps the axes. The transformation is exact when the two
    /// frames have the same rotation rate; the caller puts any transport
    /// term into `velocity_shift_mps`. The change is a history event, so
    /// a delayed observation before `at` replays through it.
    ///
    /// # Errors
    ///
    /// [`RejectionReason::NotInitialized`],
    /// [`RejectionReason::FrameMismatch`] in a surface filter,
    /// [`RejectionReason::NonFiniteValue`] for an implausible input,
    /// [`RejectionReason::AcquisitionTimeRegression`] for an `at` before
    /// the state time, and [`RejectionReason::NoInertialData`] for an
    /// `at` after the state time before the first IMU sample.
    pub fn recenter_space_frame(
        &mut self,
        new_env: SpaceEnvironment,
        origin_shift_m: [f64; 3],
        velocity_shift_mps: [f64; 3],
        orientation_old_to_new: Option<AttitudeQuaternion>,
        at: MonotonicNanos,
    ) -> Result<(), RejectionReason> {
        let result = (|| {
            let core = self.core.as_ref().ok_or(RejectionReason::NotInitialized)?;
            if core.regime.surface().is_some() {
                return Err(RejectionReason::FrameMismatch {
                    expected: core.regime.frame(),
                    got: new_env.frame,
                });
            }
            let finite = origin_shift_m
                .iter()
                .chain(velocity_shift_mps.iter())
                .all(|c| c.is_finite());
            let rotation_valid = orientation_old_to_new
                .as_ref()
                .is_none_or(is_unit_quaternion);
            if !(new_env.is_plausible() && finite && rotation_valid) {
                return Err(RejectionReason::NonFiniteValue);
            }
            if at < core.state_at {
                return Err(RejectionReason::AcquisitionTimeRegression {
                    last: core.state_at,
                    got: at,
                });
            }
            let recenter = Recenter {
                env: Arc::new(new_env),
                origin_shift: Vector3::from(origin_shift_m),
                velocity_shift: Vector3::from(velocity_shift_mps),
                rotation: orientation_old_to_new
                    .as_ref()
                    .map_or_else(UnitQuaternion::identity, to_unit_quaternion),
            };
            self.apply_event(Event::Recenter(recenter), at).map(|_| ())
        })();
        self.record_failure(result)
    }

    /// Sets a different space environment in the same frame, for example
    /// new third-body positions. The change applies from the state time
    /// and is a history event.
    ///
    /// # Errors
    ///
    /// [`RejectionReason::FrameMismatch`] for a different frame or a
    /// surface filter, and [`RejectionReason::NonFiniteValue`] for an
    /// implausible environment.
    pub fn update_space_environment(
        &mut self,
        env: SpaceEnvironment,
    ) -> Result<(), RejectionReason> {
        let result = (|| {
            let expected = self.frame();
            let is_space = matches!(self.setup, Setup::CartesianSpace(_))
                && self
                    .core
                    .as_ref()
                    .is_none_or(|core| matches!(core.regime, Regime::CartesianSpace(_)));
            if !is_space || env.frame != expected {
                return Err(RejectionReason::FrameMismatch {
                    expected,
                    got: env.frame,
                });
            }
            if !env.is_plausible() {
                return Err(RejectionReason::NonFiniteValue);
            }
            let env = Arc::new(env);
            let Some(state_at) = self.core.as_ref().map(|core| core.state_at) else {
                self.setup = Setup::CartesianSpace(env);
                return Ok(());
            };
            self.apply_event(Event::Environment(env), state_at)
                .map(|_| ())
        })();
        self.record_failure(result)
    }
}
