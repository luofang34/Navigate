//! Changes of the navigation frame and of the regime (ADR-0014,
//! ADR-0017).
//!
//! A [`FrameChange`] moves the state from the active frame `O` into a
//! new frame `N` at one time. `N` can have a different center, other
//! axes, and a different rotation rate. Thus one change goes from the
//! rotating body-fixed frame of a body to an inertial frame, or from an
//! inertial frame to the body-fixed frame of another body. With the
//! center `o` and its velocity `ȯ` in `O`, the rotation `C` from the axes
//! of `O` to the axes of `N`, and the angular rate `ω_rel` of `N`
//! relative to `O` in the axes of `O`, the change is exact:
//!
//! - `p_N = C·(p_O − o)`
//! - `v_N = C·(v_O − ȯ − ω_rel × (p_O − o))`
//! - `q_b→N = q_O→N ⊗ q_b→O`
//!
//! The empirical acceleration of the dynamic model compensation is a
//! force per unit mass in the frame axes, so it turns with the axes:
//! `a_N = C·a_O`.
//!
//! The Jacobian of the error state has `J_pp = C`, `J_vv = C`,
//! `J_vp = −C·[ω_rel×]`, and `J_aa = C` for the empirical acceleration.
//! The attitude error is in the body axes, each bias error is in the
//! body or sensor axes, and the clock does not depend on the frame, so
//! those blocks are the identity. The covariance is `P_N = J·P_O·Jᵀ`.

use nalgebra::{Const, DimMin, Matrix3, Vector3};
use navigate_contract::{AttitudeQuaternion, MonotonicNanos};
use navigate_geodesy::BodyEllipsoid;

use super::InertialNavigationFilter;
use super::estimate::InertialCore;
use super::layout::StateLayout;
use super::nominal::{ErrorMat, InertialNominal, POS, VEL, renormalized, to_unit_quaternion};
use super::regime::Regime;
use super::update::Event;
use crate::admission::is_unit_quaternion;
use crate::rejection::RejectionReason;
use navigate_astrodynamics::{GravityField, SpaceEnvironment};

/// One change of the navigation frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameChange {
    /// The environment of the new frame.
    pub new_env: SpaceEnvironment,
    /// The reference ellipsoid of the new frame, or `None` for a frame
    /// without geodetic models. Its frame must be the frame of
    /// `new_env`.
    pub new_surface_ellipsoid: Option<BodyEllipsoid>,
    /// Position of the center of the new frame in meters, in the active
    /// frame.
    pub origin_in_old_m: [f64; 3],
    /// Velocity of the center of the new frame in m/s, relative to the
    /// active frame, in the active axes.
    pub origin_velocity_in_old_mps: [f64; 3],
    /// Unit rotation from the active axes to the axes of the new frame.
    pub orientation_old_to_new: AttitudeQuaternion,
    /// Angular rate of the new axes relative to the active axes in
    /// rad/s, in the active axes.
    pub relative_angular_rate_in_old_rad_s: [f64; 3],
}

impl FrameChange {
    /// The change from the frame of `old_env` to the frame of `new_env`
    /// when the two frames rotate relative to one inertial space. The
    /// relative rate is `ω_rel = C_N→O·ω_N − ω_O`, with each rotation
    /// vector from [`GravityField::frame_rotation`] in the axes of its
    /// own frame.
    #[must_use]
    pub fn between_environments(
        old_env: &SpaceEnvironment,
        new_env: SpaceEnvironment,
        new_surface_ellipsoid: Option<BodyEllipsoid>,
        origin_in_old_m: [f64; 3],
        origin_velocity_in_old_mps: [f64; 3],
        orientation_old_to_new: AttitudeQuaternion,
    ) -> Self {
        let new_to_old = to_unit_quaternion(&orientation_old_to_new).inverse();
        let relative = new_to_old * new_env.frame_rotation() - old_env.frame_rotation();
        Self {
            new_env,
            new_surface_ellipsoid,
            origin_in_old_m,
            origin_velocity_in_old_mps,
            orientation_old_to_new,
            relative_angular_rate_in_old_rad_s: relative.into(),
        }
    }

    /// Whether every value is finite, the orientation is a unit
    /// quaternion, and the environment is plausible.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        let finite = self
            .origin_in_old_m
            .iter()
            .chain(&self.origin_velocity_in_old_mps)
            .chain(&self.relative_angular_rate_in_old_rad_s)
            .all(|c| c.is_finite());
        finite && is_unit_quaternion(&self.orientation_old_to_new) && self.new_env.is_plausible()
    }

    /// The rotation matrix `C` from the active axes to the new axes.
    fn rotation(&self) -> Matrix3<f64> {
        to_unit_quaternion(&self.orientation_old_to_new)
            .to_rotation_matrix()
            .into_inner()
    }
}

/// The nominal state in the new frame of `change`. The biases and the
/// clock do not change.
pub(crate) fn transformed_nominal(x: &InertialNominal, change: &FrameChange) -> InertialNominal {
    let c = change.rotation();
    let omega = Vector3::from(change.relative_angular_rate_in_old_rad_s);
    let dp = x.p - Vector3::from(change.origin_in_old_m);
    let dv = x.v - Vector3::from(change.origin_velocity_in_old_mps);
    InertialNominal {
        p: c * dp,
        v: c * (dv - omega.cross(&dp)),
        q: renormalized(to_unit_quaternion(&change.orientation_old_to_new) * x.q),
        a_dmc: c * x.a_dmc,
        ..*x
    }
}

/// The Jacobian of the error state for `change`.
pub(crate) fn frame_change_jacobian<const N: usize>(
    change: &FrameChange,
    layout: &StateLayout,
) -> ErrorMat<N> {
    let c = change.rotation();
    let omega = Vector3::from(change.relative_angular_rate_in_old_rad_s);
    let mut j = ErrorMat::<N>::identity();
    j.fixed_view_mut::<3, 3>(POS, POS).copy_from(&c);
    j.fixed_view_mut::<3, 3>(VEL, VEL).copy_from(&c);
    j.fixed_view_mut::<3, 3>(VEL, POS)
        .copy_from(&(-c * omega.cross_matrix()));
    if let Some(dmc) = layout.dmc() {
        j.fixed_view_mut::<3, 3>(dmc, dmc).copy_from(&c);
    }
    j
}

/// Moves `core` into the new frame of `change`, with its regime.
pub(crate) fn apply_frame_change<const N: usize>(core: &mut InertialCore<N>, change: &FrameChange) {
    core.nominal = transformed_nominal(&core.nominal, change);
    let mut j = frame_change_jacobian::<N>(change, &core.layout);
    let layout = core.layout;
    core.correlated
        .rotate(&change.rotation(), &layout, &mut core.nominal, &mut j);
    let moved = j * core.p * j.transpose();
    core.p = (moved + moved.transpose()) * 0.5;
    core.regime = Regime {
        env: change.new_env,
        surface: change.new_surface_ellipsoid,
    };
}

impl<const N: usize> InertialNavigationFilter<N>
where
    Const<N>: DimMin<Const<N>, Output = Const<N>>,
{
    /// Moves the state into the new frame of `change` at `at`. The change
    /// is a history event, so a delayed observation before `at` replays
    /// through it in the frame of its acquisition time.
    ///
    /// # Errors
    ///
    /// [`RejectionReason::NotInitialized`],
    /// [`RejectionReason::FrameMismatch`] for an ellipsoid that is not on
    /// the new frame, [`RejectionReason::NonFiniteValue`] for an
    /// implausible change, [`RejectionReason::AcquisitionTimeRegression`]
    /// for an `at` before the state time, and
    /// [`RejectionReason::NoInertialData`] for an `at` after the state
    /// time before the first IMU sample.
    pub fn change_frame(
        &mut self,
        change: FrameChange,
        at: MonotonicNanos,
    ) -> Result<(), RejectionReason> {
        let result = (|| {
            let core = self.core.as_ref().ok_or(RejectionReason::NotInitialized)?;
            if let Some(ellipsoid) = change.new_surface_ellipsoid {
                check_ellipsoid(&change.new_env, ellipsoid)?;
            }
            if !change.is_plausible() {
                return Err(RejectionReason::NonFiniteValue);
            }
            if at < core.state_at {
                return Err(RejectionReason::AcquisitionTimeRegression {
                    last: core.state_at,
                    got: at,
                });
            }
            self.apply_event(Event::FrameChange(change), at).map(|_| ())
        })();
        self.record_failure(result)
    }

    /// Moves the state into the frame of `new_env` at `at` without a
    /// reference ellipsoid, for example from `EARTH_J2000` to
    /// `MOON_J2000` at the sphere of influence. `origin_shift_m` and
    /// `velocity_shift_mps` are the position and the velocity of the
    /// center of `new_env` in the active frame. `orientation_old_to_new`
    /// rotates the active axes into the axes of `new_env`; `None` keeps
    /// the axes. The relative angular rate is zero, so the change is
    /// exact when the two frames have the same rotation rate. For frames
    /// with different rates, use [`Self::change_frame`] with
    /// [`FrameChange::between_environments`].
    ///
    /// # Errors
    ///
    /// The refusals of [`Self::change_frame`].
    pub fn recenter_space_frame(
        &mut self,
        new_env: SpaceEnvironment,
        origin_shift_m: [f64; 3],
        velocity_shift_mps: [f64; 3],
        orientation_old_to_new: Option<AttitudeQuaternion>,
        at: MonotonicNanos,
    ) -> Result<(), RejectionReason> {
        let change = FrameChange {
            new_env,
            new_surface_ellipsoid: None,
            origin_in_old_m: origin_shift_m,
            origin_velocity_in_old_mps: velocity_shift_mps,
            orientation_old_to_new: orientation_old_to_new
                .unwrap_or(AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0)),
            relative_angular_rate_in_old_rad_s: [0.0; 3],
        };
        self.change_frame(change, at)
    }

    /// Sets a different environment in the same frame, for example new
    /// third-body positions. The reference ellipsoid stays. On an
    /// initialized filter the change applies from the state time and is
    /// a history event.
    ///
    /// # Errors
    ///
    /// [`RejectionReason::FrameMismatch`] for a different frame and
    /// [`RejectionReason::NonFiniteValue`] for an implausible
    /// environment.
    pub fn update_space_environment(
        &mut self,
        env: SpaceEnvironment,
    ) -> Result<(), RejectionReason> {
        let result = (|| {
            let regime = self.active_regime();
            if env.frame != regime.frame() {
                return Err(RejectionReason::FrameMismatch {
                    expected: regime.frame(),
                    got: env.frame,
                });
            }
            if !env.is_plausible() {
                return Err(RejectionReason::NonFiniteValue);
            }
            self.set_regime(Regime { env, ..regime })
        })();
        self.record_failure(result)
    }

    /// Attaches the reference ellipsoid `ellipsoid` to the active frame,
    /// or removes it with `None`. With an ellipsoid the filter has the
    /// geodetic models and [`Self::tick`] publishes a solution. On an
    /// initialized filter the change applies from the state time and is
    /// a history event.
    ///
    /// # Errors
    ///
    /// [`RejectionReason::FrameMismatch`] for an ellipsoid that is not on
    /// the active frame, and [`RejectionReason::NonFiniteValue`] for an
    /// implausible ellipsoid.
    pub fn attach_surface_ellipsoid(
        &mut self,
        ellipsoid: Option<BodyEllipsoid>,
    ) -> Result<(), RejectionReason> {
        let result = (|| {
            let regime = self.active_regime();
            if let Some(ellipsoid) = ellipsoid {
                check_ellipsoid(&regime.env, ellipsoid)?;
            }
            self.set_regime(Regime {
                surface: ellipsoid,
                ..regime
            })
        })();
        self.record_failure(result)
    }

    /// The environment of the active frame.
    #[must_use]
    pub fn environment(&self) -> SpaceEnvironment {
        self.active_regime().env
    }

    /// The reference ellipsoid of the active frame, if any.
    #[must_use]
    pub fn surface_ellipsoid(&self) -> Option<BodyEllipsoid> {
        self.active_regime().surface
    }

    /// Sets `regime` on an uninitialized filter, or applies it at the
    /// state time as a history event.
    fn set_regime(&mut self, regime: Regime) -> Result<(), RejectionReason> {
        let Some(state_at) = self.core.as_ref().map(|core| core.state_at) else {
            self.setup = regime;
            return Ok(());
        };
        self.apply_event(Event::Regime(regime), state_at)
            .map(|_| ())
    }
}

/// Refuses an ellipsoid that is not on the frame of `env` or that is
/// not plausible.
fn check_ellipsoid(
    env: &SpaceEnvironment,
    ellipsoid: BodyEllipsoid,
) -> Result<(), RejectionReason> {
    if ellipsoid.frame() != env.frame {
        return Err(RejectionReason::FrameMismatch {
            expected: env.frame,
            got: ellipsoid.frame(),
        });
    }
    if !ellipsoid.contract().is_plausible() {
        return Err(RejectionReason::NonFiniteValue);
    }
    Ok(())
}
