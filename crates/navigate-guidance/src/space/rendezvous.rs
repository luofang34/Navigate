//! The rendezvous derivation in the LVLH frame of a target.

use navigate_contract::{
    CartesianSolution, ClockDomainId, FrameId, GuidanceCommand, GuidanceSetpoint, MonotonicNanos,
};
use navigate_trajectory::{RendezvousTarget, TrajectoryError, evaluate_rendezvous};

use crate::admission::admit_space_solution;
use crate::config::SpaceGuidanceConfig;
use crate::float_math::hypot;
use crate::refusal::GuidanceRefusal;

use super::{seconds_until, state_of};

/// One approach to a hold point near a target, in one inertial frame.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct RendezvousPlan {
    /// The inertial frame of the chaser and the target solutions.
    pub frame: FrameId,
    /// The hold point, the corridor, and the floor on the time to go.
    pub target: RendezvousTarget,
    /// The time at which the chaser must be at the hold point.
    pub arrival_at: MonotonicNanos,
}

impl RendezvousPlan {
    /// Builds a plan from its parts.
    #[must_use]
    pub const fn new(frame: FrameId, target: RendezvousTarget, arrival_at: MonotonicNanos) -> Self {
        Self {
            frame,
            target,
            arrival_at,
        }
    }
}

/// Derives one Cartesian velocity command that moves the `chaser` to the
/// hold point of `plan` near the `target`, or refuses with a typed
/// reason.
///
/// The command is [`GuidanceSetpoint::CartesianVelocity`] in
/// `plan.frame`. The corridor check uses the root sum square of the
/// one-sigma position uncertainties of the two solutions.
///
/// # Errors
///
/// [`GuidanceRefusal::TrajectoryInvalid`] when `plan.frame` is not
/// inertial, the refusals of [`crate::admit_space_solution`] for each
/// solution with `expected_frame = plan.frame`,
/// [`GuidanceRefusal::CorridorViolated`] when the chaser is outside the
/// approach cone in the keep-out sphere or the uncertainty does not fit
/// in the margin, and [`GuidanceRefusal::TrajectoryInvalid`] for every
/// other error of the law.
pub fn guide_rendezvous(
    chaser: &CartesianSolution,
    target: &CartesianSolution,
    plan: &RendezvousPlan,
    now: MonotonicNanos,
    now_clock: ClockDomainId,
    config: &SpaceGuidanceConfig,
) -> Result<GuidanceCommand, GuidanceRefusal> {
    if !plan.frame.is_inertial() {
        return Err(GuidanceRefusal::from_trajectory(
            TrajectoryError::NonInertialFrame { frame: plan.frame },
        ));
    }
    admit_space_solution(chaser, plan.frame, now, now_clock, config)?;
    admit_space_solution(target, plan.frame, now, now_clock, config)?;
    let relative_1sigma_m = hypot(
        chaser.integrity.uncertainty.max_axis_1sigma_m(),
        target.integrity.uncertainty.max_axis_1sigma_m(),
    );
    let command = evaluate_rendezvous(
        &state_of(target)?,
        &state_of(chaser)?,
        &plan.target,
        seconds_until(now, plan.arrival_at),
        relative_1sigma_m,
    )
    .map_err(GuidanceRefusal::from_trajectory)?;
    Ok(GuidanceCommand::new(
        now,
        GuidanceSetpoint::CartesianVelocity {
            frame: plan.frame,
            velocity_mps: command.velocity_mps.into(),
        },
        chaser.stamp,
    ))
}
